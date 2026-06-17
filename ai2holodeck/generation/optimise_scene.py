import os
import copy
import base64
import json
import re
import glob
import editdistance
from openai import OpenAI
from shapely.geometry import Polygon
import datetime
from dataclasses import dataclass

from ai2holodeck.generation.floor_objects import DFS_Solver_Floor
from ai2holodeck.generation.utils import get_top_down_frame, get_all_room_top_down_frames, get_room_images as get_room_corner_images
from ai2holodeck.constants import OBJATHOR_ASSETS_DIR


# ── Criteria & Action Registry ────────────────────────────────────────────────

@dataclass
class ActionConfig:
    name: str                  # "REMOVE", "REPOSITION", "ROTATE"
    prompt_description: str    # shown in the "decide whether it requires" block
    output_format: str         # shown in the "list each flagged object" block
    operator_class: str        # "remover", "rotator", "repositioner"

ACTION_REGISTRY: dict[str, ActionConfig] = {
    "REMOVE": ActionConfig(
        name="REMOVE",
        prompt_description="REMOVE: the object is wrong for the room and cannot be fixed by moving or rotating",
        output_format="REMOVE: <object description>, <reason>",
        operator_class="remover",
    ),
    "REPOSITION": ActionConfig(
        name="REPOSITION",
        prompt_description="REPOSITION: the object is right for the room but in the wrong location",
        output_format="REPOSITION: <object description>, <reason>",
        operator_class="repositioner",
    ),
    "ROTATE": ActionConfig(
        name="ROTATE",
        prompt_description="ROTATE: the object is in the right location but facing the wrong direction",
        output_format="ROTATE: <object description>, <reason>",
        operator_class="rotator",
    ),
}

@dataclass
class ViolationCriteria:
    name: str
    display_name: str
    description: str
    allowed_actions: list[str]   # keys into ACTION_REGISTRY

CRITERIA_REGISTRY: dict[str, ViolationCriteria] = {
    "object_orientation": ViolationCriteria(
        name="object_orientation",
        display_name="OBJECT ORIENTATION",
        description=(
            "- A seat (chair, stool, sofa, armchair) not facing its paired surface or the room focal point\n"
            "- An object with its back unnecessarily facing the centre of the room\n"
            "- A desk or workspace facing a wall with no reason for it"
        ),
        allowed_actions=["ROTATE"],
    ),
    "object_placement": ViolationCriteria(
        name="object_placement",
        display_name="OBJECT PLACEMENT",
        description=(
            "- An object in the wrong location for its function\n"
            "- An object not grouped with the items it belongs with (e.g. chair far from its table)\n"
            "- An object causing crowding or blocking access to another\n"
            "- An object pushed against the wrong wall or in an illogical corner\n"
            "- An object placed seemingly at random with no clear relationship to other objects\n"
            "- Multiple chairs belonging to a dining table that are not evenly distributed around it "
            "(e.g. all chairs on one side, or chairs scattered across the room rather than placed "
            "one per side or symmetrically around the table perimeter)"
        ),
        allowed_actions=["REMOVE", "REPOSITION"],
    ),
    "functional_zones": ViolationCriteria(
        name="functional_zones",
        display_name="FUNCTIONAL ZONE",
        description=(
            "- Related items not grouped together\n"
            "- Conflicting activities placed too close together\n"
            "- A zone that is incomplete (e.g. a desk with no chair, a dining table with no chairs)\n"
            "- Seating not arranged to serve its paired surface — chairs for a dining table should "
            "be distributed around it, not clustered on one side or left elsewhere in the room"
        ),
        allowed_actions=["REMOVE", "REPOSITION"],
    ),
    "object_selection": ViolationCriteria(
        name="object_selection",
        display_name="OBJECT SELECTION",
        description=(
            "- An object completely wrong for the room type\n"
            "- A duplicate object that adds no functional value\n"
            "- An object inappropriate for the original prompt: {prompt}"
        ),
        allowed_actions=["REMOVE"],
    ),
}

CRITERIA_PRESETS: dict[str, list[str]] = {
    "full":           list(CRITERIA_REGISTRY.keys()),
    "layout_only":    ["object_placement", "functional_zones"],
    "rotation_only":  ["object_orientation"],
    "selection_only": ["object_selection"],
}

# Fixed execution order — operators are skipped if not active
OPERATOR_RUN_ORDER = ["remover", "rotator", "repositioner"]

# Maps operator_class -> which flags list to pass and which parse key to read
OPERATOR_FLAG_KEY = {
    "remover":      "removal",
    "rotator":      "rotation",
    "repositioner": "reposition",
}


# ── Logger ────────────────────────────────────────────────────────────────────

class PromptLogger:
    def __init__(self, scene_folder: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.path.join(scene_folder, f"prompt_log_{timestamp}.txt")
        self.entry_count = 0
        with open(self.log_path, 'w') as f:
            f.write(f"Prompt Log — {timestamp}\n")
            f.write(f"Scene: {scene_folder}\n")
            f.write("=" * 60 + "\n\n")
        print(f"Logging prompts to: {self.log_path}")

    def log(self, label: str, prompt: str, response: str,
            room_id: str = None, attempt: int = None):
        self.entry_count += 1
        with open(self.log_path, 'a') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Entry #{self.entry_count}\n")
            f.write(f"Label: {label}\n")
            if room_id:
                f.write(f"Room: {room_id}\n")
            if attempt is not None:
                f.write(f"Attempt: {attempt}\n")
            f.write(f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"{'─' * 40}\n")
            f.write("PROMPT:\n")
            f.write(prompt + "\n")
            f.write(f"{'─' * 40}\n")
            f.write("RESPONSE:\n")
            f.write(response + "\n")


# ── LLM Wrapper ───────────────────────────────────────────────────────────────

class LLMWrapper:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o",
                 max_tokens: int = 2048):
        self.api_key = openai_api_key
        self.model = model_name
        self.max_tokens = max_tokens

    def _token_kwargs(self):
        use_new_param = self.model.startswith(("gpt-5", "o1", "o3", "o4"))
        return ({"max_completion_tokens": self.max_tokens}
                if use_new_param else {"max_tokens": self.max_tokens})

    def __call__(self, prompt: str) -> str:
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self._token_kwargs()
        )
        return response.choices[0].message.content

    def vision_multi(self, prompt: str, image_paths: list) -> str:
        client = OpenAI(api_key=self.api_key)
        content = []
        for path in image_paths:
            with open(path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"}
            })
        content.append({"type": "text", "text": prompt})
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            **self._token_kwargs()
        )
        return response.choices[0].message.content

    def vision(self, prompt: str, image_path: str) -> str:
        return self.vision_multi(prompt, [image_path])


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_evaluation(evaluation_text: str) -> tuple[int, list, list, list]:
    """Parse violation count and flags. Deduplicates by object id.
    Priority order: REMOVE > ROTATE > REPOSITION."""
    violation_count = 0
    removal_flags = []
    reposition_flags = []
    rotation_flags = []

    for line in evaluation_text.split('\n'):
        line = line.strip()

        m = re.match(r'VIOLATIONS:\s*(\d+)', line, re.IGNORECASE)
        if m:
            violation_count = int(m.group(1))

        m = re.match(r'REMOVE:\s*(.+),\s*(.+)', line, re.IGNORECASE)
        if m:
            removal_flags.append({"description": m.group(1).strip(),
                                   "reason": m.group(2).strip()})

        m = re.match(r'REPOSITION:\s*(.+),\s*(.+)', line, re.IGNORECASE)
        if m:
            reposition_flags.append({"description": m.group(1).strip(),
                                      "reason": m.group(2).strip()})

        m = re.match(r'ROTATE:\s*(.+),\s*(.+)', line, re.IGNORECASE)
        if m:
            rotation_flags.append({"description": m.group(1).strip(),
                                    "reason": m.group(2).strip()})

    def extract_object_id(description: str) -> str:
        return description.split()[0].lower()

    seen: set[str] = set()

    def deduplicate(flags: list) -> list:
        unique = []
        for flag in flags:
            obj_id = extract_object_id(flag["description"])
            if obj_id not in seen:
                seen.add(obj_id)
                unique.append(flag)
        return unique

    removal_flags    = deduplicate(removal_flags)
    rotation_flags   = deduplicate(rotation_flags)
    reposition_flags = deduplicate(reposition_flags)

    return violation_count, removal_flags, reposition_flags, rotation_flags


def clean_json(raw: str) -> str:
    return re.sub(r'^```json\s*|^```\s*|```$', '', raw, flags=re.MULTILINE).strip()


def parse_constraints(constraint_text: str, object_names: list,
                      fixed_object_names: list = None) -> dict:
    constraint_name2type = {
        "edge": "global", "middle": "global",
        "in front of": "relative", "behind": "relative",
        "left of": "relative", "right of": "relative",
        "side of": "relative", "around": "relative",
        "face to": "direction", "face same as": "direction",
        "aligned": "alignment", "center alignment": "alignment",
        "center aligned": "alignment", "aligned center": "alignment",
        "edge alignment": "alignment",
        "near": "distance", "far": "distance",
    }

    object_names_lower = [o.lower() for o in object_names]
    object2constraints = {}
    if fixed_object_names:
        for name in fixed_object_names:
            object2constraints[name.lower()] = []

    plans = [p.lower() for p in constraint_text.split("\n") if "|" in p]
    for plan in plans:
        plan = re.sub(r"^(\d+[\.\)]\s*|- )", "", plan)
        if plan and plan[-1] == ".":
            plan = plan[:-1]

        object_name = plan.split("|")[0].replace("*", "").strip()
        if object_name not in object_names_lower:
            continue

        object2constraints[object_name] = []
        for constraint in plan.split("|")[1:]:
            constraint = constraint.strip()
            constraint_name = constraint.split(",")[0].strip()
            if constraint_name == "n/a":
                continue

            try:
                constraint_type = constraint_name2type[constraint_name]
            except KeyError:
                _, constraint_name = min(
                    [(editdistance.eval(cn, constraint_name), cn)
                     for cn in constraint_name2type]
                )
                print(f"  Unknown constraint, using '{constraint_name}' instead.")
                constraint_type = constraint_name2type[constraint_name]

            if constraint_type == "global":
                object2constraints[object_name].append(
                    {"type": constraint_type, "constraint": constraint_name}
                )
            elif constraint_type in ["relative", "direction", "alignment", "distance"]:
                try:
                    target = constraint.split(",")[1].strip()
                except IndexError:
                    print(f"  Wrong format: {constraint}")
                    continue
                if target not in object2constraints:
                    print(f"  Target '{target}' not found — skipping")
                    continue
                if constraint_name == "around":
                    object2constraints[object_name].append(
                        {"type": "distance", "constraint": "near", "target": target})
                    object2constraints[object_name].append(
                        {"type": "direction", "constraint": "face to", "target": target})
                elif constraint_name == "in front of":
                    object2constraints[object_name].append(
                        {"type": "relative", "constraint": "in front of", "target": target})
                    object2constraints[object_name].append(
                        {"type": "alignment", "constraint": "center aligned", "target": target})
                else:
                    object2constraints[object_name].append(
                        {"type": constraint_type, "constraint": constraint_name, "target": target})

    # Deduplicate constraint types per object
    return {
        name: [c for i, c in enumerate(cs)
               if c["type"] not in [x["type"] for x in cs[:i]]]
        for name, cs in object2constraints.items()
    }


def get_door_window_obstacles(room: dict, doors: list, windows: list,
                               open_walls) -> dict:
    room_vertices = [(x * 100, y * 100) for x, y in room["vertices"]]
    room_poly = Polygon(room_vertices)
    obstacles = {}
    i = 0

    for door in doors:
        for door_box in door.get("doorBoxes", []):
            door_vertices = [(x * 100, z * 100) for x, z in door_box]
            door_poly = Polygon(door_vertices)
            door_center = door_poly.centroid
            if room_poly.contains(door_center):
                obstacles[f"door-{i}"] = (
                    (door_center.x, door_center.y), 0, door_vertices, 1)
                cx, cz = door_center.x, door_center.y
                clearance = 100
                obstacles[f"door-{i}_clearance"] = (
                    (cx, cz + clearance / 2), 0,
                    [(cx - clearance, cz), (cx + clearance, cz),
                     (cx + clearance, cz + clearance), (cx - clearance, cz + clearance)],
                    1)
                i += 1

    for window in windows:
        for window_box in window.get("windowBoxes", []):
            window_vertices = [(x * 100, z * 100) for x, z in window_box]
            window_poly = Polygon(window_vertices)
            window_center = window_poly.centroid
            if room_poly.contains(window_center):
                obstacles[f"window-{i}"] = (
                    (window_center.x, window_center.y), 0, window_vertices, 1)
                i += 1

    if open_walls and isinstance(open_walls, dict) and open_walls.get("openWallBoxes"):
        for open_wall_box in open_walls["openWallBoxes"]:
            open_wall_vertices = [(x * 100, z * 100) for x, z in open_wall_box]
            open_wall_poly = Polygon(open_wall_vertices)
            open_wall_center = open_wall_poly.centroid
            if room_poly.contains(open_wall_center):
                obstacles[f"open-{i}"] = (
                    (open_wall_center.x, open_wall_center.y), 0, open_wall_vertices, 1)
                i += 1

    return obstacles


def get_parent_id(small_obj_id: str) -> str | None:
    if "|" in small_obj_id:
        return small_obj_id.split("|", 1)[1]
    return None


def get_children(parent_id: str, scene: dict) -> list:
    return [o for o in scene.get("small_objects", [])
            if get_parent_id(o["id"]) == parent_id]


def move_children(parent_old: dict, parent_new: dict, scene: dict) -> None:
    dx = parent_new["position"]["x"] - parent_old["position"]["x"]
    dy = parent_new["position"]["y"] - parent_old["position"]["y"]
    dz = parent_new["position"]["z"] - parent_old["position"]["z"]
    for child in get_children(parent_old["id"], scene):
        child["position"]["x"] += dx
        child["position"]["y"] += dy
        child["position"]["z"] += dz
        print(f"  Moved child: {child['id']} by ({dx:.2f}, {dy:.2f}, {dz:.2f})")


def remove_children(parent_id: str, scene: dict) -> None:
    before = len(scene.get("small_objects", []))
    scene["small_objects"] = [
        o for o in scene.get("small_objects", [])
        if get_parent_id(o["id"]) != parent_id
    ]
    removed = before - len(scene["small_objects"])
    if removed > 0:
        print(f"  Removed {removed} small object(s) parented to '{parent_id}'")


def clean_scene_graph(scene: dict, removed_ids: set) -> None:
    scene_graph = scene.get("scene_graph", {})
    removed_short = {rid.split(" (")[0] for rid in removed_ids}

    for room_graph in scene_graph.values():
        for short_name in removed_short:
            if short_name in room_graph:
                del room_graph[short_name]
                print(f"  Scene graph: removed '{short_name}'")
        for obj_name, constraints in room_graph.items():
            before = len(constraints)
            room_graph[obj_name] = [
                c for c in constraints if c.get("target") not in removed_short
            ]
            n = before - len(room_graph[obj_name])
            if n > 0:
                print(f"  Scene graph: removed {n} dangling constraint(s) from '{obj_name}'")


def _fuzzy_validate_ids(candidates: list, room_objects: list,
                         id_key: str = "id") -> list:
    """Shared id validation with editdistance fallback used by all operators."""
    valid_ids = {o["id"] for o in room_objects}
    validated = []
    for item in candidates:
        raw_id = item[id_key]
        if raw_id in valid_ids:
            validated.append(item)
        else:
            best_id, best_dist = min(
                ((vid, editdistance.eval(raw_id.lower(), vid.lower()))
                 for vid in valid_ids),
                key=lambda x: x[1]
            )
            if best_dist <= 5:
                print(f"  Fuzzy matched '{raw_id}' -> '{best_id}' (dist={best_dist})")
                item[id_key] = best_id
                validated.append(item)
            else:
                print(f"  WARNING: Could not match '{raw_id}' — skipping")
    return validated


def render_room_images(candidate_scene: dict, room_id: str,
                       scene_folder: str, attempt: int | str) -> list:
    render_dir = os.path.join(scene_folder, f"attempt_{attempt}")
    os.makedirs(render_dir, exist_ok=True)

    top_image = get_top_down_frame(candidate_scene, OBJATHOR_ASSETS_DIR, 1024, 1024)
    ai2thor_path = os.path.join(render_dir, f"attempt_{attempt}_ai2thor.png")
    top_image.save(ai2thor_path)

    get_all_room_top_down_frames(
        scene=candidate_scene,
        objaverse_asset_dir=OBJATHOR_ASSETS_DIR,
        save_dir=render_dir,
        query_name=f"attempt_{attempt}",
    )

    room_corner_images = get_room_corner_images(
        candidate_scene, objaverse_asset_dir=OBJATHOR_ASSETS_DIR)
    for rname, images in room_corner_images.items():
        for i, img in enumerate(images):
            img.save(os.path.join(render_dir, f"{rname}_corner_{i}.png"))

    new_corner_images  = sorted(glob.glob(os.path.join(render_dir, '*corner*.png')))
    new_topdown_images = sorted(glob.glob(os.path.join(render_dir, '*topdown*.png')))
    new_ai2thor = glob.glob(os.path.join(render_dir, '*ai2thor*.png'))
    new_ai2thor = new_ai2thor[0] if new_ai2thor else ai2thor_path

    return get_room_images(room_id, new_corner_images, new_topdown_images, new_ai2thor)


# ── Operators ─────────────────────────────────────────────────────────────────

class SceneRemover:
    def __init__(self, llm: LLMWrapper, confidence_threshold: float = 0.8):
        self.llm = llm
        self.confidence_threshold = confidence_threshold

    def get_removal_candidates(self, evaluation_text: str, room_objects: list,
                               room_id: str, removal_flags: list = None,
                               logger: PromptLogger = None,
                               attempt: int = None) -> list:
        removal_flags = removal_flags or []
        flag_text = "\n".join(
            f"- {f['description']}: {f['reason']}" for f in removal_flags
        ) or "None"

        enriched = []
        for o in room_objects:
            pos = o.get("position", {})
            rot = o.get("rotation", {})
            enriched.append({
                "id": o["id"],
                "object_name": o.get("object_name", "").replace("-", " ").replace("_", " "),
                "position_x": round(pos.get("x", 0), 2),
                "position_z": round(pos.get("z", 0), 2),
                "rotation_y": round(rot.get("y", 0), 2),
            })

        prompt = f"""You are matching flagged objects from a room evaluation to their scene IDs.
The evaluator described objects using visual/spatial language (e.g. "chair by the bookcase",
"chair at bottom left corner"). Use the position data to identify which specific object
matches each description.

Flagged for removal (natural language descriptions from an evaluator):
{flag_text}

Scene objects (id, name, and position in metres):
{json.dumps(enriched, indent=2)}

Instructions:
- For each flagged description, find the BEST matching object id from the scene list above
- Use position_x and position_z to resolve spatial descriptions like "left", "corner", "by the window"
  — lower X is further left, lower Z is further toward the bottom of the room
- Use object_name to resolve type descriptions like "bookshelf", "chair"
- You MUST return one of the exact id strings from the scene list above, nothing else
- Only include an object if you are confident it matches the flagged description
- If no object in the scene matches a flagged description, omit it
- Do not invent ids or paraphrase them

Return JSON only:
{{
  "removals": [
    {{
      "id": "<exact id from scene list>",
      "matched_description": "<the flag text it matched>",
      "reason": "<why this id matches, referencing position if relevant>"
    }}
  ]
}}

Return {{"removals": []}} if nothing matches."""

        raw = self.llm(prompt)
        if logger:
            logger.log("SceneRemover", prompt, raw, room_id=room_id, attempt=attempt)

        try:
            candidates = json.loads(clean_json(raw))["removals"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARNING: Failed to parse removal candidates: {e}")
            return []

        return _fuzzy_validate_ids(candidates, room_objects)

    def apply(self, scene_objects: list, candidates: list) -> tuple:
        removal_ids = {c["id"] for c in candidates}
        kept    = [o for o in scene_objects if o["id"] not in removal_ids]
        removed = [o for o in scene_objects if o["id"] in removal_ids]
        return kept, removed


class SceneRotator:
    def __init__(self, llm: LLMWrapper):
        self.llm = llm

    def get_rotation_targets(self, evaluation_text: str, room_objects: list,
                             scene_graph: dict = None, room_id: str = None,
                             logger: PromptLogger = None, attempt: int = None,
                             rotation_flags: list = None) -> list:
        rotation_flags = rotation_flags or []
        scene_graph = scene_graph or {}

        flag_text = "\n".join(
            f"- {f['description']}: {f['reason']}" for f in rotation_flags
        ) or "None"

        enriched = []
        for o in room_objects:
            pos = o.get("position", {})
            rot = o.get("rotation", {})
            enriched.append({
                "id": o["id"],
                "name": o.get("object_name", "").replace("-", " ").replace("_", " "),
                "position_x": round(pos.get("x", 0), 2),
                "position_z": round(pos.get("z", 0), 2),
                "rotation_y": round(rot.get("y", 0), 2),
            })

        graph_lines = []
        for obj_name, constraints in scene_graph.items():
            face = [c for c in constraints
                    if c.get("constraint") == "face to" and "target" in c]
            if face:
                graph_lines.append(
                    f"{obj_name} → should face: {', '.join(c['target'] for c in face)}")
        graph_text = "\n".join(graph_lines) or "None"

        prompt = f"""You are analysing a 3D room scene. Your job is ONLY to fix object rotations.
Do NOT reposition objects — only change their facing direction.

Rotation reference:
- 0 = facing north (+Z direction)
- 90 = facing east (+X direction)
- 180 = facing south (-Z direction)
- 270 = facing west (-X direction)

Explicitly flagged for rotation fixes:
{flag_text}

Full evaluation:
{evaluation_text}

Original intended facing directions (from scene graph):
{graph_text}

Current object positions and rotations:
{json.dumps(enriched, indent=2)}

For each object that only needs its rotation fixed (not repositioned), provide:
1. The exact object id
2. What the current rotation problem is
3. The correct rotation in degrees (must be 0, 90, 180, or 270)

To determine the correct rotation:
- A chair should face its table — calculate which direction points toward the table's position
- An object facing a wall unnecessarily should face inward (toward room centre)
- Use position_x and position_z to calculate the correct facing direction

Return JSON only:
{{
  "rotations": [
    {{
      "id": "dining_chair-1 (dining room)",
      "issue": "facing away from dining table",
      "current_rotation": 195.8,
      "correct_rotation": 270
    }}
  ]
}}

Return {{"rotations": []}} if no rotation fixes are needed."""

        raw = self.llm(prompt)
        if logger:
            logger.log("SceneRotator", prompt, raw, room_id=room_id, attempt=attempt)

        try:
            targets = json.loads(clean_json(raw))["rotations"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARNING: Failed to parse rotation targets: {e}")
            return []

        return _fuzzy_validate_ids(targets, room_objects)

    def apply(self, room_objects: list, rotation_targets: list,
              scene: dict) -> list:
        rotation_map = {t["id"]: t["correct_rotation"] for t in rotation_targets}
        valid_rotations = {0, 90, 180, 270}
        updated = []
        for obj in room_objects:
            if obj["id"] in rotation_map:
                new_rot = rotation_map[obj["id"]]
                if new_rot not in valid_rotations:
                    new_rot = min(valid_rotations, key=lambda r: abs(r - new_rot))
                    print(f"  Snapped rotation to {new_rot} for {obj['id']}")
                updated_obj = obj.copy()
                updated_obj["rotation"] = {"x": 0, "y": new_rot, "z": 0}
                updated.append(updated_obj)
                print(f"Rotated: {obj['id']} {obj['rotation']['y']}° -> {new_rot}°")
            else:
                updated.append(obj)
        return updated


class SceneRepositioner:
    def __init__(self, llm: LLMWrapper):
        self.llm = llm

    def get_reposition_targets(self, evaluation_text: str, room_objects: list,
                               scene_graph: dict = None, room_id: str = None,
                               logger: PromptLogger = None, attempt: int = None,
                               reposition_flags: list = None) -> list:
        reposition_flags = reposition_flags or []
        scene_graph = scene_graph or {}

        flag_text = "\n".join(
            f"- {f['description']}: {f['reason']}" for f in reposition_flags
        ) or "None"

        enriched = []
        for o in room_objects:
            pos = o.get("position", {})
            rot = o.get("rotation", {})
            enriched.append({
                "id": o["id"],
                "name": o.get("object_name", "").replace("-", " ").replace("_", " "),
                "position_x": round(pos.get("x", 0), 2),
                "position_z": round(pos.get("z", 0), 2),
                "rotation_y": round(rot.get("y", 0), 2),
            })

        graph_lines = []
        for obj_name, constraints in scene_graph.items():
            parts = [obj_name]
            for c in constraints:
                if c["type"] == "global":
                    parts.append(c["constraint"])
                elif "target" in c:
                    parts.append(f"{c['constraint']},{c['target']}")
                else:
                    parts.append(c["constraint"])
            graph_lines.append(" | ".join(parts))
        graph_text = "\n".join(graph_lines) or "None"

        prompt = f"""You are analysing a 3D room scene. Your job is ONLY to identify which
objects need to be repositioned based on the evaluation. Do NOT generate new constraints —
the original design intent (scene graph) will be used to re-place them correctly.

Explicitly flagged for repositioning:
{flag_text}

Full evaluation:
{evaluation_text}

Original design intent (scene graph — shows intended spatial relationships):
{graph_text}

Current object positions:
{json.dumps(enriched, indent=2)}

For each object that needs repositioning, identify:
1. Which object it is (exact id)
2. What the problem is
3. Whether the original scene graph constraints should be KEPT or OVERRIDDEN
   - KEEP: the constraints are correct but the object ended up in the wrong place
   - OVERRIDE: the constraints themselves are wrong

Return JSON only:
{{
  "repositions": [
    {{
      "id": "dining_chair-1 (dining room)",
      "issue": "not facing dining table",
      "use_scene_graph": true,
      "override_constraints": []
    }},
    {{
      "id": "bar_cart-0 (dining room)",
      "issue": "blocking doorway, needs to move to different wall",
      "use_scene_graph": false,
      "override_constraints": ["edge", "far,dining_table-0 (dining room)"]
    }}
  ]
}}

Return {{"repositions": []}} if nothing needs repositioning."""

        raw = self.llm(prompt)
        if logger:
            logger.log("SceneRepositioner.get_reposition_targets", prompt, raw,
                       room_id=room_id, attempt=attempt)

        try:
            targets = json.loads(clean_json(raw))["repositions"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARNING: Failed to parse reposition targets: {e}")
            return []

        return _fuzzy_validate_ids(targets, room_objects)

    def _to_solver_format(self, obj: dict) -> tuple:
        pos = obj["position"]
        if "vertices" in obj and obj["vertices"]:
            bbox = obj["vertices"]
            xs = [v[0] for v in bbox]
            if (max(xs) - min(xs)) < 5:
                bbox = [(v[0] * 100, v[1] * 100) for v in bbox]
        else:
            x, z = pos["x"] * 100, pos["z"] * 100
            bbox = [(x - 25, z - 25), (x + 25, z - 25),
                    (x + 25, z + 25), (x - 25, z + 25)]
        return ((pos["x"] * 100, pos["z"] * 100), obj["rotation"]["y"], bbox, 1)

    def _solution_to_scene_format(self, object_id: str, solution: tuple,
                                  original_obj: dict) -> dict:
        center, rotation, bbox, _ = solution
        updated = original_obj.copy()
        updated["position"] = {
            "x": center[0] / 100,
            "y": original_obj["position"]["y"],
            "z": center[1] / 100,
        }
        updated["rotation"] = {"x": 0, "y": rotation, "z": 0}
        updated["vertices"] = list(bbox)
        return updated

    def _order_by_size(self, reposition_targets: list, room_objects: list) -> list:
        obj_lookup = {o["id"]: o for o in room_objects}

        def get_area(target):
            obj = obj_lookup.get(target["id"])
            if not obj or "vertices" not in obj or not obj["vertices"]:
                return 0
            verts = obj["vertices"]
            xs, zs = [v[0] for v in verts], [v[1] for v in verts]
            return (max(xs) - min(xs)) * (max(zs) - min(zs))

        return sorted(reposition_targets, key=get_area, reverse=True)

    def apply(self, scene, room, reposition_targets, room_objects, attempt=0):
        room_id = room["id"]
        scene_graph = scene.get("scene_graph", {})
        room_graph = scene_graph.get(room_id, {})

        reposition_targets = self._order_by_size(reposition_targets, room_objects)
        target_ids = {t["id"] for t in reposition_targets}

        fixed_objects = {
            obj["id"]: self._to_solver_format(obj)
            for obj in room_objects if obj["id"] not in target_ids
        }
        fixed_objects.update(get_door_window_obstacles(
            room, scene.get("doors", []),
            scene.get("windows", []),
            scene.get("open_walls", {})
        ))

        all_constraint_lines = []
        for target in reposition_targets:
            short_name = target["id"].split(" (")[0]
            use_scene_graph = target.get("use_scene_graph", True)
            override = target.get("override_constraints", [])

            if use_scene_graph and short_name in room_graph:
                original = room_graph[short_name]
                parts = [target["id"]]
                for c in original:
                    if c["type"] == "global":
                        parts.append(c["constraint"])
                    elif "target" in c:
                        full_target = next(
                            (o["id"] for o in room_objects
                             if o["id"].split(" (")[0] == c["target"]),
                            c["target"]
                        )
                        parts.append(f"{c['constraint']},{full_target}")
                    else:
                        parts.append(c["constraint"])
                constraint_line = " | ".join(parts)
                print(f"  Scene graph constraints: {target['id']}")
            elif override:
                constraint_line = f"{target['id']} | {' | '.join(override)}"
                print(f"  Override constraints: {target['id']}: {override}")
            else:
                constraint_line = f"{target['id']} | edge"
                print(f"  Default edge constraint: {target['id']}")

            all_constraint_lines.append(constraint_line)

        all_object_names = [o.lower() for o in list(target_ids) + list(fixed_objects.keys())]
        parsed_all = parse_constraints(
            "\n".join(all_constraint_lines),
            all_object_names,
            fixed_object_names=list(fixed_objects.keys())
        )
        constraints = {
            target["id"]: parsed_all.get(
                target["id"].lower(),
                [{"type": "global", "constraint": "edge"}]
            )
            for target in reposition_targets
        }

        room_poly = Polygon([(x * 100, y * 100) for x, y in room["vertices"]])
        objects_to_place = []
        for target in reposition_targets:
            obj = next((o for o in room_objects if o["id"] == target["id"]), None)
            if obj and "vertices" in obj and obj["vertices"]:
                verts = obj["vertices"]
                xs, zs = [v[0] for v in verts], [v[1] for v in verts]
                dim = (max(xs) - min(xs), max(zs) - min(zs))
                objects_to_place.append((obj["id"], dim))
                print(f"  Queued: {obj['id']} dim={dim}")
            else:
                print(f"  WARNING: No vertices for {target['id']} — skipping")

        if not objects_to_place:
            print(f"WARNING: No valid dimensions found in {room_id}")
            return room_objects

        solver = DFS_Solver_Floor(
            grid_size=20, max_duration=15,
            random_seed=attempt if attempt is not None else 0
        )
        solution = solver.get_solution(
            room_poly, objects_to_place, constraints,
            initial_state=fixed_objects
        )

        updated_objects = []
        for obj in room_objects:
            if obj["id"] in solution:
                updated_obj = self._solution_to_scene_format(
                    obj["id"], solution[obj["id"]], obj)
                move_children(obj, updated_obj, scene)
                updated_objects.append(updated_obj)
                print(f"Repositioned: {obj['id']} "
                      f"{obj['position']} -> {updated_obj['position']}")
            else:
                updated_objects.append(obj)

        return updated_objects


# ── Room Image Lookup ─────────────────────────────────────────────────────────

def get_room_images(room_id: str, all_corner_images: list,
                    all_topdown_images: list, ai2thor_image: str) -> list:
    room_name_clean = room_id.replace(' ', '_')
    room_corners = sorted([
        p for p in all_corner_images
        if room_id.lower() in os.path.basename(p).lower()
        or room_name_clean.lower() in os.path.basename(p).lower()
    ])
    room_topdown = [
        p for p in all_topdown_images
        if room_id.lower() in os.path.basename(p).lower()
        or room_name_clean.lower() in os.path.basename(p).lower()
    ]
    return (room_topdown if room_topdown else [ai2thor_image]) + room_corners


# ── Evaluation ────────────────────────────────────────────────────────────────

def score_room(
    llm: LLMWrapper,
    room_id: str,
    room_images: list,
    prompt: str,
    room_objects: list = None,
    logger: PromptLogger = None,
    attempt: int = None,
    criteria: list[str] | None = None,
) -> tuple[float, str]:
    """Score the room on active criteria. Returns (overall_score, raw_response)."""

    if criteria is None:
        active_criteria = list(CRITERIA_REGISTRY.values())
    else:
        active_criteria = [CRITERIA_REGISTRY[c] for c in criteria
                           if c in CRITERIA_REGISTRY]

    score_block = "\n".join(
        f"{c.display_name.lower()} | [1-5] | [rationale]"
        for c in active_criteria
    )

    spatial_context = _build_spatial_context(room_id, room_objects)

    score_prompt = f"""You are an experienced interior designer and space planner. Evaluate the provided images focusing only on the {room_id}.
{spatial_context}

Scoring criteria for {room_id} — apply strictly, do not round up:
- 5: Flawless. Every object is in the optimal position for its function. Sightlines, 
     traffic flow, and groupings are all correct. A professional would make no changes.
- 4: One or two objects are slightly off but the room is fundamentally sound. 
     Issues are cosmetic and do not affect how the room is used.
- 3: Clear problems that a non-expert would notice. At least one functional zone is 
     compromised, a circulation path is partially blocked, or a key grouping is wrong.
- 2: The room has serious layout problems. Multiple objects are in the wrong place, 
     access is obstructed, or the arrangement contradicts the room's purpose.
- 1: The layout is dysfunctional. Objects are placed randomly, zones make no sense, 
     or the arrangement would be unusable in practice.

Scoring rules:
- Score what you see, not what you assume. If an object looks misplaced, it is misplaced.
- A single egregious violation (e.g. all dining chairs on one side of the table, a sofa 
  facing a wall, a desk with no chair) is enough to drop a criterion to 2 or below.
- Do not give a 4 or 5 if any object is clearly in the wrong position.
- Do not give a 3 if the problem is merely aesthetic — 3 means functional impact.
- Err toward lower scores when uncertain. A generous score on a bad room causes 
  incorrect fixes to be skipped entirely.

Score each criterion in this exact format:
{score_block}

Then on a new line:
OVERALL: <average of the above scores, one decimal>

Be honest — if the room is well arranged, score it highly. Only score low if there are clear, 
specific problems visible in the images. Do not assume violations exist.

Answer without additional text at the beginning or end."""

    response = llm.vision_multi(score_prompt, room_images)
    if logger:
        logger.log("score_room", score_prompt, response,
                   room_id=room_id, attempt=attempt)

    overall_score = 5.0
    m = re.search(r'OVERALL:\s*([0-9]+(?:\.[0-9]+)?)', response, re.IGNORECASE)
    if m:
        overall_score = float(m.group(1))

    return overall_score, response


def evaluate_room(
    llm: LLMWrapper,
    room_id: str,
    room_images: list,
    prompt: str,
    room_objects: list = None,
    logger: PromptLogger = None,
    attempt: int = None,
    criteria: list[str] | None = None,
) -> tuple[int, list, list, list, str]:
    """Identify violations in a room that has already failed the score threshold.
    Returns (violation_count, removal_flags, reposition_flags, rotation_flags, evaluation_text)."""

    if criteria is None:
        active_criteria = list(CRITERIA_REGISTRY.values())
    else:
        active_criteria = [CRITERIA_REGISTRY[c] for c in criteria
                           if c in CRITERIA_REGISTRY]

    active_actions: set[str] = set()
    for c in active_criteria:
        active_actions.update(c.allowed_actions)

    criteria_block = ""
    for c in active_criteria:
        desc = c.description.replace("{prompt}", prompt)
        criteria_block += f"{c.display_name} violations:\n{desc}\n\n"

    action_block = "\n".join(
        f"- {ACTION_REGISTRY[a].prompt_description}"
        for a in ["REMOVE", "REPOSITION", "ROTATE"]
        if a in active_actions
    )
    output_block = "\n".join(
        ACTION_REGISTRY[a].output_format
        for a in ["REMOVE", "REPOSITION", "ROTATE"]
        if a in active_actions
    )
    rotate_note = (
        "- ROTATE is for objects in the correct position but wrong orientation "
        "(e.g. chair facing wall instead of table)\n"
        if "ROTATE" in active_actions else ""
    )

    spatial_context = _build_spatial_context(room_id, room_objects)

    eval_prompt = f"""You are an experienced interior designer and space planner. Evaluate the provided images focusing only on the {room_id}.
{spatial_context}

This room has been flagged as needing improvement. Identify every violation of good interior 
design practice. A violation is any of the following:

{criteria_block}

For each violation found, decide whether it requires:
{action_block}

After listing all violations, output the total count on its own line in this exact format:
VIOLATIONS: <number>

Then list each flagged object:
{output_block}

IMPORTANT:
- Do not flag the same object twice
- Do not flag objects that are correctly placed
{rotate_note}- Be specific — use the exact object id from the list above in your descriptions
  (e.g. "dining_chair-2 (dining room) at (1.1, 0.8)" not just "chair on the left")
- If there are no violations, output VIOLATIONS: 0
- If removing an object would uphold the rooms aesthetic, remove it rather than reposition it
- Do not consider small objects

Answer without additional text at the beginning or end."""

    evaluation = llm.vision_multi(eval_prompt, room_images)
    if logger:
        logger.log("evaluate_room", eval_prompt, evaluation,
                   room_id=room_id, attempt=attempt)

    violation_count, removal_flags, reposition_flags, rotation_flags = \
        parse_evaluation(evaluation)

    return violation_count, removal_flags, reposition_flags, rotation_flags, evaluation

def _build_spatial_context(room_id: str, room_objects: list | None) -> str:
    if not room_objects:
        return ""
    enriched = []
    for o in room_objects:
        pos = o.get("position", {})
        rot = o.get("rotation", {})
        enriched.append({
            "id": o["id"],
            "name": o.get("object_name", "").replace("-", " ").replace("_", " "),
            "position_x": round(pos.get("x", 0), 2),
            "position_z": round(pos.get("z", 0), 2),
            "rotation_y": round(rot.get("y", 0), 2),
        })
    return (
        f"\nThe following objects are present in the {room_id} with their exact positions "
        f"(in metres) and current rotation (Y axis, degrees). Use these to make precise "
        f"identifications in your flags rather than vague descriptions. "
        f"Reference the exact id when flagging an object.\n\n"
        f"{json.dumps(enriched, indent=2)}\n\n"
        f"Rotation reference: 0 = facing north (+Z), 90 = facing east (+X), "
        f"180 = facing south (-Z), 270 = facing west (-X)\n"
    )

# ── Single Improvement Pass ───────────────────────────────────────────────────

def improve_room(
    scene,
    room,
    evaluation,
    removal_flags,
    reposition_flags,
    rotation_flags,
    operators: dict,                     # {"remover": ..., "rotator": ..., "repositioner": ...}
    criteria: list[str] | None = None,   # controls which operators actually run
    logger=None,
    attempt=None,
):
    scene_copy = copy.deepcopy(scene)
    room_id = room["id"]
    scene_graph = scene_copy.get("scene_graph", {})
    room_graph = scene_graph.get(room_id, {})

    floor_objects = {
        obj['id']: obj for obj in scene_copy['objects']
        if "|" not in obj['id']
    }
    small_objects_in_scene = [
        obj for obj in scene_copy['objects'] if "|" in obj['id']
    ]
    room_objects = [
        o for o in floor_objects.values() if o.get("roomId") == room_id
    ]

    # Derive which operator classes are needed from the active criteria
    if criteria is None:
        active_criteria = list(CRITERIA_REGISTRY.values())
    else:
        active_criteria = [CRITERIA_REGISTRY[c] for c in criteria
                           if c in CRITERIA_REGISTRY]

    active_operator_keys: set[str] = set()
    for c in active_criteria:
        for action_name in c.allowed_actions:
            active_operator_keys.add(ACTION_REGISTRY[action_name].operator_class)

    # Only run operators that are both active and provided
    run_order = [k for k in OPERATOR_RUN_ORDER
                 if k in active_operator_keys and k in operators]

    flags_by_operator = {
        "remover":      removal_flags,
        "rotator":      rotation_flags,
        "repositioner": reposition_flags,
    }

    # Track repositioned ids so the rotator skips them
    repositioned_ids: set[str] = set()

    for operator_key in run_order:
        operator = operators[operator_key]
        flags = flags_by_operator[operator_key]
        remaining_ids = {o["id"] for o in room_objects}

        print(f"\n  --- {operator_key.upper()} pass ---")

        if operator_key == "remover":
            candidates = operator.get_removal_candidates(
                evaluation, room_objects, room_id, flags,
                logger=logger, attempt=attempt
            )
            print(f"  Removal candidates: {[c['id'] for c in candidates]}")
            room_objects, removed = operator.apply(room_objects, candidates)

            removed_ids = {obj["id"] for obj in removed}
            for obj in removed:
                target_id = obj["id"]
                # Normalise to actual key (case-insensitive fallback)
                if target_id not in floor_objects:
                    target_id = next(
                        (k for k in floor_objects if k.lower() == target_id.lower()),
                        None
                    )
                if target_id:
                    del floor_objects[target_id]
                    remove_children(target_id, scene_copy)
                    small_objects_in_scene = [
                        o for o in small_objects_in_scene
                        if get_parent_id(o["id"]) != target_id
                    ]
                    print(f"  Deleted: {target_id}")
                else:
                    print(f"  WARNING: '{obj['id']}' not found — skipping")

            clean_scene_graph(scene_copy, removed_ids)

        elif operator_key == "rotator":
            targets = operator.get_rotation_targets(
                evaluation, room_objects,
                scene_graph=room_graph,
                room_id=room_id, logger=logger, attempt=attempt,
                rotation_flags=flags
            )
            targets = [t for t in targets
                       if t["id"] in remaining_ids
                       and t["id"] not in repositioned_ids]
            print(f"  Rotating: {[t['id'] for t in targets]}")
            if targets:
                room_objects = operator.apply(room_objects, targets, scene_copy)

        elif operator_key == "repositioner":
            targets = operator.get_reposition_targets(
                evaluation, room_objects,
                scene_graph=room_graph,
                room_id=room_id, logger=logger, attempt=attempt,
                reposition_flags=flags
            )
            targets = [t for t in targets if t["id"] in remaining_ids]
            print(f"  Repositioning: {[t['id'] for t in targets]}")
            if targets:
                room_objects = operator.apply(
                    scene_copy, room, targets, room_objects,
                    attempt=attempt or 0
                )
                repositioned_ids.update(t["id"] for t in targets)

    for obj in room_objects:
        floor_objects[obj["id"]] = obj

    scene_copy['objects'] = list(floor_objects.values()) + small_objects_in_scene
    return scene_copy


# ── Persistence ───────────────────────────────────────────────────────────────

def save_scene(scene: dict, scene_path: str, suffix: str) -> str:
    base, ext = os.path.splitext(scene_path)
    out_path = f"{base}_{suffix}{ext}"
    scene_to_save = copy.deepcopy(scene)
    scene_to_save['objects'] = [
        o for o in scene_to_save['objects'] if "|" not in o['id']
    ]
    with open(out_path, 'w') as f:
        json.dump(scene_to_save, f, indent=2)
    print(f"  Saved: {out_path}")
    return out_path


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_room_check(scene_folder, model,
                   criteria: list[str] | None = None):
    api_key = ''
    llm       = LLMWrapper(openai_api_key=api_key, model_name=model, max_tokens=2048)
    llm_large = LLMWrapper(openai_api_key=api_key, model_name=model, max_tokens=16384)

    PASS_VIOLATION_THRESHOLD = 2
    NUM_ATTEMPTS = 3
    SCORE_THRESHOLD = 4

    # ── Load scene ────────────────────────────────────────────────────────────
    scene_path = [
        f for f in glob.glob(os.path.join(scene_folder, '*.json'))
        if 'improved' not in f
    ][0]
    ai2thor_image      = glob.glob(os.path.join(scene_folder, '*ai2thor*.png'))[0]
    all_corner_images  = sorted(glob.glob(os.path.join(scene_folder, '*corner*.png')))
    all_topdown_images = sorted(glob.glob(os.path.join(scene_folder, '*topdown*.png')))

    print(f"Scene: {scene_path}")
    folder_name = os.path.basename(scene_folder)
    prompt = ' '.join(folder_name.split('-')[:-7]).replace('_', ' ')

    with open(scene_path, 'r') as f:
        scene = json.load(f)

    logger = PromptLogger(scene_folder)

    # Build operator dict — only instantiate what the criteria actually need
    operators = {
        "remover":      SceneRemover(llm),
        "rotator":      SceneRotator(llm_large),
        "repositioner": SceneRepositioner(llm_large),
    }

    best_scene = copy.deepcopy(scene)

    for room in scene['rooms']:
        room_id     = room['id']
        room_slug   = room_id.replace(' ', '_')
        room_images = get_room_images(
            room_id, all_corner_images, all_topdown_images, ai2thor_image)

        print(f"\n{'=' * 60}")
        print(f"EVALUATING: {room_id.upper()}")
        print(f"{'=' * 60}")

        room_objects = [
            o for o in scene['objects']
            if o.get("roomId") == room_id and "|" not in o["id"]
        ]

        # ── Multiple evaluation + fix attempts, each rendered out ────────────
        attempts_results = []

        for attempt in range(1, NUM_ATTEMPTS + 1):
            print(f"\n--- {room_id.upper()} Attempt {attempt}/{NUM_ATTEMPTS} ---")

            # Stage 1: score only
            overall_score, score_response = score_room(
                llm, room_id, room_images, prompt,
                room_objects=room_objects, logger=logger,
                attempt=attempt, criteria=criteria
            )
            print(f"  Overall score: {overall_score}")
            print(score_response)

            if overall_score >= SCORE_THRESHOLD:
                print(f"  ✓ Room scored {overall_score} — no fixes needed")
                attempts_results.append({
                    "attempt":         attempt,
                    "violation_count": 0,
                    "overall_score":   overall_score,
                    "scene":           copy.deepcopy(best_scene),
                    "render_dir":      None,
                })
                continue

            # Stage 2: only reached if score failed
            print(f"  Score {overall_score} below threshold — identifying violations...")
            violation_count, removal_flags, reposition_flags, rotation_flags, evaluation = \
                evaluate_room(llm, room_id, room_images, prompt,
                            room_objects=room_objects, logger=logger,
                            attempt=attempt, criteria=criteria)
            print(f"  Violations found: {violation_count}")
            print(evaluation)

            if violation_count <= PASS_VIOLATION_THRESHOLD:
                print(f"  ✓ Attempt {attempt} found no significant violations — skipping fix")
                attempts_results.append({
                    "attempt":          attempt,
                    "violation_count":  violation_count,
                    "scene":            copy.deepcopy(best_scene),
                    "render_dir":       None,
                })
                continue

            try:
                candidate_scene = improve_room(
                    copy.deepcopy(best_scene),
                    room,
                    evaluation,
                    removal_flags,
                    reposition_flags,
                    rotation_flags,
                    operators=operators,
                    criteria=criteria,
                    logger=logger,
                    attempt=attempt,
                )

                save_scene(candidate_scene, scene_path,
                        f"{room_slug}_attempt{attempt}")

                improved_images = render_room_images(
                    candidate_scene, room_id, scene_folder, f"{room_slug}_attempt{attempt}"
                )
                print(f"  Rendered attempt {attempt} to: "
                    f"{os.path.join(scene_folder, f'{room_slug}_attempt{attempt}')}")

                attempts_results.append({
                    "attempt":         attempt,
                    "violation_count": violation_count,
                    "scene":           candidate_scene,
                    "render_dir":      improved_images,
                })

            except Exception as e:
                print(f"  Attempt {attempt} failed: {e}")
                continue

        if not attempts_results:
            print(f"  All attempts failed for {room_id} — keeping original")
            continue

        # ── Pick best from rendered attempts ─────────────────────────────────
        # Default: lowest violation count from the evaluation that drove each attempt.
        # You can swap this for a manual or LLM-based selection over the rendered images.
        best_result = min(attempts_results, key=lambda r: r["violation_count"])
        print(f"\n  Selected attempt {best_result['attempt']} "
            f"({best_result['violation_count']} violations)")
        best_scene = best_result["scene"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate and improve room layouts in a generated scene."
    )
    parser.add_argument(
        "scene_folder",
        type=str,
        help="Path to the scene folder containing the .json scene file and rendered images."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)."
    )
    parser.add_argument(
        "--criteria",
        type=str,
        nargs="+",
        choices=list(CRITERIA_REGISTRY.keys()) + list(CRITERIA_PRESETS.keys()),
        default=None,
        help=(
            "Criteria to check. Pass individual names or a preset. "
            f"Presets: {list(CRITERIA_PRESETS.keys())}. "
            "Default: all criteria."
        ),
    )

    args = parser.parse_args()

    if not os.path.isdir(args.scene_folder):
        print(f"ERROR: '{args.scene_folder}' is not a valid directory.")
        exit(1)

    # Expand a preset to its individual criteria names if one was passed
    resolved_criteria = None
    if args.criteria:
        resolved_criteria = []
        for name in args.criteria:
            if name in CRITERIA_PRESETS:
                resolved_criteria.extend(CRITERIA_PRESETS[name])
            else:
                resolved_criteria.append(name)
        resolved_criteria = list(dict.fromkeys(resolved_criteria))  # preserve order, dedup

    run_room_check(args.scene_folder, args.model, criteria=resolved_criteria)