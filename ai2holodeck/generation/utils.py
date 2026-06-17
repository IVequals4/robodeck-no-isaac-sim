import copy
import os
from argparse import ArgumentParser
from typing import Dict, Any
import json
import base64

import compress_json
import numpy as np
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner
from moviepy.editor import (
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
    ImageSequenceClip,
)
from tqdm import tqdm

from ai2holodeck.constants import HOLODECK_BASE_DATA_DIR, THOR_COMMIT_ID

from openai import OpenAI


class LLMWrapper:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-2024-05-13", max_tokens: int = 2048):
        self.api_key = openai_api_key
        self.model = model_name
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> str:
        client = OpenAI(api_key=self.api_key)
        use_new_param = self.model.startswith(("gpt-5", "o1", "o3", "o4"))
        token_kwargs = (
            {"max_completion_tokens": self.max_tokens}
            if use_new_param
            else {"max_tokens": self.max_tokens}
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **token_kwargs,
        )
        return response.choices[0].message.content

    def vision(self, prompt: str, image_path: str) -> str:
        """Same model, but with a base64 image attached."""
        import base64
        client = OpenAI(api_key=self.api_key)
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        use_new_param = self.model.startswith(("gpt-5", "o1", "o3", "o4"))
        token_kwargs = (
            {"max_completion_tokens": self.max_tokens}
            if use_new_param
            else {"max_tokens": self.max_tokens}
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            **token_kwargs,
        )
        return response.choices[0].message.content


def all_edges_white(img):
    # Define a white pixel
    white = [255, 255, 255]

    # Check top edge
    if not np.all(np.all(img[0, :] == white, axis=-1)):
        return False
    # Check bottom edge
    if not np.all(np.all(img[-1, :] == white, axis=-1)):
        return False
    # Check left edge
    if not np.all(np.all(img[:, 0] == white, axis=-1)):
        return False
    # Check right edge
    if not np.all(np.all(img[:, -1] == white, axis=-1)):
        return False

    # If all the conditions met
    return True


def get_top_down_frame(scene, objaverse_asset_dir, width=1024, height=1024):
    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=1.5,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=90,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
    )

    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]

    pose["fieldOfView"] = 60
    pose["position"]["y"] = bounds["y"]
    del pose["orthographicSize"]

    try:
        wall_height = wall_height = max(
            [point["y"] for point in scene["walls"][0]["polygon"]]
        )
    except:
        wall_height = 2.5

    for i in range(20):
        pose["orthographic"] = False

        pose["farClippingPlane"] = pose["position"]["y"] + 10
        pose["nearClippingPlane"] = pose["position"]["y"] - wall_height

        # add the camera to the scene
        event = controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_frame = event.third_party_camera_frames[-1]

        # check if the edge of the frame is white
        if all_edges_white(top_down_frame):
            break

        pose["position"]["y"] += 0.75

    controller.stop()
    image = Image.fromarray(top_down_frame)

    return image

import copy
from PIL import Image

def get_top_down_frame_for_objects(scene_base, objects_to_include, objaverse_asset_dir, width=1024, height=1024):
    """
    Renders a top-down frame with only the specified objects included.
    scene_base: the full scene dict (rooms, walls, doors, windows, lights etc.)
    objects_to_include: list of object dicts to place in the scene
    """
    # Deep copy so we don't mutate the original
    scene = copy.deepcopy(scene_base)
    scene["objects"] = objects_to_include

    return get_top_down_frame(scene, objaverse_asset_dir, width, height)


def capture_progressive_top_down_frames(scene, objaverse_asset_dir, save_dir, query_name, width=1024, height=1024):
    """
    Captures three top-down images:
      1. Floor objects only
      2. Floor + wall objects
      3. All objects (floor + wall + small)

    Args:
        scene:               Full generated scene dict (must have floor_objects,
                             wall_objects, small_objects keys)
        objaverse_asset_dir: Path to objaverse assets
        save_dir:            Directory to save PNGs
        query_name:          Base name for output files
    """
    floor_objects = scene.get("floor_objects", [])
    wall_objects  = scene.get("wall_objects", [])
    small_objects = scene.get("small_objects", [])

    stages = [
        ("stage1_floor",            floor_objects),
        ("stage2_floor_and_wall",   floor_objects + wall_objects),
        ("stage3_all",              floor_objects + wall_objects + small_objects),
    ]

    images = {}
    for stage_name, objects in stages:
        print(f"\n[Top-down capture] {stage_name} — {len(objects)} objects")
        image = get_top_down_frame_for_objects(
            scene, objects, objaverse_asset_dir, width, height
        )
        out_path = os.path.join(save_dir, f"{query_name}_{stage_name}.png")
        image.save(out_path)
        print(f"  Saved → {out_path}")
        images[stage_name] = image

    return images


def get_room_top_down_frame(scene, room_id, objaverse_asset_dir, width=1024, height=1024, padding=1.05):
    import copy
    import numpy as np

    scene_copy = copy.deepcopy(scene)
    scene_copy["objects"] = [
        obj for obj in scene.get("objects", [])
        if obj.get("roomId") == room_id
    ]

    room = next((r for r in scene["rooms"] if r["id"] == room_id), None)
    if room is None:
        raise ValueError(f"Room '{room_id}' not found in scene")

    floor_polygon = room["floorPolygon"]
    xs = [p["x"] for p in floor_polygon]
    zs = [p["z"] for p in floor_polygon]

    room_center_x = (min(xs) + max(xs)) / 2
    room_center_z = (min(zs) + max(zs)) / 2
    room_width    = (max(xs) - min(xs)) * padding
    room_depth    = (max(zs) - min(zs)) * padding

    try:
        wall_height = max(point["y"] for point in scene["walls"][0]["polygon"])
    except Exception:
        wall_height = 2.7

    # Use a narrow FOV so the camera sits closer to the scene
    fov_deg = 30.0
    fov_rad = np.radians(fov_deg)
    aspect  = width / height

    h_for_depth = (room_depth / 2) / np.tan(fov_rad / 2)
    h_for_width = (room_width / 2) / (np.tan(fov_rad / 2) * aspect)
    cam_height  = max(h_for_depth, h_for_width) + wall_height

    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=1.5,
        scene=scene_copy,
        width=width,
        height=height,
        fieldOfView=fov_deg,   # must match pose FOV
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
    )

    pose = {
        "orthographic": False,
        "fieldOfView": fov_deg,   # must match controller FOV
        "position": {
            "x": room_center_x,
            "y": cam_height,
            "z": room_center_z,
        },
        "rotation": {"x": 90, "y": 0, "z": 0},
        "farClippingPlane":  cam_height + 10,
        "nearClippingPlane": max(0.01, cam_height - wall_height - 1),
    }

    for _ in range(10):
        pose["farClippingPlane"]  = pose["position"]["y"] + 10
        pose["nearClippingPlane"] = max(0.01, pose["position"]["y"] - wall_height - 1)

        event = controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        frame = event.third_party_camera_frames[-1]

        if all_edges_white(frame):
            break

        pose["position"]["y"] += 0.25   # smaller nudge steps too

    controller.stop()
    return Image.fromarray(frame)


def get_all_room_top_down_frames(scene, objaverse_asset_dir, save_dir=None, query_name="scene", width=1024, height=1024):
    """
    Renders a top-down image for every room in the scene.
    Returns a dict of {room_id: PIL.Image}.
    Optionally saves PNGs to save_dir.
    """
    import os
    images = {}

    for room in scene["rooms"]:
        room_id = room["id"]
        print(f"[Room top-down] Rendering '{room_id}'...")

        try:
            img = get_room_top_down_frame(
                scene, room_id, objaverse_asset_dir, width, height
            )
            images[room_id] = img

            if save_dir:
                safe_name = room_id.replace(" ", "_")
                out_path = os.path.join(save_dir, f"{query_name}_{safe_name}_topdown.png")
                img.save(out_path)
                print(f"  Saved → {out_path}")

        except Exception as e:
            print(f"  [warn] Failed for room '{room_id}': {e}")

    return images


def get_top_down_frame_ithor(scene, objaverse_asset_dir, width=1024, height=1024):
    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=1.5,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=90,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
    )

    controller.reset(scene)

    event = controller.step(action="GetMapViewCameraProperties")
    pose = copy.deepcopy(event.metadata["actionReturn"])

    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )

    controller.stop()

    top_down_frame = event.third_party_camera_frames[0]

    return Image.fromarray(top_down_frame)


def main(save_path):
    scene = compress_json.load(save_path + f"scene.json", "r")
    image = get_top_down_frame(scene)
    image.save(f"test1.png")

    compress_json.dump(scene, save_path + f"scene.json", json_kwargs=dict(indent=4))


def visualize_asset(asset_id, version):
    empty_house = compress_json.load("empty_house.json")
    empty_house["objects"] = [
        {
            "assetId": asset_id,
            "id": "test_asset",
            "kinematic": True,
            "position": {"x": 0, "y": 0, "z": 0},
            "rotation": {"x": 0, "y": 0, "z": 0},
            "material": None,
        }
    ]
    image = get_top_down_frame(empty_house, version)
    image.show()


def get_room_images(scene, objaverse_asset_dir, width=1024, height=1024):
    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=1.5,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=135,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
    )

    wall_height = max([point["y"] for point in scene["walls"][0]["polygon"]])

    room_images = {}
    for room in scene["rooms"]:
        room_name = room["roomType"]
        camera_height = wall_height - 0.2

        room_vertices = [[point["x"], point["z"]] for point in room["floorPolygon"]]

        room_center = np.mean(room_vertices, axis=0)
        floor_center = np.array([room_center[0], 0, room_center[1]])
        camera_center = np.array([room_center[0], camera_height, room_center[1]])
        corners = np.array(
            [[point[0], camera_height, point[1]] for point in room_vertices]
        )
        farest_corner = np.argmax(np.linalg.norm(corners - camera_center, axis=1))

        vector_1 = floor_center - camera_center
        vector_2 = farest_corner - camera_center
        x_angle = (
            90
            - np.arccos(
                np.dot(vector_1, vector_2)
                / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
            )
            * 180
            / np.pi
        )

        if not controller.last_event.third_party_camera_frames:
            controller.step(
                action="AddThirdPartyCamera",
                position=dict(
                    x=camera_center[0], y=camera_center[1], z=camera_center[2]
                ),
                rotation=dict(x=0, y=0, z=0),
            )

        images = []
        for angle in tqdm(range(0, 360, 90)):
            controller.step(
                action="UpdateThirdPartyCamera",
                rotation=dict(x=x_angle, y=angle + 45, z=0),
                position=dict(
                    x=camera_center[0], y=camera_center[1], z=camera_center[2]
                ),
            )
            images.append(
                Image.fromarray(controller.last_event.third_party_camera_frames[0])
            )

        room_images[room_name] = images

    controller.stop()
    return room_images


def ithor_video(scene, objaverse_asset_dir, width, height, scene_type):
    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=2,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=90,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
    )

    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    wall_height = 2.5
    camera_height = wall_height - 0.2

    if not controller.last_event.third_party_camera_frames:
        controller.step(
            action="AddThirdPartyCamera",
            position=dict(
                x=pose["position"]["x"], y=camera_height, z=pose["position"]["z"]
            ),
            rotation=dict(x=0, y=0, z=0),
        )

    images = []

    for angle in tqdm(range(0, 360, 1)):
        controller.step(
            action="UpdateThirdPartyCamera",
            rotation=dict(x=45, y=angle, z=0),
            position=dict(
                x=pose["position"]["x"], y=camera_height, z=pose["position"]["z"]
            ),
        )
        images.append(controller.last_event.third_party_camera_frames[0])

    imsn = ImageSequenceClip(images, fps=30)

    # Create text clips
    txt_clip_query = (
        TextClip(f"Query: {scene_type}", fontsize=30, color="white", font="Arial-Bold")
        .set_pos(("center", "top"))
        .set_duration(imsn.duration)
    )
    txt_clip_room = (
        TextClip(
            f"Room Type: {scene_type}", fontsize=30, color="white", font="Arial-Bold"
        )
        .set_pos(("center", "bottom"))
        .set_duration(imsn.duration)
    )

    # Overlay the text clip on the first video clip
    video = CompositeVideoClip([imsn, txt_clip_query, txt_clip_room])

    controller.stop()

    return video


def room_video(scene, objaverse_asset_dir, width, height):
    def add_line_breaks(text, max_line_length):
        words = text.split(" ")
        lines = []
        current_line = []

        for word in words:
            if len(" ".join(current_line + [word])) <= max_line_length:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]

        lines.append(" ".join(current_line))

        return "\n".join(lines)

    """Saves a top-down video of the house."""
    controller = Controller(
        commit_id=THOR_COMMIT_ID,
        agentMode="default",
        makeAgentsVisible=False,
        visibilityDistance=2,
        scene=scene,
        width=width,
        height=height,
        fieldOfView=90,
        action_hook_runner=ProceduralAssetHookRunner(
            asset_directory=objaverse_asset_dir,
            asset_symlink=True,
            verbose=True,
        ),
    )

    try:
        query = scene["query"]
    except:
        query = scene["rooms"][0]["roomType"]

    wall_height = max([point["y"] for point in scene["walls"][0]["polygon"]])

    text_query = add_line_breaks(query, 60)
    videos = []
    for room in scene["rooms"]:
        room_name = room["roomType"]
        camera_height = wall_height - 0.2
        print("camera height: ", camera_height)

        room_vertices = [[point["x"], point["z"]] for point in room["floorPolygon"]]

        room_center = np.mean(room_vertices, axis=0)
        floor_center = np.array([room_center[0], 0, room_center[1]])
        camera_center = np.array([room_center[0], camera_height, room_center[1]])
        corners = np.array(
            [[point["x"], point["y"], point["z"]] for point in room["floorPolygon"]]
        )
        farest_corner = corners[
            np.argmax(np.linalg.norm(corners - camera_center, axis=1))
        ]

        vector_1 = floor_center - camera_center
        vector_2 = farest_corner - camera_center
        x_angle = (
            90
            - np.arccos(
                np.dot(vector_1, vector_2)
                / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
            )
            * 180
            / np.pi
        )

        images = []
        if not controller.last_event.third_party_camera_frames:
            controller.step(
                action="AddThirdPartyCamera",
                position=dict(
                    x=camera_center[0], y=camera_center[1], z=camera_center[2]
                ),
                rotation=dict(x=0, y=0, z=0),
            )

        for angle in tqdm(range(0, 360, 1)):
            controller.step(
                action="UpdateThirdPartyCamera",
                rotation=dict(x=x_angle, y=angle, z=0),
                position=dict(
                    x=camera_center[0], y=camera_center[1], z=camera_center[2]
                ),
            )
            images.append(controller.last_event.third_party_camera_frames[0])

        imsn = ImageSequenceClip(images, fps=30)

        # Create text clips
        txt_clip_query = (
            TextClip(
                f"Query: {text_query}", fontsize=30, color="white", font="Arial-Bold"
            )
            .set_pos(("center", "top"))
            .set_duration(imsn.duration)
        )
        txt_clip_room = (
            TextClip(
                f"Room Type: {room_name}", fontsize=30, color="white", font="Arial-Bold"
            )
            .set_pos(("center", "bottom"))
            .set_duration(imsn.duration)
        )

        # Overlay the text clip on the first video clip
        video = CompositeVideoClip([imsn, txt_clip_query, txt_clip_room])

        # Add this room's video to the list
        videos.append(video)

    # Concatenate all room videos into one final video
    final_video = concatenate_videoclips(videos)
    controller.stop()

    return final_video


def get_asset_metadata(obj_data: Dict[str, Any]):
    if "assetMetadata" in obj_data:
        return obj_data["assetMetadata"]
    elif "thor_metadata" in obj_data:
        return obj_data["thor_metadata"]["assetMetadata"]
    else:
        raise ValueError("Can not find assetMetadata in obj_data")


def get_annotations(obj_data: Dict[str, Any]):
    if "annotations" in obj_data:
        return obj_data["annotations"]
    else:
        # The assert here is just double-checking that a field that should exist does.
        assert "onFloor" in obj_data, f"Can not find annotations in obj_data {obj_data}"

        return obj_data


def get_bbox_dims(obj_data: Dict[str, Any]):
    am = get_asset_metadata(obj_data)

    bbox_info = am["boundingBox"]

    if "x" in bbox_info:
        return bbox_info

    if "size" in bbox_info:
        return bbox_info["size"]

    mins = bbox_info["min"]
    maxs = bbox_info["max"]

    return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}


def get_secondary_properties(obj_data: Dict[str, Any]):
    am = get_asset_metadata(obj_data)
    return am["secondaryProperties"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        help="Mode to run (top_down_frame, top_down_video, room_image).",
        default="top_down_frame",
    )
    parser.add_argument(
        "--objaverse_asset_dir",
        help="Directory to load assets from.",
        default="./objaverse/processed_2023_09_23_combine_scale",
    )
    parser.add_argument(
        "--scene",
        help="Scene to load.",
        default=os.path.join(
            HOLODECK_BASE_DATA_DIR, "scenes/a_living_room/a_living_room.json"
        ),
    )

    args = parser.parse_args()
    scene = compress_json.load(args.scene)

    if "query" not in scene:
        scene["query"] = args.scene.split("/")[-1].split(".")[0]

    if args.mode == "top_down_frame":
        image = get_top_down_frame(scene, args.objaverse_asset_dir)
        image.show()

    elif args.mode == "room_video":
        video = room_video(scene, args.objaverse_asset_dir, 1024, 1024)
        video.write_videofile(args.scene.replace(".json", ".mp4"), fps=30)

    elif args.mode == "room_image":
        room_images = get_room_images(scene, args.objaverse_asset_dir, 1024, 1024)
        save_folder = "/".join(args.scene.split("/")[:-1])
        for room_name, images in room_images.items():
            for i, image in enumerate(images):
                image.save(f"{save_folder}/{room_name}_{i}.png")


def critique_scene_image(image_path: str, query: str, constraints: dict, llm: LLMWrapper) -> dict:
    constraint_summary = "\n".join(
        f"  {obj} | " + " | ".join(
            c["constraint"] + (f", {c['target']}" if "target" in c else "")
            for c in cons
        )
        for obj, cons in constraints.items()
    )

    prompt = f"""You are an experienced room designer evaluating a procedurally generated room layout.
Original prompt: "{query}"

The layout was generated using these placement constraints (object | global | constraint, target | ...):
{constraint_summary}

The available constraints are:
- Global (pick one per object): edge, middle
- Distance: near <object>, far <object>
- Position: in front of <object>, around <object>, side of <object>, left of <object>, right of <object>
- Alignment: center aligned <object>
- Rotation: face to <object>

Rules the original designer followed:
1. Objects are placed iteratively — later objects depend on earlier ones only
2. Larger objects are placed first
3. Objects of the same type are usually aligned
4. Most objects should be at the edge to keep the room spacious
5. Chairs should use "around" and must be near and face to their table/desk

Analyze the top-down floor plan and return ONLY a JSON object with this exact structure:
{{
  "scores": {{
    "theme_match": <1-5>,
    "object_placement": <1-5>,
    "traffic_flow": <1-5>
  }},
  "overall_score": <1-5>,
  "issues": [
    {{
      "object": "<object name exactly as in constraints above>",
      "problem": "<what looks wrong visually>",
      "new_constraint_line": "<object> | <global> | <constraint, target> | ..."
    }}
  ],
  "reasoning": "<brief overall explanation>"
}}

For new_constraint_line, follow the exact same format as the constraint summary above.
Only include objects that genuinely need fixing. Return raw JSON only, no markdown fences."""

    raw = llm.vision(prompt, image_path)

    # ── Robust JSON extraction ─────────────────────────────────────────────
    raw = raw.strip()

    # Strip markdown fences if present (```json ... ``` or ``` ... ```)
    if "```" in raw:
        # Extract content between first and last fence
        parts = raw.split("```")
        # parts[1] is the content inside the fences
        if len(parts) >= 3:
            raw = parts[1]
            # Remove language tag if present (e.g. "json\n")
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

    # Find the outermost { } in case there's any preamble text
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response:\n{raw}")
    raw = raw[start:end]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from response:\n{raw}\nError: {e}")

    # ── Validate structure ─────────────────────────────────────────────────
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object (dict), got {type(result).__name__}: {result}")

    # Ensure required keys exist with safe defaults
    result.setdefault("scores", {"theme_match": 0, "object_placement": 0, "traffic_flow": 0})
    result.setdefault("overall_score", 0)
    result.setdefault("issues", [])
    result.setdefault("reasoning", "")

    # Ensure overall_score is a number
    if not isinstance(result["overall_score"], (int, float)):
        try:
            result["overall_score"] = float(result["overall_score"])
        except (ValueError, TypeError):
            result["overall_score"] = 0

    print(f"\n[Vision critique raw response]\n{raw}\n")

    return result


def apply_critique_to_constraints(
    constraints: dict,
    critique: dict,
    floor_generator,
) -> tuple:
    import copy
    updated = copy.deepcopy(constraints)
    change_log = []

    for issue in critique.get("issues", []):
        obj = issue["object"]
        new_line = issue.get("new_constraint_line", "").strip()

        if not new_line or obj not in updated:
            if obj not in updated:
                print(f"  [warn] '{obj}' not in constraint graph, skipping")
            continue

        valid_names = list(updated.keys())
        try:
            parsed = floor_generator.parse_constraints(new_line, valid_names)
        except Exception as e:
            print(f"  [warn] Could not parse constraint line for {obj}: {e}")
            continue

        if obj not in parsed:
            print(f"  [warn] Parser didn't extract '{obj}' from line: {new_line}")
            continue

        # Validate all targets exist in the constraint graph
        validated = []
        for c in parsed[obj]:
            if "target" not in c:
                validated.append(c)
            elif c["target"] in updated:
                validated.append(c)
            else:
                print(
                    f"  [warn] Dropping constraint for '{obj}': "
                    f"target '{c['target']}' not in constraint graph"
                )

        has_global = any(v["type"] == "global" for v in validated)
        if not has_global:
            print(
                f"  [warn] Skipping override for '{obj}': "
                f"no global constraint after validation"
            )
            continue

        old = updated[obj]
        updated[obj] = validated
        change_log.append(
            f"{obj}:\n"
            f"    before: {old}\n"
            f"    after:  {validated}"
        )

    return updated, change_log
