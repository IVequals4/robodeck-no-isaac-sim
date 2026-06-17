import sys
import json
import os
sys.path.append('/home/arc-lp/projects/24560221/Holodeck')
sys.path.append('/home/arc-lp/projects/24560221/Holodeck/ai2holodeck/generation')

from utils import get_top_down_frame, get_all_room_top_down_frames, get_room_images

scene_folder = '/home/arc-lp/projects/24560221/output/scenes ready/a_kitchen-2026-05-29-22-47-33-254599'
objaverse_asset_dir = '/home/arc-lp/projects/24560221/datasets/2023_09_23/assets'
query_name = os.path.basename(scene_folder).split('-')[0]

# with open(os.path.join(scene_folder, f'a_dining_room_dining_room_attempt1 .json'), 'r') as f:
with open('/home/arc-lp/projects/24560221/output/scenes ready/a_kitchen-2026-05-29-22-47-33-254599/a_kitchen.json', 'r') as f:
    scene = json.load(f)

print(f"Rendering {len(scene['objects'])} objects...")

# Full scene top-down
top_image = get_top_down_frame(scene, objaverse_asset_dir, 1024, 1024)
top_image.save(os.path.join(scene_folder, f'{query_name} ai2thor improved.png'))
print("Saved full top-down image")


# # Corner images per room
# room_images = get_room_images(scene, objaverse_asset_dir=objaverse_asset_dir)
# for room_name, images in room_images.items():
#     for i, img in enumerate(images):
#         img.save(os.path.join(scene_folder, f"{room_name}_corner_{i}.png"))
# print(f"Saved corner images for {len(room_images)} room(s)")

print("Done")