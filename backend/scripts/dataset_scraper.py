import os
import shutil
import kagglehub
import fiftyone.zoo as foz

# Resolve BASE_DIR relative to the script's parent directory (backend/)
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
os.makedirs(BASE_DIR, exist_ok=True)

# download COCO samples
try:
    coco_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=10
    )
    print(f"copying photos to: {BASE_DIR}")
    copied_count = 0
    for sample in coco_dataset:
        source_path = sample.filepath
        filename = os.path.basename(source_path)
        destination_path = os.path.join(BASE_DIR, filename)
        shutil.copy2(source_path, destination_path)
        copied_count += 1

    print(f"{copied_count} photos are now in {BASE_DIR}")

except Exception as e:
    print(f"Error: {e}")

# # xray samples
# try:
#     medical_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
#     src_medical = os.path.join(medical_path, "chest_xray", "train", "NORMAL")
#     dst_medical = os.path.join(BASE_DIR, "medical")
#     shutil.copytree(src_medical, dst_medical, dirs_exist_ok=True)
#     print("medical dataset copied")
# except Exception as e:
#     print(f"medical dataset error: {e}")

# # night samples
# try:
#     night_path = kagglehub.dataset_download("goutham7/exdark-dataset")
#     dst_night = os.path.join(BASE_DIR, "night")
#     shutil.copytree(night_path, dst_night, dirs_exist_ok=True)
#     print("night images copied")
# except Exception as e:
#     print(f"night images error: {e}")

print("All downloads finished")
