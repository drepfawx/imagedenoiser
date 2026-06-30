import os
import shutil
import random
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(_BACKEND_DIR, "dataset")
COCO_JSON_PATH = os.path.join(BASE_DIR, "instances_val2017.json")

# clear the dataset folder of previous images, but preserve the COCO JSON
_KEEP_FILES = {"instances_val2017.json"}
if os.path.exists(BASE_DIR):
    for f in os.listdir(BASE_DIR):
        if f in _KEEP_FILES:
            continue
        f_path = os.path.join(BASE_DIR, f)
        try:
            if os.path.isfile(f_path):
                os.remove(f_path)
            elif os.path.isdir(f_path):
                shutil.rmtree(f_path)
        except Exception as ex:
            print(f"Error clearing {f_path}: {ex}")
os.makedirs(BASE_DIR, exist_ok=True)

try:
    if os.path.exists(COCO_JSON_PATH):
        print(f"Reading COCO metadata from: {COCO_JSON_PATH}")
        with open(COCO_JSON_PATH, 'r') as f:
            metadata = json.load(f)
        
        images = metadata.get("images", [])
        if len(images) > 0:
            # select 10 random images from the 5000 validation images
            selected_images = random.sample(images, min(100, len(images)))

            def _download(img):
                filename = img["file_name"]
                url = f"http://images.cocodataset.org/val2017/{filename}"
                dest = os.path.join(BASE_DIR, filename)
                headers = {'User-Agent': 'Mozilla/5.0'}
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    with open(dest, 'wb') as out:
                        out.write(resp.read())
                return filename

            print(f"Downloading {len(selected_images)} random images to: {BASE_DIR}")
            copied_count = 0
            with ThreadPoolExecutor(max_workers=min(len(selected_images), 20)) as pool:
                futures = {pool.submit(_download, img): img for img in selected_images}
                for future in as_completed(futures):
                    try:
                        name = future.result()
                        print(f"{name}")
                        copied_count += 1
                    except Exception as dl_err:
                        img = futures[future]
                        print(f"Failed to download {img['file_name']}: {dl_err}")

            print(f"{copied_count} photos are now in {BASE_DIR}")
        else:
            print("No image records found in COCO JSON.")
    else:
        print(f"Error: COCO annotations JSON not found at {COCO_JSON_PATH}")

except Exception as e:
    print(f"Error: {e}")

print("All downloads finished")
