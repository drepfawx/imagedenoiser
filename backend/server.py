import cv2
import numpy as np
import io
import base64
import os
import time
import glob as _glob
import random as _random
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import modularized components from core package
from core.model import model
from core.metrics import compute_edge_preservation, compute_psnr, compute_ssim, compute_mse
from core.noise import add_gaussian_noise, add_salt_pepper_noise, detect_noise_type
from core.filters import FILTER_REGISTRY, run_filter_with_metrics

# fastapi app and cors config
app = FastAPI(title="Image Denoiser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# helper functions
def decode_uploaded_image(file):
    "read and decode an uploaded image file into an opencv BGR image."
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def mat_to_base64(img):
    "encode opencv image (numpy array) to base64 png string."
    _, buffer = cv2.imencode('.png', img)
    io_buf = io.BytesIO(buffer)
    return base64.b64encode(io_buf.getvalue()).decode('utf-8')

def compute_luminance_histogram(img, bins=48):
    "compute normalized 48-bin luminance histogram."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray, bins=bins, range=(0, 256))
    max_val = float(np.max(hist))
    if max_val == 0:
        max_val = 1.0
    return [float(x / max_val) for x in hist]

def build_residual_mask(noisy, cleaned):
    "generate visual difference mask highlighting changes."
    diff = cv2.absdiff(noisy, cleaned)
    intensity = np.max(diff, axis=2).astype(np.float32)
    
    h, w = noisy.shape[:2]
    # bgra layout (opencv encodes bgr/bgra natively)
    mask = np.zeros((h, w, 4), dtype=np.uint8)
    mask[..., 0] = 0 # blue
    mask[..., 1] = np.clip(intensity * 3.0, 0, 255).astype(np.uint8) # green
    mask[..., 2] = 255 # red
    mask[..., 3] = np.clip(intensity * 8.0, 0, 255).astype(np.uint8) # alpha
    
    _, buffer = cv2.imencode('.png', mask)
    return base64.b64encode(buffer).decode('utf-8')

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

import threading
import subprocess
import sys

_scrape_lock = threading.Lock()
_last_scrape_time = 0.0

def run_scraper_if_needed():
    global _last_scrape_time
    with _scrape_lock:
        now = time.time()
        # If it was run in the last 5 seconds, skip running it again to avoid race conditions
        if now - _last_scrape_time < 5.0:
            return
        
        # Run the scraper script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "dataset_scraper.py")
        subprocess.run([sys.executable, script_path], check=True)
        _last_scrape_time = time.time()

def _load_dataset_images(n, max_dim):
    try:
        run_scraper_if_needed()
    except Exception as ex:
        print(f"Scraper execution failed: {ex}")

    files = []
    for pat in ["*.png", "*.jpg", "*.JPG", "*.jpeg", "*.PNG", "*.JPEG"]:
        files.extend(_glob.glob(os.path.join(DATASET_DIR, pat)))
    if not files:
        return []
    files.sort()
    
    imgs = []
    for path in files[:n]:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
        imgs.append(img)
    return imgs

# api endpoints
@app.post("/api/process")
def process_image(file: UploadFile = File(...)):
    try:
        input_img = decode_uploaded_image(file)
        
        if input_img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format."})

        # noise analysis (estimates sigma and histogram spikes)
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        mad = np.median(np.abs(laplacian - np.median(laplacian))) / 0.6745
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        # calculating isolated white and black pixels to filter out large solid regions (e.g. skies, borders)
        white_mask = (gray == 255).astype(np.uint8)
        black_mask = (gray == 0).astype(np.uint8)
        
        kernel = np.ones((3, 3), np.uint8)
        eroded_white = cv2.erode(white_mask, kernel, iterations=1)
        eroded_black = cv2.erode(black_mask, kernel, iterations=1)
        
        isolated_white = float(np.sum(white_mask) - np.sum(eroded_white))
        isolated_black = float(np.sum(black_mask) - np.sum(eroded_black))
        
        # average of neighboring bins to stabilize the denominator against quantization/single-bin dropouts
        near_white_avg = float(np.mean(hist[250:255]))
        near_black_avg = float(np.mean(hist[1:6]))
        
        salt_spike = isolated_white / (near_white_avg + 1e-5)
        pepper_spike = isolated_black / (near_black_avg + 1e-5)
        
        final_sigma = float(mad)
        final_spike = float(max(salt_spike, pepper_spike))

        # if sigma's near zero, the image is clean and any histogram spike is from solid regions (black bars, white UI), not noise
        if final_sigma <= 1.0:
            final_spike = 0.0

        # determine noise type
        if final_spike > 3.5 and final_sigma > 1.0:
            system_decision = "salt_and_pepper"
        elif final_sigma > 12.0:
            system_decision = "gaussian"
        else:
            system_decision = "none"

        # run filters only if noise is detected
        detected_noise = system_decision.upper() != "NONE"
        filter_results = {}
        best_filter_id = None

        if detected_noise:
            def _run_filter(fid):
                finfo = FILTER_REGISTRY[fid]
                cleaned, metrics = run_filter_with_metrics(input_img, finfo["fn"])
                return fid, cleaned, metrics

            with ThreadPoolExecutor(max_workers=len(FILTER_REGISTRY)) as pool:
                futures = {pool.submit(_run_filter, fid): fid for fid in FILTER_REGISTRY}
                for future in as_completed(futures):
                    fid, cleaned, metrics = future.result()
                    filter_results[fid] = (cleaned, metrics)

            if system_decision == "salt_and_pepper":
                best_filter_id = "median"
            else:
                best_filter_id = max(
                    filter_results.keys(),
                    key=lambda fid: filter_results[fid][1].get("utility_score") or 0
                )

        algorithm_used = FILTER_REGISTRY[best_filter_id]["name"] if best_filter_id else "no significant noise detected"
        cleaned_img = filter_results[best_filter_id][0] if best_filter_id else input_img.copy()

        # build response structures
        filter_metrics = []
        all_cleaned = {}
        all_histograms = {}
        all_masks = {}

        for fid, finfo in FILTER_REGISTRY.items():
            if fid in filter_results:
                cleaned_f, metrics_f = filter_results[fid]
                filter_metrics.append({"id": fid, "name": finfo["name"], **metrics_f})
                all_cleaned[fid] = mat_to_base64(cleaned_f)
                all_histograms[fid] = compute_luminance_histogram(cleaned_f)
                all_masks[fid] = build_residual_mask(input_img, cleaned_f)
            else:
                filter_metrics.append({
                    "id": fid, "name": finfo["name"],
                    "runtime_ms": None, "edge_preservation": None,
                    "laplacian_var": None, "brisque": None,
                    "niqe": None, "utility_score": None
                })

        noisy_hist = compute_luminance_histogram(input_img)
        cleaned_hist = compute_luminance_histogram(cleaned_img)
        residual_mask = build_residual_mask(input_img, cleaned_img) if detected_noise else None

        return {
            "status": "success",
            "analysis": {
                "estimated_sigma": round(final_sigma, 2),
                "histogram_spike": round(final_spike, 2),
                "detected_noise": system_decision.upper(),
                "selected_algorithm": algorithm_used,
                "best_filter_id": best_filter_id
            },
            "images": {
                "noisy": mat_to_base64(input_img),
                "cleaned": mat_to_base64(cleaned_img),
                "all_cleaned": all_cleaned,
                "all_histograms": all_histograms,
                "all_masks": all_masks,
                "noisy_hist": noisy_hist,
                "cleaned_hist": cleaned_hist,
                "residual_mask": residual_mask
            },
            "filter_metrics": filter_metrics
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/apply-filter")
def apply_filter(
    file: UploadFile = File(...),
    filter: str = Query(..., description="Filter to apply: gaussian, median, bilateral, nlm, cnn")
):
    try:
        input_img = decode_uploaded_image(file)

        if input_img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format."})

        filter_name = filter.lower().strip()
        if filter_name not in FILTER_REGISTRY:
            return JSONResponse(status_code=400, content={"error": f"Unknown filter: {filter_name}"})

        finfo = FILTER_REGISTRY[filter_name]
        cleaned_img, metrics = run_filter_with_metrics(input_img, finfo["fn"])

        cleaned_hist = compute_luminance_histogram(cleaned_img)
        residual_mask = build_residual_mask(input_img, cleaned_img)

        return {
            "status": "success",
            "filter_applied": filter_name,
            "algorithm": finfo["name"],
            "images": {
                "cleaned": mat_to_base64(cleaned_img),
                "cleaned_hist": cleaned_hist,
                "residual_mask": residual_mask
            },
            "metrics": metrics
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/benchmark")
def run_benchmark(n: int = Query(default=10, description="Number of dataset images to evaluate")):
    try:
        images = _load_dataset_images(n, max_dim=256)
        if not images:
            return JSONResponse(status_code=404, content={"error": "No dataset images found in dataset/"})

        filter_ids = list(FILTER_REGISTRY.keys())

        gaussian_configs = [
            {"label": "σ=15", "sigma": 15},
            {"label": "σ=25", "sigma": 25},
            {"label": "σ=35", "sigma": 35},
            {"label": "σ=50", "sigma": 50},
        ]
        sp_configs = [
            {"label": "2%",  "ratio": 0.02},
            {"label": "5%",  "ratio": 0.05},
            {"label": "10%", "ratio": 0.10},
            {"label": "20%", "ratio": 0.20},
        ]

        def eval_level(noisy_images):
            acc = {fid: {"psnr": [], "ssim": [], "mse": [], "time": []} for fid in filter_ids}

            def _eval_one(orig, noisy, fid):
                t0 = time.perf_counter()
                cleaned = FILTER_REGISTRY[fid]["fn"](noisy)
                rt = (time.perf_counter() - t0) * 1000
                return fid, compute_psnr(orig, cleaned), compute_ssim(orig, cleaned), compute_mse(orig, cleaned), rt

            tasks = [
                (orig, noisy, fid)
                for orig, noisy in zip(images, noisy_images)
                for fid in filter_ids
            ]
            with ThreadPoolExecutor(max_workers=min(len(tasks), len(filter_ids) * 2)) as pool:
                futures = {pool.submit(_eval_one, *t): t for t in tasks}
                for future in as_completed(futures):
                    try:
                        fid, psnr, ssim, mse, rt = future.result()
                        acc[fid]["psnr"].append(psnr)
                        acc[fid]["ssim"].append(ssim)
                        acc[fid]["mse"].append(mse)
                        acc[fid]["time"].append(rt)
                    except Exception as ex:
                        _, _, fid = futures[future]
                        print(f"benchmark error [{fid}]: {ex}")

            return {
                fid: {
                    "psnr":    round(float(np.mean(v["psnr"])),    2) if v["psnr"]  else None,
                    "ssim":    round(float(np.mean(v["ssim"])),    4) if v["ssim"]  else None,
                    "mse":     round(float(np.mean(v["mse"])),     6) if v["mse"]   else None,
                    "time_ms": round(float(np.mean(v["time"])),    1) if v["time"]  else None,
                }
                for fid, v in acc.items()
            }

        gaussian_levels = []
        for cfg in gaussian_configs:
            noisy_imgs = [add_gaussian_noise(img, cfg["sigma"]) for img in images]
            gaussian_levels.append({"label": cfg["label"], "results": eval_level(noisy_imgs)})

        sp_levels = []
        for cfg in sp_configs:
            noisy_imgs = [add_salt_pepper_noise(img, cfg["ratio"]) for img in images]
            sp_levels.append({"label": cfg["label"], "results": eval_level(noisy_imgs)})

        # summary: average across all noise configs
        all_levels = gaussian_levels + sp_levels
        summary = {}
        for fid in filter_ids:
            agg = {"psnr": [], "ssim": [], "mse": [], "time_ms": []}
            for lvl in all_levels:
                r = lvl["results"].get(fid, {})
                for k in agg:
                    if r.get(k) is not None:
                        agg[k].append(r[k])
            summary[fid] = {
                "psnr":    round(float(np.mean(agg["psnr"])),    2) if agg["psnr"]    else None,
                "ssim":    round(float(np.mean(agg["ssim"])),    4) if agg["ssim"]    else None,
                "mse":     round(float(np.mean(agg["mse"])),     6) if agg["mse"]     else None,
                "time_ms": round(float(np.mean(agg["time_ms"])), 1) if agg["time_ms"] else None,
            }

        return {
            "status": "success",
            "n_images": len(images),
            "filter_names": {fid: FILTER_REGISTRY[fid]["name"] for fid in filter_ids},
            "gaussian":    gaussian_levels,
            "salt_pepper": sp_levels,
            "summary":     summary,
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/confusion-matrix")
def run_confusion_matrix(n: int = Query(default=10, description="Number of source images")):
    try:
        selected_images = _load_dataset_images(n, max_dim=512)
        if not selected_images:
            return JSONResponse(status_code=404, content={"error": "No dataset images found in dataset/"})

        classes = ["gaussian", "salt_and_pepper", "none"]
        matrix = {actual: {pred: 0 for pred in classes} for actual in classes}
        total  = {cls: 0 for cls in classes}

        def _eval_image(orig):
            test_cases = [
                ("none",            orig),
                ("gaussian",        add_gaussian_noise(orig, 25)),
                ("gaussian",        add_gaussian_noise(orig, 50)),
                ("salt_and_pepper", add_salt_pepper_noise(orig, 0.05)),
                ("salt_and_pepper", add_salt_pepper_noise(orig, 0.10)),
            ]
            return [(actual, detect_noise_type(noisy)) for actual, noisy in test_cases]

        with ThreadPoolExecutor(max_workers=len(selected_images)) as pool:
            futures = [pool.submit(_eval_image, orig) for orig in selected_images]
            for future in as_completed(futures):
                for actual, predicted in future.result():
                    matrix[actual][predicted] += 1
                    total[actual] += 1

        return {
            "status": "success",
            "n_images": len(selected_images),
            "matrix": matrix,
            "total_per_class": total,
            "classes": classes,
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})