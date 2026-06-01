import cv2
import numpy as np
import torch
import torch.nn as nn
import io
import base64
import os
import time
import datetime
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from brisque import BRISQUE
from niqe import niqe as compute_niqe

# fastapi app and cors config
app = FastAPI(title = "Image Denoiser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# simple cnn denoiser
class SimpleCNNDenoiser(nn.Module):
    def __init__(self):
        super(SimpleCNNDenoiser, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        noise = self.decoder(self.encoder(x))
        return x - noise

# initialize model and load weights
model = SimpleCNNDenoiser()
try:
    model.load_state_dict(torch.load('my_model.pth', map_location='cpu'))
    model.eval()
    # optimize layout and compile graph with jit
    model.to(memory_format=torch.channels_last)
    model = torch.jit.script(model)
    print("model loaded and jit compiled successfully.")
except Exception as e:
    print(f"warning: could not load model weights (my_model.pth): {e}")

# set pytorch thread count for vps cpu execution
torch.set_num_threads(4)

# patch brisque feature scaling to avoid float/double casting issue
def patched_scale_features(self, features):
    min_flat = np.array([float(m.item() if hasattr(m, 'item') else m) for m in self.scale_params['min_']], dtype=np.float64)
    max_flat = np.array([float(m.item() if hasattr(m, 'item') else m) for m in self.scale_params['max_']], dtype=np.float64)
    features_flat = np.array([float(f.item() if hasattr(f, 'item') else f) for f in features], dtype=np.float64)
    return -1 + (2.0 / (max_flat - min_flat) * (features_flat - min_flat))

BRISQUE.scale_features = patched_scale_features
brisque_scorer = BRISQUE()

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

def compute_edge_preservation(img1, img2):
    "normalized correlation of high-pass laplacian-filtered images"
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    lap1 = cv2.Laplacian(g1, cv2.CV_64F)
    lap2 = cv2.Laplacian(g2, cv2.CV_64F)
    
    d1 = lap1 - np.mean(lap1)
    d2 = lap2 - np.mean(lap2)
    
    num = np.sum(d1 * d2)
    den = np.sqrt(np.sum(d1 ** 2) * np.sum(d2 ** 2))
    if den == 0:
        return 0.0
    val = float(num / den)
    return round(float(np.clip(val, 0.0, 1.0)), 4)

# filter functions
def apply_tiled_filter(input_img, filter_fn, tile_size=512, overlap=16):
    "generic wrapper that runs heavy filters in tiles to prevent timeouts when processing big images"
    h, w = input_img.shape[:2]
    
    # if the image is small enough, run it as a single block
    if max(h, w) <= tile_size:
        return filter_fn(input_img)
        
    # initialize output image as uint8 (since filters return uint8)
    output_img = np.zeros_like(input_img)
    
    # pad borders to handle overlaps nicely (reflect border replicates details)
    padded_img = cv2.copyMakeBorder(
        input_img, overlap, overlap, overlap, overlap, 
        cv2.BORDER_REFLECT_101
    )
    
    step = tile_size - 2 * overlap
    
    for y in range(0, h, step):
        y_start = y
        y_end = min(y + step, h)
        py_start = y_start
        py_end = py_start + tile_size
        if py_end > h + 2 * overlap:
            py_end = h + 2 * overlap
            py_start = max(0, py_end - tile_size)
            
        for x in range(0, w, step):
            x_start = x
            x_end = min(x + step, w)
            px_start = x_start
            px_end = px_start + tile_size
            if px_end > w + 2 * overlap:
                px_end = w + 2 * overlap
                px_start = max(0, px_end - tile_size)
                
            # extract tile and run the filter
            tile = padded_img[py_start:py_end, px_start:px_end]
            out_tile = filter_fn(tile)
            
            # map cropped patch back to original coordinates
            crop_y_start = overlap + (y_start - py_start)
            crop_y_end = crop_y_start + (y_end - y_start)
            crop_x_start = overlap + (x_start - px_start)
            crop_x_end = crop_x_start + (x_end - x_start)
            
            output_img[y_start:y_end, x_start:x_end] = out_tile[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            
    return output_img

def run_cnn_filter(input_img):
    "run pytorch cnn denoiser"
    def process_tile(tile):
        img_input = tile.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(memory_format=torch.channels_last)
        with torch.inference_mode():
            output_tensor = model(img_tensor)
        cleaned = output_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
        return np.clip(cleaned, 0, 255).astype(np.uint8)
        
    return apply_tiled_filter(input_img, process_tile)

def run_filter_with_metrics(input_img, filter_fn):
    "apply a filter, measure runtime, edge preservation, sharpness (laplacian var), brisque, and niqe."
    start = time.perf_counter()
    cleaned = filter_fn(input_img)
    runtime_ms = (time.perf_counter() - start) * 1000

    # 1. edge preservation ratio
    edge_preservation = compute_edge_preservation(input_img, cleaned)

    # 2. variance of laplacian (sharpness)
    gray_input = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    gray_cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    
    input_lap_var = float(np.var(cv2.Laplacian(gray_input, cv2.CV_32F)))
    cleaned_lap_var = float(np.var(cv2.Laplacian(gray_cleaned, cv2.CV_32F)))
    
    if input_lap_var > 0:
        laplacian_var = round(100.0 * (cleaned_lap_var / input_lap_var), 2)
    else:
        laplacian_var = 100.0

    # downscale the cleaned image for fast brisque/niqe metric evaluation (does not affect output resolution)
    h_c, w_c = cleaned.shape[:2]
    max_metric_dim = 1024
    if max(h_c, w_c) > max_metric_dim:
        scale_m = max_metric_dim / max(h_c, w_c)
        metric_img = cv2.resize(cleaned, (int(w_c * scale_m), int(h_c * scale_m)), interpolation=cv2.INTER_AREA)
    else:
        metric_img = cleaned

    # 3. brisque
    try:
        brisque_score = float(brisque_scorer.score(metric_img))
        brisque_score = round(brisque_score, 2)
    except Exception as e:
        print(f"brisque calculation failed: {e}")
        brisque_score = 0.0

    # 4. niqe
    try:
        niqe_score = float(compute_niqe(metric_img))
        niqe_score = round(niqe_score, 2)
    except Exception as e:
        print(f"niqe calculation failed: {e}")
        niqe_score = 0.0

    # 5. utility score (balances edge preservation, quality/brisque, and execution speed)
    Q = max(0.0, min(1.0, (100.0 - brisque_score) / 100.0))
    S = 1.0 / (1.0 + runtime_ms / 250.0)
    E = edge_preservation
    utility_score = round(float(0.4 * E + 0.3 * Q + 0.3 * S), 4)

    return cleaned, {
        "runtime_ms": round(runtime_ms, 1),
        "edge_preservation": edge_preservation,
        "laplacian_var": round(laplacian_var, 1),
        "brisque": brisque_score,
        "niqe": niqe_score,
        "utility_score": utility_score,
    }

# all available filters
FILTER_REGISTRY = {
    "gaussian":  {"name": "Gaussian Filter",       "fn": lambda img: cv2.GaussianBlur(img, (5, 5), 0)},
    "median":    {"name": "Median Filter",          "fn": lambda img: cv2.medianBlur(img, 5)},
    "bilateral": {"name": "Bilateral Filter",       "fn": lambda img: cv2.bilateralFilter(img, 9, 75, 75)},
    "nlm":       {"name": "Non-Local Means (NLM)",  "fn": lambda img: apply_tiled_filter(img, lambda tile: cv2.fastNlMeansDenoisingColored(tile, None, 10, 10, 7, 21), overlap=32)},
    "cnn":       {"name": "PyTorch CNN Denoiser",    "fn": run_cnn_filter},
}

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
        
        
        salt_spike = hist[255] / (hist[254] + 1e-5)
        pepper_spike = hist[0] / (hist[1] + 1e-5)
        
        final_sigma = float(mad)
        final_spike = float(max(salt_spike, pepper_spike))

        # if sigma's near zero, the image is clean and any histogram spike is from solid regions (black bars, white UI), not noise
        if final_sigma <= 1.0:
            final_spike = 0.0

        # determine auto-selected filter
        if final_spike > 3.5 and final_sigma > 1.0:
            system_decision = "salt_and_pepper"
            algorithm_used = "Median filter (OpenCV MedianBlur)"
            best_filter_id = "median"
        elif final_sigma > 12.0:
            system_decision = "gaussian"
            algorithm_used = "custom-trained CNN model"
            best_filter_id = "cnn"
        else:
            system_decision = "none"
            algorithm_used = "no significant noise detected"
            best_filter_id = None

        # run ONLY the recommended filter initially
        filter_metrics = []
        filter_outputs = {}

        if best_filter_id and best_filter_id in FILTER_REGISTRY:
            finfo = FILTER_REGISTRY[best_filter_id]
            cleaned_img, metrics = run_filter_with_metrics(input_img, finfo["fn"])
            filter_outputs[best_filter_id] = cleaned_img
        else:
            cleaned_img = input_img.copy()
            metrics = None

        # build initial filter metrics list (placeholders for other filters)
        for fid, finfo in FILTER_REGISTRY.items():
            if fid == best_filter_id and metrics is not None:
                filter_metrics.append({
                    "id": fid,
                    "name": finfo["name"],
                    **metrics,
                })
            else:
                filter_metrics.append({
                    "id": fid,
                    "name": finfo["name"],
                    "runtime_ms": None,
                    "edge_preservation": None,
                    "laplacian_var": None,
                    "brisque": None,
                    "niqe": None,
                    "utility_score": None
                })

        # compute noisy and cleaned histograms
        noisy_hist = compute_luminance_histogram(input_img)
        cleaned_hist = compute_luminance_histogram(cleaned_img)
        
        # compute residual mask if noise was detected
        detected_noise = system_decision.upper() != "NONE"
        residual_mask = build_residual_mask(input_img, cleaned_img) if detected_noise else None

        # build client cache dictionaries for the initially computed recommended filter
        all_cleaned = {}
        all_histograms = {}
        all_masks = {}
        if best_filter_id:
            best_base64 = mat_to_base64(cleaned_img)
            all_cleaned[best_filter_id] = best_base64
            all_histograms[best_filter_id] = cleaned_hist
            if detected_noise and residual_mask:
                all_masks[best_filter_id] = residual_mask

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