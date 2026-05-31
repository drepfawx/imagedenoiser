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

# initialize model and load trained weights
model = SimpleCNNDenoiser()
try:
    model.load_state_dict(torch.load('my_model.pth', map_location='cpu'))
    print("Model loaded from my_model.pth")
except Exception as e:
    print(f"Warning: could not load model weights (my_model.pth): {e}")

# fastapi app and CORS config
app = FastAPI(title="AI Denoising Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def mat_to_base64(img):
    "encode OpenCV image (numpy array) to base64 PNG string."
    _, buffer = cv2.imencode('.png', img)
    io_buf = io.BytesIO(buffer)
    return base64.b64encode(io_buf.getvalue()).decode('utf-8')

def run_cnn_filter(input_img):
    "Run the PyTorch CNN denoiser on an image."
    img_input = input_img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output_tensor = model(img_tensor)
    cleaned = output_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
    return np.clip(cleaned, 0, 255).astype(np.uint8)

def compute_edge_preservation(img1, img2):
    "Normalized correlation of high-pass Laplacian-filtered images."
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

def patched_scale_features(self, features):
    min_flat = np.array([float(m.item() if hasattr(m, 'item') else m) for m in self.scale_params['min_']], dtype=np.float64)
    max_flat = np.array([float(m.item() if hasattr(m, 'item') else m) for m in self.scale_params['max_']], dtype=np.float64)
    features_flat = np.array([float(f.item() if hasattr(f, 'item') else f) for f in features], dtype=np.float64)
    return -1 + (2.0 / (max_flat - min_flat) * (features_flat - min_flat))

BRISQUE.scale_features = patched_scale_features
brisque_scorer = BRISQUE()

def run_filter_with_metrics(input_img, filter_fn):
    """Apply a filter, measure runtime, edge preservation, sharpness (laplacian var), BRISQUE, and NIQE."""
    start = time.perf_counter()
    cleaned = filter_fn(input_img)
    runtime_ms = (time.perf_counter() - start) * 1000

    # 1. Edge Preservation Ratio
    edge_preservation = compute_edge_preservation(input_img, cleaned)

    # 2. Variance of Laplacian (Sharpness)
    gray_cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(np.var(cv2.Laplacian(gray_cleaned, cv2.CV_64F)))

    # 3. BRISQUE
    try:
        brisque_score = float(brisque_scorer.score(cleaned))
        brisque_score = round(brisque_score, 2)
    except Exception as e:
        print(f"BRISQUE calculation failed: {e}")
        brisque_score = 0.0

    # 4. NIQE
    try:
        niqe_score = float(compute_niqe(cleaned))
        niqe_score = round(niqe_score, 2)
    except Exception as e:
        print(f"NIQE calculation failed: {e}")
        niqe_score = 0.0

    # 5. Utility Score (balances Edge Preservation, Quality/BRISQUE, and Execution Speed)
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
    "nlm":       {"name": "Non-Local Means (NLM)",  "fn": lambda img: cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)},
    "cnn":       {"name": "PyTorch CNN Denoiser",    "fn": run_cnn_filter},
}

@app.post("/api/process")
async def process_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        input_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
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

        # reference metric for sharpness comparison
        input_lap_var = float(np.var(laplacian))

        # determine auto-selected filter
        if final_spike > 3.5:
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

        # Build initial filter metrics list (placeholders for other filters)
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

        # saving cleaned image to disk
        output_dir = "saved_results"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleaned_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, cleaned_img)
        print(f"Saved cleaned image: {filepath}")

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
                "all_cleaned": {
                    best_filter_id: mat_to_base64(cleaned_img)
                } if best_filter_id else {}
            },
            "filter_metrics": filter_metrics
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/apply-filter")
async def apply_filter(
    file: UploadFile = File(...),
    filter: str = Query(..., description="Filter to apply: gaussian, median, bilateral, nlm, cnn")
):
    """Apply a specific denoising filter, compute metrics on-the-fly, and return them."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        input_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if input_img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format."})

        filter_name = filter.lower().strip()
        if filter_name not in FILTER_REGISTRY:
            return JSONResponse(status_code=400, content={"error": f"Unknown filter: {filter_name}"})

        finfo = FILTER_REGISTRY[filter_name]
        cleaned_img, metrics = run_filter_with_metrics(input_img, finfo["fn"])

        return {
            "status": "success",
            "filter_applied": filter_name,
            "algorithm": finfo["name"],
            "images": {
                "cleaned": mat_to_base64(cleaned_img)
            },
            "metrics": metrics
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})