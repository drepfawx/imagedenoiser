import cv2
import numpy as np
import torch
import time
from .model import model
from .metrics import compute_edge_preservation, brisque_scorer, compute_niqe

def apply_tiled_filter(input_img, filter_fn, tile_size=512, overlap=16):
    "generic wrapper that runs heavy filters in tiles to prevent timeouts when processing big images"
    h, w = input_img.shape[:2]
    
    if max(h, w) <= tile_size:
        return filter_fn(input_img)
        
    output_img = np.zeros_like(input_img)
    
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
                
            tile = padded_img[py_start:py_end, px_start:px_end]
            out_tile = filter_fn(tile)
            
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

    edge_preservation = compute_edge_preservation(input_img, cleaned)

    gray_input = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    gray_cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    
    input_lap_var = float(np.var(cv2.Laplacian(gray_input, cv2.CV_32F)))
    cleaned_lap_var = float(np.var(cv2.Laplacian(gray_cleaned, cv2.CV_32F)))
    
    if input_lap_var > 0:
        laplacian_var = round(100.0 * (cleaned_lap_var / input_lap_var), 2)
    else:
        laplacian_var = 100.0

    h_c, w_c = cleaned.shape[:2]
    max_metric_dim = 1024
    if max(h_c, w_c) > max_metric_dim:
        scale_m = max_metric_dim / max(h_c, w_c)
        metric_img = cv2.resize(cleaned, (int(w_c * scale_m), int(h_c * scale_m)), interpolation=cv2.INTER_AREA)
    else:
        metric_img = cleaned

    try:
        brisque_score = float(brisque_scorer.score(metric_img))
        brisque_score = round(brisque_score, 2)
    except Exception as e:
        print(f"brisque calculation failed: {e}")
        brisque_score = 0.0

    try:
        niqe_score = float(compute_niqe(metric_img))
        niqe_score = round(niqe_score, 2)
    except Exception as e:
        print(f"niqe calculation failed: {e}")
        niqe_score = 0.0

    Q = max(0.0, min(1.0, (100.0 - brisque_score) / 100.0))
    E = edge_preservation
    utility_score = round(float(0.5 * E + 0.5 * Q), 4)

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
    "nlm":       {"name": "Non-Local Means (NLM)",  "fn": lambda img: apply_tiled_filter(img, lambda tile: cv2.fastNlMeansDenoisingColored(tile, None, 10, 10, 7, 11), overlap=32)},
    "cnn":       {"name": "PyTorch CNN Denoiser",    "fn": run_cnn_filter},
}
