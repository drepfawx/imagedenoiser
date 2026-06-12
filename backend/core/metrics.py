import cv2
import numpy as np
from brisque import BRISQUE
from .niqe import niqe as compute_niqe
from skimage.metrics import peak_signal_noise_ratio as _psnr, structural_similarity as _ssim

# patch brisque feature scaling to avoid float/double casting issue
def patched_scale_features(self, features):
    min_flat = np.array([float(m.item() if hasattr(m, 'item') else m) for m in self.scale_params['min_']], dtype=np.float64)
    max_flat = np.array([float(m.item() if hasattr(m, 'item') else m) for m in self.scale_params['max_']], dtype=np.float64)
    features_flat = np.array([float(f.item() if hasattr(f, 'item') else f) for f in features], dtype=np.float64)
    return -1 + (2.0 / (max_flat - min_flat) * (features_flat - min_flat))

BRISQUE.scale_features = patched_scale_features
brisque_scorer = BRISQUE()

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

def compute_psnr(original, restored):
    return round(float(_psnr(original, restored, data_range=255)), 2)

def compute_ssim(original, restored):
    g1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
    return round(float(_ssim(g1, g2, data_range=255)), 4)

def compute_mse(original, restored):
    raw = np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2)
    return round(float(raw / (255.0 ** 2)), 6)
