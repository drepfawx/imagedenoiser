import cv2
import numpy as np

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img, ratio):
    noisy = img.copy()
    h, w = img.shape[:2]
    n = int(h * w * ratio / 2)
    ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
    noisy[ys, xs] = 255
    ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
    noisy[ys, xs] = 0
    return noisy

def detect_noise_type(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    mad = np.median(np.abs(laplacian - np.median(laplacian))) / 0.6745
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    white_mask = (gray == 255).astype(np.uint8)
    black_mask = (gray == 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded_white = cv2.erode(white_mask, kernel, iterations=1)
    eroded_black = cv2.erode(black_mask, kernel, iterations=1)
    isolated_white = float(np.sum(white_mask) - np.sum(eroded_white))
    isolated_black = float(np.sum(black_mask) - np.sum(eroded_black))
    near_white_avg = float(np.mean(hist[250:255]))
    near_black_avg = float(np.mean(hist[1:6]))
    spike = float(max(isolated_white / (near_white_avg + 1e-5), isolated_black / (near_black_avg + 1e-5)))
    sigma = float(mad)
    if sigma <= 1.0:
        spike = 0.0
    if spike > 3.5 and sigma > 1.0:
        return "salt_and_pepper"
    elif sigma > 12.0:
        return "gaussian"
    return "none"
