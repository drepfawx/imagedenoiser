import os
import scipy.io
import numpy as np
import math
import cv2
from scipy.special import gamma
from scipy.ndimage import gaussian_filter

# Find path to modelparameters.mat in the same directory as this file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_mat_path = os.path.join(current_dir, 'modelparameters.mat')

# Load pre-trained model parameters
try:
    model_mat = scipy.io.loadmat(model_mat_path)
    model_mu = model_mat['mu_prisparam']
    model_cov = model_mat['cov_prisparam']
except Exception as e:
    print(f"Warning: could not load modelparameters.mat from {model_mat_path}: {e}")
    model_mu = None
    model_cov = None

def generalized_gaussian_ratio(alpha):
    return (gamma(2.0/alpha)**2) / (gamma(1.0/alpha) * gamma(3.0/alpha))

def generalized_gaussian_ratio_inverse(k):
    a1 = -0.535707356
    a2 = 1.168939911
    a3 = -0.1516189217
    b1 = 0.9694429
    b2 = 0.8727534
    b3 = 0.07350824
    c1 = 0.3655157
    c2 = 0.6723532
    c3 = 0.033834

    if k < 0.131246:
        if k <= 0:
            return 0.1
        val = 3.0 / (4 * k**2)
        if val <= 0:
            return 0.1
        return 2 * math.log(27.0/16.0) / math.log(val)
    elif k < 0.448994:
        inner = a2**2 - 4*a1*a3 + 4*a1*k
        if inner < 0:
            return 0.2
        return (1/(2 * a1)) * (-a2 + math.sqrt(inner))
    elif k < 0.671256:
        inner = (b1 - b2*k)**2 - 4*b3*(k**2)
        if inner < 0:
            return 0.5
        return (1/(2*b3*k)) * (b1 - b2*k - math.sqrt(inner))
    elif k < 0.75:
        val = (3-4*k)/(4*c1)
        if val <= 0:
            return 1.5
        inner = c2**2 + 4*c3*math.log(val)
        if inner < 0:
            return 1.5
        return (1/(2*c3)) * (c2 - math.sqrt(inner))
    else:
        return 10.0

def estimate_aggd_params(x):
    x_left = x[x < 0]
    x_right = x[x >= 0]
    
    if x_left.size <= 1:
        stddev_left = 1.0
    else:
        stddev_left = math.sqrt((1.0/(x_left.size - 1)) * np.sum(x_left ** 2))
        
    if x_right.size <= 1:
        stddev_right = 1.0
    else:
        stddev_right = math.sqrt((1.0/(x_right.size - 1)) * np.sum(x_right ** 2))
        
    if stddev_right == 0:
        stddev_right = 1e-5
        
    mean_abs = np.mean(np.abs(x))
    mean_sq = np.mean(x**2)
    if mean_sq == 0:
        mean_sq = 1e-5
        
    r_hat = mean_abs**2 / mean_sq
    y_hat = stddev_left / stddev_right
    
    den = (y_hat**2 + 1) ** 2
    if den == 0:
        den = 1e-5
    R_hat = r_hat * (y_hat**3 + 1) * (y_hat + 1) / den
    
    alpha = generalized_gaussian_ratio_inverse(R_hat)
    if np.isnan(alpha) or alpha <= 0:
        alpha = 0.2
        
    alpha = max(0.15, alpha)
    
    beta_left = stddev_left * math.sqrt(gamma(3.0/alpha) / gamma(1.0/alpha))
    beta_right = stddev_right * math.sqrt(gamma(3.0/alpha) / gamma(1.0/alpha))
    return alpha, beta_left, beta_right

def compute_features(img_norm):
    features = []
    alpha, beta_left, beta_right = estimate_aggd_params(img_norm)
    features.extend([alpha, (beta_left + beta_right) / 2])

    for x_shift, y_shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        rolled = np.roll(np.roll(img_norm, y_shift, axis=0), x_shift, axis=1)
        img_pair_products = img_norm * rolled
        alpha, beta_left, beta_right = estimate_aggd_params(img_pair_products)
        eta = (beta_right - beta_left) * (gamma(2.0 / alpha) / gamma(1.0 / alpha))
        features.extend([alpha, eta, beta_left, beta_right])

    return features

def normalize_image(img, sigma=7/6):
    img = img.astype(np.float64)
    mu = gaussian_filter(img, sigma, mode='nearest')
    mu_sq = mu * mu
    sigma_map = np.sqrt(np.abs(gaussian_filter(img * img, sigma, mode='nearest') - mu_sq))
    img_norm = (img - mu) / (sigma_map + 1)
    return img_norm

def niqe(img):
    if model_mu is None or model_cov is None:
        return 0.0

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    gray = gray.astype(np.float64)
    
    features = None
    img_scaled = gray
    for scale in [1, 2]:
        if scale != 1:
            h, w = gray.shape
            img_scaled = cv2.resize(gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)

        img_norm = normalize_image(img_scaled)

        scale_features = []
        block_size = 96 // scale
        
        num_cols = max(1, img_norm.shape[0] // block_size)
        num_rows = max(1, img_norm.shape[1] // block_size)
        
        for block_col in range(num_cols):
            for block_row in range(num_rows):
                block = img_norm[
                    block_col * block_size : (block_col + 1) * block_size,
                    block_row * block_size : (block_row + 1) * block_size
                ]
                if block.shape[0] < block_size or block.shape[1] < block_size:
                    block = cv2.copyMakeBorder(
                        block, 0, block_size - block.shape[0], 0, block_size - block.shape[1],
                        cv2.BORDER_REPLICATE
                    )
                block_features = compute_features(block)
                scale_features.append(block_features)
                
        if len(scale_features) == 0:
            continue
            
        scale_features_arr = np.vstack(scale_features)
        if features is None:
            features = scale_features_arr
        else:
            features = np.hstack([features, scale_features_arr])

    if features is None or features.shape[0] < 2:
        return 0.0

    features_mu = np.mean(features, axis=0)
    features_cov = np.cov(features.T)

    pseudoinv_of_avg_cov = np.linalg.pinv((model_cov + features_cov) / 2)
    diff = model_mu.ravel() - features_mu
    niqe_quality = math.sqrt(diff.dot(pseudoinv_of_avg_cov.dot(diff)))
    return round(float(niqe_quality), 4)

