import cv2
import numpy as np
import torch
import torch.nn as nn
import io
import base64
import os
import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

@app.post("/api/process")
async def process_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        input_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if input_img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format."})

        # Basic noise analysis (estimates sigma and histogram spikes)
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        mad = np.median(np.abs(laplacian - np.median(laplacian))) / 0.6745
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        salt_spike = hist[255] / (hist[254] + 1e-5)
        pepper_spike = hist[0] / (hist[1] + 1e-5)
        
        final_sigma = float(mad)
        final_spike = float(max(salt_spike, pepper_spike))
        
        system_decision = "none"
        algorithm_used = "no significant noise detected"
        cleaned_img = input_img.copy()
        
        if final_spike > 3.5:
            system_decision = "salt_and_pepper"
            algorithm_used = "Median filter (OpenCV MedianBlur)"
            cleaned_img = cv2.medianBlur(input_img, 5)
            
        elif final_sigma > 12.0:
            system_decision = "gaussian"
            algorithm_used = "custom-trained CNN model"
            
            # running pytorch model
            img_input = input_img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                output_tensor = model(img_tensor)
            cleaned_img = output_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
            cleaned_img = np.clip(cleaned_img, 0, 255).astype(np.uint8)

        # saving cleaned image to disk currently (will prob remove it later)
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
                "selected_algorithm": algorithm_used
            },
            "images": {
                "noisy": mat_to_base64(input_img),
                "cleaned": mat_to_base64(cleaned_img)
            }
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})