import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from datetime import datetime

_log_file = None

def log(msg):
    print(msg)
    if _log_file:
        _log_file.write(msg + "\n")
        _log_file.flush()

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

class DenoisingDataset(Dataset):
    def __init__(self, image_dir):
        all_files = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

        self.image_paths = []
        log(f"loading images from '{image_dir}'...")
        for f in tqdm(all_files, desc="Loading images", unit="img"):
            path = os.path.join(image_dir, f)
            try:
                img = Image.open(path)
                img.verify()  # validate that the file is a proper image
                self.image_paths.append(path)
            except Exception:
                log(f"  skipping invalid image: {f}")

        log(f"loaded {len(self.image_paths)} valid images (each will produce 20 random crops per epoch)")

        # we crop images into small patches for faster training and lower memory use
        self.transform = transforms.Compose([
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __len__(self):
        # multiply the dataset size so each image yields multiple crops per epoch
        return len(self.image_paths) * 20 

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_paths)
        image = Image.open(self.image_paths[real_idx]).convert('RGB')
        clean_img = self.transform(image)
        
        # Add synthetic Gaussian noise
        noise = torch.randn_like(clean_img) * 0.15
        noisy_img = torch.clamp(clean_img + noise, 0., 1.)
        
        return noisy_img, clean_img

# training loop
def train():
    global _log_file
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    log_path = os.path.join(parent_dir, "training_log.txt")
    _log_file = open(log_path, "w", encoding="utf-8")
    log(f"training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"training with: {device}")
    
    model = SimpleCNNDenoiser().to(device)
    
    dataset_dir = os.path.join(parent_dir, "dataset")
    dataset = DenoisingDataset(dataset_dir)
    
    if len(dataset) == 0:
        log(f"no images found in the '{dataset_dir}' folder")
        _log_file.close()
        return
        
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    
    log("starting model training...")
    for epoch in range(epochs):
        running_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch")
        
        for noisy_imgs, clean_imgs in progress_bar:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            
            optimizer.zero_grad()
            
            # predict a cleaned image
            outputs = model(noisy_imgs)
            
            # compare the prediction to the target image
            loss = criterion(outputs, clean_imgs)
            
            # update model weights
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.5f}")

        avg_loss = running_loss / len(dataloader)    
        log(f"epoch [{epoch+1}/{epochs}] | loss: {avg_loss:.5f}\n")
        
    model_save_path = os.path.join(parent_dir, "core", "my_model.pth")
    torch.save(model.state_dict(), model_save_path)
    log(f"training complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. saved model as '{model_save_path}'.")
    _log_file.close()

if __name__ == "__main__":
    train()
