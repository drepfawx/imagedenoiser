import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

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
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training with: {device}")
    
    model = SimpleCNNDenoiser().to(device)
    dataset = DenoisingDataset("dataset")
    
    if len(dataset) == 0:
        print("no images found in the 'dataset' folder")
        return
        
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    
    print("starting model training...")
    for epoch in range(epochs):
        running_loss = 0.0
        for noisy_imgs, clean_imgs in dataloader:
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
            
        print(f"epoch [{epoch+1}/{epochs}] | loss: {running_loss/len(dataloader):.5f}")
        
    # Save the trained model
    torch.save(model.state_dict(), "my_model.pth")
    print("training complete. saved model as 'my_model.pth'.")

if __name__ == "__main__":
    train()