import torch
import torch.nn as nn
import os

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

# Find the model file path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'my_model.pth')

try:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        # optimize layout and compile graph with jit
        model.to(memory_format=torch.channels_last)
        model = torch.jit.script(model)
        print("Model loaded and JIT compiled successfully.")
    else:
        print(f"Warning: model weights file not found at {model_path}")
except Exception as e:
    print(f"Warning: could not load model weights: {e}")

# set pytorch thread count for execution
torch.set_num_threads(4)
