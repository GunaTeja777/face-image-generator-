import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import io
import base64

# Define Generator class (same as in your notebook)
class Generator(nn.Module):
    def __init__(self, z_dim=200, i_dim=16):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.i_dim = i_dim
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, i_dim * 16, 4, 1, 0),
            nn.BatchNorm2d(i_dim * 16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(i_dim * 16, i_dim * 8, 4, 2, 1),
            nn.BatchNorm2d(i_dim * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(i_dim * 8, i_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(i_dim * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(i_dim * 4, i_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(i_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(i_dim * 2, i_dim, 4, 2, 1),
            nn.BatchNorm2d(i_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(i_dim, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)

z_dim = 200  # Match your notebook's z_dim
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load generator
gen = Generator(z_dim).to(device)

# Check if checkpoint exists and load it
checkpoint_path = r"C:\face\face-image-generator--1\checkpoints\G-latest.pkl"
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen.eval()
    print("Generator checkpoint loaded successfully!")
except FileNotFoundError:
    print(f"Checkpoint not found at {checkpoint_path}")
    print("Using randomly initialized generator.")
    gen.eval()
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    gen.eval()

def generate_image():
    noise = torch.randn(1, z_dim).to(device)
    with torch.no_grad():
        fake = gen(noise).detach().cpu().squeeze(0)
        fake = (fake + 1) / 2  # [-1, 1] to [0, 1]
    img = T.ToPILImage()(fake)
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded
