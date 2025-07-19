import torch
from model import Generator  # or however your gen is defined
import torchvision.transforms as T
from PIL import Image
import io
import base64

z_dim = 100
device = 'cuda'  # or 'cuda' if GPU is used

# Load generator
gen = Generator(z_dim).to(device)
gen.load_state_dict(torch.load("generator.pth", map_location=device))
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
