import os
import torch
from torchvision.models import densenet121, DenseNet121_Weights

def main():
    save_dir = "data/weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "densenet121.pth")
    
    print("Downloading DenseNet-121 weights...")
    try:
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights=weights)
    except AttributeError:
        # Fallback for older torchvision versions
        model = densenet121(pretrained=True)
        
    torch.save(model.state_dict(), save_path)
    print(f"Weights successfully saved to: {save_path}")

if __name__ == "__main__":
    main()