import os
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image

def read_and_resize(folder_path):
    """Reads images from a folder and applies a transformation."""

    transformed_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".JPG", ".BMP",".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)

            # Apply transformation here (e.g., resize)
            img = img.resize((256, 256))
            img = img.convert("L") 
            # Convert to numpy array if needed
            img_array = np.array(img)
            
            transformed_images.append(img_array)

    return transformed_images

# Handle pair images such as original and cropped
class ImageDataset(Dataset):
    def __init__(self, images_org, images_crop):
        self.images_org = images_org
        self.images_crop = images_crop
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert numpy array to PyTorch tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale image
        ])
        
    def __len__(self):
        return len(self.images_org)
    
    def __getitem__(self, idx):
        image_org = self.images_org[idx]
        image_crop = self.images_crop[idx]
        
        # Apply transformation to both images
        image_org = self.transform(image_org)
        image_crop = self.transform(image_crop)
        
        # Ensure both images are on the correct device
        return image_org.to(device), image_crop.to(device)
