import os
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch 
import sys
import random
 
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
# Read and resize the images
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


def view_random_image(dataset, model):
    # Randomly select an index
    idx = random.randint(0, len(dataset) - 1)
    
    # Retrieve the image pair
    image_org = dataset[idx]
    
    # Move the image to the appropriate device
    image_org = image_org.unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Get the model prediction
    model.eval()
    with torch.no_grad():
        predicted_image = model(image_org)
    
    # Reverse normalization for visualization
    image_org = image_org.cpu().numpy().squeeze() * 0.5 + 0.5  # Denormalize
    predicted_image = predicted_image.cpu().numpy().squeeze() * 0.5 + 0.5  # Denormalize
    
    # Plot the images
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(image_org, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(predicted_image, cmap='gray')
    axes[1].set_title("Predicted Image")
    axes[1].axis("off")
    
    plt.show()
    return image_org, predicted_image