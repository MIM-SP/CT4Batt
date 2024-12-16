import os
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image

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


# View random pair of images
def view_random_image(dataset):
    # Randomly select an index
    idx = random.randint(0, len(dataset) - 1)
    
    # Retrieve the image pair
    image_org, image_crop = dataset[idx]
    
    # Reverse normalization for visualization
    image_org = image_org.cpu().numpy().squeeze() * 0.5 + 0.5  # Denormalize
    image_crop = image_crop.cpu().numpy().squeeze() * 0.5 + 0.5  # Denormalize
    # Plot the images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_org, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(image_crop, cmap='gray')
    axes[1].set_title("Cropped Image")
    axes[1].axis("off")
    
    plt.show()
    return image_crop
