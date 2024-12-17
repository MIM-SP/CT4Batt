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

# Read and resize the images
class ImageDataset(Dataset):
    def __init__(self, org_dir, crop_dir, mean=None, std=None):
        """
        Args:
            org_dir (str): Directory containing the original images.
            crop_dir (str): Directory containing the cropped images.
            mean (list): Mean for normalization.
            std (list): Std for normalization.
        """
        self.mean = mean
        self.std = std

        # Create mappings: {filename_without_ext: full_path}
        self.org_files = {os.path.splitext(f)[0]: os.path.join(org_dir, f)
                          for f in os.listdir(org_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".bmp", ".png"))}

        self.crop_files = {os.path.splitext(f)[0]: os.path.join(crop_dir, f)
                           for f in os.listdir(crop_dir)
                           if f.lower().endswith((".jpg", ".jpeg", ".bmp", ".png"))}

        # Find common keys (matching filenames without extensions)
        self.common_keys = sorted(set(self.org_files.keys()) & set(self.crop_files.keys()))

        if len(self.common_keys) == 0:
            raise ValueError("No matching filenames found between original and cropped images.")

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to 256x256
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean, std) if mean and std else lambda x: x  # Normalize if mean/std are provided
        ])

    def __len__(self):
        return len(self.common_keys)

    def __getitem__(self, idx):
        key = self.common_keys[idx]

        # Paths to the original and cropped images
        org_path = self.org_files[key]
        crop_path = self.crop_files[key]

        # Load images as grayscale
        image_org = Image.open(org_path).convert("L")
        image_crop = Image.open(crop_path).convert("L")

        # Apply transformations
        image_org = self.transform(image_org)
        image_crop = self.transform(image_crop)

        return image_org, image_crop

class normalize_img(Dataset):
    def __init__(self, images_org, images_crop, mean=None, std=None):
        """
        Args:
            images_org (list): List of original images (file paths or PIL Images).
            images_crop (list): List of cropped images (file paths or PIL Images).
            mean (float): Mean for normalization.
            std (float): Std for normalization.
        """
        self.images_org = images_org
        self.images_crop = images_crop
        self.mean = mean
        self.std = std

        # Transformation: ensures grayscale and normalization
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure single channel
            transforms.ToTensor(),  # Convert to Tensor, shape [1, H, W]
            transforms.Normalize(mean=[mean] if mean else [0.0], 
                                 std=[std] if std else [1.0])  # Normalize if provided
        ])
        
    def __len__(self):
        return len(self.images_org)
    
    def __getitem__(self, idx):
        # Load original and cropped images
        image_org = self.images_org[idx]
        image_crop = self.images_crop[idx]

        # If images are paths, open them
        if isinstance(image_org, str):
            image_org = Image.open(image_org).convert("L")  # Grayscale
        if isinstance(image_crop, str):
            image_crop = Image.open(image_crop).convert("L")  # Grayscale

        # Apply transformation
        image_org = self.transform(image_org)
        image_crop = self.transform(image_crop)
        
        return image_org, image_crop
    
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

def denormalize(tensor, mean, std):
    """
    Denormalize a tensor image given the mean and std.
    Args:
        tensor (torch.Tensor): Image tensor (C, H, W)
        mean (float or list): Mean used for normalization
        std (float or list): Std used for normalization
    """
    if isinstance(mean, (int, float)):
        mean = [mean] * tensor.size(0)
    if isinstance(std, (int, float)):
        std = [std] * tensor.size(0)

    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return tensor * std + mean  # Undo normalization

def denormalize2(tensor, mean, std):
    """
    Denormalize a tensor image given the mean and std.
    Args:
        tensor (torch.Tensor): Image tensor (C, H, W)
        mean (float or list): Mean used for normalization
        std (float or list): Std used for normalization
    """
    if isinstance(mean, (int, float)):
        mean = [mean] * tensor.size(0)
    if isinstance(std, (int, float)):
        std = [std] * tensor.size(0)

    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return tensor * std + mean  # Undo normalization

def visualize_predictions(outputs, targets, mean, std, idx=2):
    """
    Visualize the denormalized outputs and targets for the [-1, 1] range.

    Args:
        outputs (torch.Tensor): Model predictions, shape (N, C, H, W) or (C, H, W).
        targets (torch.Tensor): Ground truth, shape (N, C, H, W) or (C, H, W).
        mean (float or list): Mean used for normalization.
        std (float or list): Std used for normalization.
        idx (int): Index of the image to visualize in the batch.
    """
    # Ensure outputs and targets are batched tensors
    if outputs.dim() == 3:
        outputs = outputs.unsqueeze(0)  # Add batch dimension
    if targets.dim() == 3:
        targets = targets.unsqueeze(0)  # Add batch dimension

    # Safeguard index range
    if idx >= outputs.size(0):
        raise ValueError(f"Index {idx} is out of range for batch size {outputs.size(0)}")

    # Select the specific image
    output = outputs[idx]  # (C, H, W)
    target = targets[idx]  # (C, H, W)

    # Denormalize the images
    denormalized_output = denormalize(output, mean, std)
    denormalized_target = denormalize(target, mean, std)

    # Map [-1, 1] to [0, 1] for visualization
    output_np = ((denormalized_output.clamp(-1, 1) + 1) / 2).cpu().detach().numpy()
    target_np = ((denormalized_target.clamp(-1, 1) + 1) / 2).cpu().detach().numpy()

    # Convert to (H, W, C) for visualization
    output_np = output_np.transpose(1, 2, 0).squeeze()
    target_np = target_np.transpose(1, 2, 0).squeeze()

    # Plot side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(output_np, cmap='gray' if output_np.ndim == 2 else None)
    axes[0].set_title(f"Prediction (Image {idx})")
    axes[0].axis('off')

    axes[1].imshow(target_np, cmap='gray' if target_np.ndim == 2 else None)
    axes[1].set_title(f"Original Target (Image {idx})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def calculate_mean_std(loader):
    """
    Calculate mean and standard deviation of a dataset.
    Args:
        dataset: PyTorch dataset with images.
    Returns:
        mean: Mean of the dataset.
        std: Standard deviation of the dataset.
    """
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for images_org, _ in loader:
        batch_samples = images_org.size(0)
        images_org = images_org.view(batch_samples, -1)  # Flatten to (batch_size, num_pixels)
        mean += images_org.mean(dim=1).sum().item()
        std += images_org.std(dim=1).sum().item()
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    
    return mean, std

class NormalizedSubset(Dataset):
    """Wrapper for Subset to apply normalization."""
    def __init__(self, subset, mean, std):
        self.subset = subset
        self.transform = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        img, target = self.subset[idx]
        img = self.transform(img)  # Apply normalization
        return img, target