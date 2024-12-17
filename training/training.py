import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.nn.functional as F
from utils.helper import denormalize, visualize_predictions
import torchvision.models as models
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from models.cnn_attention import CNNWithAttention as CNN_model
from models.dynamic_cnn_attention_roi import CNN2D as dynamic_CNN_model
from models.cnnKan_attention import CNNKanWithAttention as CNN_model_w_Kan
from models.dynamic_cnnKan_attention import CNN2DKan as dynamic_CNN_model_w_Kan
from pytorch_msssim import ssim
from ray import tune


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# Lightweight Perceptual Loss using VGG11
class PerceptualLoss(nn.Module):
    def __init__(self, layers=[2, 5, 8], loss_fn=nn.L1Loss(), device=None):
        """
        Lightweight Perceptual Loss using VGG11.

        Args:
            layers (list): VGG layer indices to extract features.
            loss_fn (nn.Module): Loss function to compare feature maps.
            device (torch.device): Device for running the VGG model.
        """
        super(PerceptualLoss, self).__init__()
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.vgg = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features.to(self.device).eval()
        
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights

        self.layers = layers
        self.loss_fn = loss_fn

    def forward(self, generated, target):
        loss = 0.0

        # Move tensors to device and ensure 3 channels
        generated = generated.to(self.device)
        target = target.to(self.device)
        if generated.shape[1] == 1 and target.shape[1] == 1:
            generated = generated.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        x_gen = generated
        x_target = target

        # Extract features and compute loss
        for idx, layer in enumerate(self.vgg):
            x_gen = layer(x_gen)
            x_target = layer(x_target)
            if idx in self.layers:
                loss += self.loss_fn(x_gen, x_target)

        return loss


# Compute Loss Function
def compute_loss(outputs, targets, perceptual_loss_fn):


    targets = F.interpolate(targets, size=outputs.shape[2:], mode='bilinear', align_corners=False)

    # Pixel-wise loss
    pixel_loss = F.l1_loss(outputs, targets)

    # Structural Similarity Index Loss
    ssim_loss = 1 - ssim(outputs, targets, data_range=1.0, size_average=True)

    # Perceptual Loss
    #perceptual_loss = perceptual_loss_fn(outputs, targets)

    # Weighted loss
    loss = 0.5 * pixel_loss + 0.5 * ssim_loss
    return loss

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def compute_simple_loss(outputs, targets):
    """Compute simple pixel-wise L1 loss."""
    # Resize targets to match outputs shape
    targets_resized = F.interpolate(targets, size=outputs.shape[2:], mode='bilinear', align_corners=False)

    return F.l1_loss(outputs, targets_resized)


def train(config, train_loader, val_loader, train_mean, train_std):
    # Simplified Perceptual Loss
    perceptual_loss_fn = PerceptualLoss(layers=[0, 2, 4], loss_fn=nn.L1Loss())  # Shallow layers

    # Model initialization
    '''
    model = CNN_model(  # Replace with CNN_model_w_Kan if needed
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        base=config["base"],
        kernel_size=config["kernel_size"],
        activation_fn=getattr(nn, config["activation_fn"])
    ).to(device)
    '''
    model = dynamic_CNN_model( #or dynamic_CNN_model_w_Kan
        in_channels=config["in_channels"], 
        out_channels=config["out_channels"], 
        kernel_size=config["kernel_size"], 
        ratio=config["ratio"], 
        stride=config["stride"], 
        num_kernels=config["num_kernels"], 
        temperature=config["temperature"],
        activation_fn=getattr(nn, config["activation_fn"]),
    ).to(device)
    model.apply(initialize_weights)  # Apply weight initialization

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    num_epochs=2000
    for epoch in range(num_epochs):  # Fewer epochs for testing
        model.train()
        running_loss = 0.0
        for batch_idx, (images_org, images_crop) in enumerate(train_loader):
            images_org, images_crop = images_org.to(device), images_crop.to(device)

            optimizer.zero_grad()
            outputs = model(images_org)  # Forward pass

            loss = compute_loss(outputs, images_crop, perceptual_loss_fn)
            #loss = compute_simple_loss(outputs, images_crop)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images_org, images_crop in val_loader:
                images_org, images_crop = images_org.to(device), images_crop.to(device)
                outputs = model(images_org)
                loss = compute_loss(outputs, images_crop, perceptual_loss_fn)
                #loss = compute_simple_loss(outputs, images_crop)

                val_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss / len(val_loader))

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")

        # Visualization at last epoch
        if epoch // 100:
            visualize_predictions(outputs,images_crop,train_mean,train_std)

            if hasattr(model, 'update_temperature'):
                model.update_temperature()
          
    torch.save(model.state_dict(), 'final_model.pth')
    print("Model saved as 'final_model.pth'")
    return {"loss":avg_val_loss}

# Load model function
def load_model(config, model_path='final_model.pth'):

    model = dynamic_CNN_model(#or dynamic_CNN_model_w_Kan
        in_channels=config["in_channels"], 
        out_channels=config["out_channels"], 
        kernel_size=config["kernel_size"], 
        ratio=config["ratio"], 
        stride=config["stride"], 
        num_kernels=config["num_kernels"], 
        temperature=config["temperature"],
        activation_fn=getattr(nn, config["activation_fn"]),
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Predict function
def predict(model, input_tensor):
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    return output
