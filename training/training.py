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
    def __init__(self, layers=[3], device= torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
        self.layers = layers
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights

    def forward(self, pred, target):
        loss = 0
        for layer in self.layers:
            pred_feat = self.vgg[:layer](pred)
            target_feat = self.vgg[:layer](target)
            loss += F.l1_loss(pred_feat, target_feat)
        return loss

def total_variation_loss(image):
    tv_h = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return tv_h + tv_w

def total_variation_loss(image):
    tv_h = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return tv_h + tv_w

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
    #loss = 0.4 * pixel_loss + 0.4 * ssim_loss + 0.2 * perceptual_loss
    loss=ssim_loss
    return loss


def compute_simple_loss(pred, target, perceptual_loss_fn=None, tv_weight=0.01):
    """
    Compute combined loss: L1 Loss + Perceptual Loss + Total Variation Loss.

    Args:
        pred (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth.
        perceptual_loss_fn (callable, optional): Function for perceptual loss.
        tv_weight (float): Weight for total variation loss.

    Returns:
        torch.Tensor: Combined loss value.
    """
    # Pixel-wise L1 loss
    l1_loss = F.l1_loss(pred, target)

    # Perceptual loss (if provided)
    perceptual_loss = 0
    if perceptual_loss_fn is not None:
        perceptual_loss = perceptual_loss_fn(pred, target)

    # Total variation loss
    tv_loss = total_variation_loss(pred)

    # Combine losses
    total_loss = l1_loss + perceptual_loss + tv_weight * tv_loss

    return total_loss


def train(config, train_loader, val_loader, train_mean, train_std):
    # Simplified Perceptual Loss
    perceptual_loss_fn = PerceptualLoss(device=device)


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

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()
    num_epochs=200
    for epoch in range(num_epochs):  # Fewer epochs for testing
        model.train()
        running_loss = 0.0
        for batch_idx, (images_org, images_crop) in enumerate(train_loader):
            images_org, images_crop = images_org.to(device), images_crop.to(device)

            optimizer.zero_grad()
            outputs = model(images_org)  # Forward pass

            #loss = compute_loss(outputs, images_crop, perceptual_loss_fn)

            denormalized_output = denormalize(outputs, train_mean, train_std)
            denormalized_target = denormalize(images_crop, train_mean, train_std)
            denormalized_output = denormalized_output.clamp(-1, 1)
            denormalized_target = denormalized_target.clamp(-1, 1)

            loss = criterion(denormalized_output, denormalized_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images_org, images_crop in val_loader:
                images_org, images_crop = images_org.to(device), images_crop.to(device)
                outputs = model(images_org)
                #loss = compute_loss(outputs, images_crop, perceptual_loss_fn)

                denormalized_output = denormalize(outputs, train_mean, train_std)
                denormalized_target = denormalize(images_crop, train_mean, train_std)
                denormalized_output = denormalized_output.clamp(-1, 1)
                denormalized_target = denormalized_target.clamp(-1, 1)

                loss = criterion(denormalized_output, denormalized_target)

                val_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss / len(val_loader))

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")

        # Visualization at last epoch
        if epoch % 10 == 0:
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
