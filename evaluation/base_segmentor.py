import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import datetime
from scipy.optimize import linear_sum_assignment

# Import the Attention2D class used in the DynamicCNN
from models.dynamic_cnn_attention_for_segmentation import DynamicCNN

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}.")

# Directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

images_dir = os.path.join(project_root, 'segmentor', 'masked_images')
results_dir = os.path.join(project_root, 'segmentor', 'results')

# Parameters
num_kernels = 50  # Max number of tabs. The network can produce up to this number.
in_channels = 1
out_channels = 1
batch_size = 1  # Batch size = 1 simplifies dealing with variable image sizes.
learning_rate = 0.001
num_epochs = 500


def compute_cost_matrix(pred, gt):
    """
    pred: [num_kernels, H, W] - predicted masks
    gt: [num_tabs, H, W] - ground truth masks

    Compute Dice-based cost in a vectorized manner:
    - intersection is computed via einsum
    - sums are computed once and broadcasted
    - no Python loops required
    """

    # intersection: shape [num_kernels, num_tabs]
    intersection = torch.einsum('ihw,jhw->ij', pred, gt)

    # sum_pred: [num_kernels], sum_gt: [num_tabs]
    sum_pred = pred.sum(dim=(1, 2))
    sum_gt = gt.sum(dim=(1, 2))

    # dice_coeff: [num_kernels, num_tabs]
    # Broadcasting:
    # sum_pred[:, None] gives shape [num_kernels, 1]
    # sum_gt[None, :] gives shape [1, num_tabs]
    dice_coeff = (2.0 * intersection) / (sum_pred[:, None] + sum_gt[None, :] + 1e-5)

    # cost_matrix: [num_kernels, num_tabs]
    cost_matrix = 1.0 - dice_coeff

    return cost_matrix


def hungarian_match(pred, gt):
    """
    pred: [num_kernels, H, W] (GPU)
    gt: [num_tabs, H, W] (GPU)

    Hungarian algorithm needs CPU/Numpy arrays.
    We'll:
    1. Compute cost on GPU
    2. Move cost matrix to CPU and convert to NumPy
    3. Run Hungarian on CPU
    4. Reorder pred on GPU
    """
    num_kernels, H, W = pred.shape
    num_tabs = gt.shape[0]

    if num_tabs == 0:
        # No tabs, return empty
        return gt.clone()  # Just return a zero tensor of shape [0, H, W]

    cost_matrix = compute_cost_matrix(pred, gt)   # stays on GPU

    # Move to CPU and Numpy for Hungarian
    cost_matrix_cpu = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix_cpu)

    # Reorder pred based on Hungarian solution
    # This reordering happens on GPU (pred and gt are on GPU)
    # We'll put result in perm_pred
    perm_pred = torch.zeros_like(gt)
    # Sort pairs by the ground-truth index (col_ind) so that perm_pred[i] corresponds to gt[i]
    sorted_pairs = sorted(zip(row_ind, col_ind), key=lambda x: x[1])
    for i, (pred_idx, gt_idx) in enumerate(sorted_pairs):
        perm_pred[i] = pred[pred_idx]

    return perm_pred


def sinkhorn_knopp(C, max_iter=50, eps=1e-3):
    """
    C: [num_kernels, num_tabs] cost matrix on GPU
    max_iter: number of iterations
    eps: regularization scaling

    Convert cost matrix to a doubly-stochastic matrix using the Sinkhorn-Knopp algorithm.
    We take negative cost scaled by 1/eps as logits, then apply iterative row/column normalization.
    """
    # Convert cost to logits (lower cost = higher logit)
    # Using negative cost / eps as "score"
    logits = -C / eps

    # Start with softmax over rows
    P = torch.softmax(logits, dim=1)  # Normalize rows

    for _ in range(max_iter):
        # Normalize columns
        P = P / (P.sum(dim=0, keepdim=True) + 1e-9)
        # Normalize rows
        P = P / (P.sum(dim=1, keepdim=True) + 1e-9)

    return P


def sinkhorn_match(pred, gt, max_iter=50, eps=1e-3):
    """
    pred: [num_kernels, H, W] predicted masks
    gt: [num_tabs, H, W] ground truth masks

    Use sinkhorn_knopp to approximate the optimal matching:
    1. Compute cost matrix.
    2. Run sinkhorn to get a doubly-stochastic matrix P.
    3. Extract assignments by picking max in each column.
    4. Reorder pred to match gt order.

    All operations stay on GPU.
    """
    num_kernels, H, W = pred.shape
    num_tabs = gt.shape[0]

    if num_tabs == 0:
        # No tabs, just return an empty corresponding shape
        return gt.clone()

    # Compute cost on GPU
    cost_matrix = compute_cost_matrix(pred, gt)  # [num_kernels, num_tabs]

    # Run sinkhorn-knopp
    P = sinkhorn_knopp(cost_matrix, max_iter=max_iter, eps=eps)
    # P is [num_kernels, num_tabs], a doubly-stochastic matrix approximating a permutation

    # Extract assignments:
    # For each gt tab (column), pick the predicted channel with max probability
    # col_ind: predicted channel per ground truth tab
    # By taking argmax along dim=0 (columns), we get max over each column
    # However, torch.argmax only works along a single dimension, so we transpose P.
    # Actually, let's just do P.T and argmax over dim=0 for clarity:
    # P.T: [num_tabs, num_kernels] - now each row is a gt tab, we pick max predicted channel
    _, col_ind = P.T.max(dim=1)  # col_ind[j] = matched pred for gt[j]

    perm_pred = torch.zeros_like(gt)
    for j in range(num_tabs):
        perm_pred[j] = pred[col_ind[j]]

    return perm_pred


class TabDataset(Dataset):
    """
    Custom Dataset that:
    - Loads a masked image from segmentor/masked_images
    - Loads a corresponding npy file from segmentor/results which contains the tab masks.

    The npy file shape: (num_tabs, H, W)
    The image shape: (H, W), can vary per image.

    Handling variable sizes:
    - Because images vary in height and width, we use batch_size=1.
    - The number of tabs (first dim) can vary, so we pad/truncate to num_kernels.
    - If label dimensions differ from image, we interpolate label to match.
    """

    def __init__(self, images_dir, results_dir, num_kernels=50, transform=None):
        self.images_dir = images_dir
        self.results_dir = results_dir
        self.transform = transform
        self.num_kernels = num_kernels

        # List all image files in masked_images
        self.image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        base_name = os.path.splitext(img_file)[0]

        # Load masked image as grayscale float32 [0,1]
        img_path = os.path.join(self.images_dir, img_file)
        img = Image.open(img_path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0
        # Convert to tensor [1, H, W]
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        # Load corresponding .npy label
        npy_path = os.path.join(self.results_dir, base_name + '.npy')
        if not os.path.exists(npy_path):
            # No mask file found, create a dummy zero mask
            label_tensor = torch.zeros((self.num_kernels, img_tensor.shape[-2], img_tensor.shape[-1]))
        else:
            label_array = np.load(npy_path).astype(np.float32)  # (num_tabs, H, W)
            label_tensor = torch.from_numpy(label_array)

            # Pad or truncate to num_kernels
            num_tabs, h, w = label_tensor.shape
            if num_tabs > self.num_kernels:
                label_tensor = label_tensor[:self.num_kernels]
            elif num_tabs < self.num_kernels:
                pad_tabs = self.num_kernels - num_tabs
                pad_tensor = torch.zeros((pad_tabs, h, w), dtype=label_tensor.dtype)
                label_tensor = torch.cat([label_tensor, pad_tensor], dim=0)

            # If label_tensor shape doesn't match image shape, interpolate label
            if label_tensor.shape[-2:] != img_tensor.shape[-2:]:
                label_tensor = nn.functional.interpolate(
                    label_tensor.unsqueeze(0),
                    size=img_tensor.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

        # Apply any optional transforms
        if self.transform:
            img_tensor, label_tensor = self.transform(img_tensor, label_tensor)

        return img_tensor, label_tensor


# Create dataset and split into train/val
dataset = TabDataset(images_dir, results_dir, num_kernels=num_kernels)

val_ratio = 0.2
dataset_size = len(dataset)
val_size = int(dataset_size * val_ratio)
train_size = dataset_size - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and loss
model = DynamicCNN(in_channels=in_channels, out_channels=out_channels, kernel_size=3, ratio=0.25,
                   num_kernels=num_kernels, temperature=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# TODO Consider using BCEWithLogitsLoss
criterion = nn.BCELoss()  # Binary cross-entropy for segmentation masks

# ---------------------------------------
# Training Loop with Hungarian or Sinkhorn approximation Matching
# ---------------------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for img_tensor, label_tensor in train_loader:
        img_tensor = img_tensor.to(device)  # [B, 1, H, W]
        label_tensor = label_tensor.to(device)  # [B, num_tabs, H, W]

        optimizer.zero_grad()
        output = model(img_tensor)  # [B, num_kernels, H, W]

        # Assuming batch_size=1 for simplicity:
        pred_single = output[0]  # [num_kernels, H, W]
        gt_single = label_tensor[0]  # [num_tabs, H, W]

        # Hungarian matching to align pred channels with ground truth tabs
        if gt_single.shape[0] > 0:  # If there are tabs
            perm_pred = hungarian_match(pred_single, gt_single)  # [num_tabs, H, W]
            perm_pred = perm_pred.unsqueeze(0)
            gt_single = gt_single.unsqueeze(0)

            loss = criterion(perm_pred, gt_single)
        else:
            # No tabs? Then no matching needed, just ensure no tabs predicted
            # Or consider a special case. For now, if no tabs exist:
            # Use output directly and expect it to predict zero masks
            # We'll just sum over all predicted channels (they should be near zero)
            loss = criterion(output, torch.zeros_like(output))

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for img_tensor, label_tensor in val_loader:
            img_tensor = img_tensor.to(device)
            label_tensor = label_tensor.to(device)
            output = model(img_tensor)

            pred_single = output[0]
            gt_single = label_tensor[0]

            if gt_single.shape[0] > 0:
                perm_pred = hungarian_match(pred_single, gt_single)
                # perm_pred = sinkhorn_match(pred_single, gt_single, max_iter=50, eps=1e-3)
                perm_pred = perm_pred.unsqueeze(0)
                gt_single = gt_single.unsqueeze(0)
                v_loss = criterion(perm_pred, gt_single)
            else:
                v_loss = criterion(output, torch.zeros_like(output))

            val_loss += v_loss.item()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {running_loss / len(train_loader):.4f} Val Loss: {val_loss / len(val_loader):.4f}")

# Save the final model
torch.save(model.state_dict(), os.path.join(project_root, 'final_model.pth'))
print("Training completed.")

# ---------------------------------------
# Inference (Testing/Prediction)
# ---------------------------------------
timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
output_results_dir = os.path.join(project_root, 'evaluation', f"results_run_{timestamp}")
os.makedirs(output_results_dir, exist_ok=True)
print(f"Results will be saved in: {output_results_dir}")

model.eval()
with torch.no_grad():
    for idx, (img_tensor, _) in enumerate(dataset):
        img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(img_tensor)  # [1, num_kernels, H, W]

        # At inference, we don't know num_tabs and we don't have gt, so no Hungarian matching here.
        # Just output the probabilities or threshold them:
        # binary_output = (output > 0.5).float() # If you want hard binary masks

        output_np = output.squeeze(0).cpu().numpy()  # shape: [num_kernels, H, W]

        base_name = os.path.splitext(dataset.image_files[idx])[0]
        output_path = os.path.join(output_results_dir, f"{base_name}.npy")
        np.save(output_path, output_np)
        print(f"Saved prediction: {output_path}")

print("Final predictions saved.")