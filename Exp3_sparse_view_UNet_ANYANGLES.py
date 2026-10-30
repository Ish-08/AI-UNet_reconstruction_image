import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from skimage.transform import iradon
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

# =========================================================================
# 1. FIXED U-NET NEURAL NETWORK ARCHITECTURE
# =========================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class CTArtifactReductionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(1, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _match_and_cat(self, x_up, x_skip):
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x_up, x_skip], dim=1)

    def forward(self, x):
        x1 = self.inc(x)        
        x2 = self.down1(x1)     
        x3 = self.down2(x2)     
        
        x = self.up1(x3)                      
        x = self._match_and_cat(x, x2)        
        x = self.conv_up1(x)
        
        x = self.up2(x)                      
        x = self._match_and_cat(x, x1)        
        x = self.conv_up2(x)
        
        logits = self.outc(x)
        return self.sigmoid(logits)

# =========================================================================
# 2. DATASET PREPARATION (VARIABLE IMAGES & ANGLES)
# =========================================================================
class CTVolumeDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def prepare_dataset(sinogram_volume: np.ndarray, num_sparse_angles: int, num_images: int):
    total_slices, detector_pixels, total_angles = sinogram_volume.shape
    
    # Subsample slices if user specifies fewer input images
    if num_images is not None and num_images < total_slices:
        slice_indices = np.linspace(0, total_slices - 1, num_images, dtype=int)
        sinogram_volume = sinogram_volume[slice_indices]
        print(f"Selected {num_images} slices out of {total_slices} for dataset.")
    else:
        num_images = total_slices
        print(f"Using all {total_slices} available slices for dataset.")

    angles_full = np.linspace(0.0, 180.0, total_angles, endpoint=False)
    sparse_indices = np.linspace(0, total_angles - 1, num_sparse_angles, dtype=int)
    angles_sparse = angles_full[sparse_indices]

    sparse_inputs = []
    full_targets = []

    print(f"Generating reconstructions for {num_images} slices with {num_sparse_angles} angles...")
    for i in range(num_images):
        sino_full = sinogram_volume[i]
        
        # Ground Truth Target (180 angles)
        recon_full = iradon(sino_full, theta=angles_full, circle=True)
        recon_full_norm = (recon_full - recon_full.min()) / (recon_full.max() - recon_full.min() + 1e-8)
        
        # Sparse Input (N angles)
        sino_sparse = sino_full[:, sparse_indices]
        recon_sparse = iradon(sino_sparse, theta=angles_sparse, circle=True)
        recon_sparse_norm = (recon_sparse - recon_sparse.min()) / (recon_sparse.max() - recon_sparse.min() + 1e-8)

        sparse_inputs.append(recon_sparse_norm)
        full_targets.append(recon_full_norm)

    return np.array(sparse_inputs), np.array(full_targets), angles_full, angles_sparse, sparse_indices

# =========================================================================
# 3. EXPERIMENT EXECUTION WITH METRIC EVALUATION
# =========================================================================
def run_experiment(sinogram_npy_path: str, output_dir: str, num_sparse_angles: int = 9, num_images: int = None, epochs: int = 60):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Load 3D sinogram array
    sinogram_volume = np.load(sinogram_npy_path)

    # Build dataset dynamically based on slice count and angle count
    inputs, targets, angles_full, angles_sparse, sparse_indices = prepare_dataset(
        sinogram_volume, num_sparse_angles=num_sparse_angles, num_images=num_images
    )
    
    dataset = CTVolumeDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True)

    # Initialize Model, Loss, Optimizer
    model = CTArtifactReductionUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train Model
    print(f"\n=== TRAINING U-NET ({len(dataset)} Slices | {num_sparse_angles} Angles) ===")
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_in, batch_target in dataloader:
            batch_in, batch_target = batch_in.to(device), batch_target.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_in)
            loss = criterion(outputs, batch_target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_in.size(0)
            
        epoch_loss = running_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] - Loss (MSE): {epoch_loss:.6f}")

    # Evaluate on up to 9 sample slices from the dataset
    eval_count = min(9, len(dataset))
    random.seed(42)
    selected_indices = sorted(random.sample(range(len(dataset)), eval_count))

    model.eval()
    print(f"\n=== EVALUATING METRICS ON {eval_count} SAMPLE SLICES ===")

    fig, axes = plt.subplots(eval_count, 4, figsize=(16, 3 * eval_count))
    if eval_count == 1:
        axes = np.expand_dims(axes, axis=0)

    results_radon = []
    results_ai = []

    with torch.no_grad():
        for row_idx, slice_idx in enumerate(selected_indices):
            target_img = targets[slice_idx]                # Ground Truth (180 angles)
            input_sparse_img = inputs[slice_idx]          # Radon (N angles)
            
            # Predict with AI Model
            tensor_in = torch.tensor(input_sparse_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            pred_ai_img = model(tensor_in).squeeze().cpu().numpy()

            # --- Radon Metrics vs Ground Truth ---
            radon_mse = mse(target_img, input_sparse_img)
            radon_psnr = psnr(target_img, input_sparse_img, data_range=1.0)
            radon_ssim = ssim(target_img, input_sparse_img, data_range=1.0)
            results_radon.append({'mse': radon_mse, 'psnr': radon_psnr, 'ssim': radon_ssim})

            # --- AI U-Net Metrics vs Ground Truth ---
            ai_mse = mse(target_img, pred_ai_img)
            ai_psnr = psnr(target_img, pred_ai_img, data_range=1.0)
            ai_ssim = ssim(target_img, pred_ai_img, data_range=1.0)
            results_ai.append({'mse': ai_mse, 'psnr': ai_psnr, 'ssim': ai_ssim})

            # Save AI image
            ai_out_img = Image.fromarray((pred_ai_img * 255).astype(np.uint8))
            ai_out_img.save(os.path.join(output_dir, f"ai_recon_slice_{slice_idx:03d}.png"))

            # Plot columns
            axes[row_idx, 0].imshow(target_img, cmap='gray')
            axes[row_idx, 0].set_title(f"Slice {slice_idx}: Ground Truth (180°)")
            axes[row_idx, 0].axis('off')

            axes[row_idx, 1].imshow(input_sparse_img, cmap='gray')
            axes[row_idx, 1].set_title(f"Radon ({num_sparse_angles} Angles)\nSSIM: {radon_ssim:.3f} | PSNR: {radon_psnr:.1f}dB")
            axes[row_idx, 1].axis('off')

            axes[row_idx, 2].imshow(pred_ai_img, cmap='gray')
            axes[row_idx, 2].set_title(f"AI U-Net ({num_sparse_angles} Angles)\nSSIM: {ai_ssim:.3f} | PSNR: {ai_psnr:.1f}dB")
            axes[row_idx, 2].axis('off')

            diff_map = np.abs(target_img - pred_ai_img)
            axes[row_idx, 3].imshow(diff_map, cmap='hot')
            axes[row_idx, 3].set_title("AI Absolute Error Map")
            axes[row_idx, 3].axis('off')

    plt.suptitle(f"Reconstruction Comparison ({len(dataset)} Slices, {num_sparse_angles} Angles)", fontsize=14, y=1.002)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"analysis_{len(dataset)}slices_{num_sparse_angles}angles.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()

    # =========================================================================
    # DETAILED METRIC COMPARISON PRINT OUT
    # =========================================================================
    avg_radon_mse = np.mean([m['mse'] for m in results_radon])
    avg_radon_psnr = np.mean([m['psnr'] for m in results_radon])
    avg_radon_ssim = np.mean([m['ssim'] for m in results_radon])

    avg_ai_mse = np.mean([m['mse'] for m in results_ai])
    avg_ai_psnr = np.mean([m['psnr'] for m in results_ai])
    avg_ai_ssim = np.mean([m['ssim'] for m in results_ai])

    mse_reduction = ((avg_radon_mse - avg_ai_mse) / avg_radon_mse) * 100.0
    psnr_gain = avg_ai_psnr - avg_radon_psnr
    ssim_gain = avg_ai_ssim - avg_radon_ssim

    print("\n" + "=" * 85)
    print(f" EXPERIMENT RESULTS SUMMARY ({len(dataset)} Input Images | {num_sparse_angles} Angles)")
    print("=" * 85)
    print(f"{'Method':<30} | {'MSE (Lower=Better)':<20} | {'PSNR (dB)':<12} | {'SSIM':<8}")
    print("-" * 85)
    print(f"{f'Radon Method ({num_sparse_angles} Angles)':<30} | {avg_radon_mse:<20.6f} | {avg_radon_psnr:<12.2f} | {avg_radon_ssim:<8.4f}")
    print(f"{f'AI U-Net Model ({num_sparse_angles} Angles)':<30} | {avg_ai_mse:<20.6f} | {avg_ai_psnr:<12.2f} | {avg_ai_ssim:<8.4f}")
    print("=" * 85)
    print(" PERFORMANCE COMPARISON (AI vs Radon):")
    print(f"  - MSE Reduction : {mse_reduction:.2f}% improvement")
    print(f"  - PSNR Gain     : +{psnr_gain:.2f} dB")
    print(f"  - SSIM Increase : +{ssim_gain:.4f}")
    print("=" * 85 + "\n")

# =========================================================================
# RUN EXPERIMENT WITH ARGPARSE
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT Reconstruction: AI U-Net vs Radon Method")
    parser.add_argument("--angles", type=int, default=9, help="Number of projection angles (e.g., 9, 18, 36)")
    parser.add_argument("--images", type=int, default=None, help="Number of input slices/images to train on (e.g., 10, 20, 36, 72)")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--sinogram", type=str, default="./sinogram_outputs/sinogram_volume.npy", help="Path to sinogram .npy file")
    
    args = parser.parse_args()

    img_str = "all" if args.images is None else f"{args.images}imgs"
    output_directory = f"./exp_results_{img_str}_{args.angles}angles"

    run_experiment(
        sinogram_npy_path=args.sinogram,
        output_dir=output_directory,
        num_sparse_angles=args.angles,
        num_images=args.images,
        epochs=args.epochs
    )