import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon
from PIL import Image

def reconstruct_3d_ct_volume(sinogram_npy_path: str, output_dir: str):
    """
    Loads a 3D sinogram array (Slices, Detector_Pixels, Angles), applies
    Filtered Back Projection (iradon) to each slice, and saves the full 3D volume.
    """
    if not os.path.exists(sinogram_npy_path):
        raise FileNotFoundError(f"Could not find sinogram volume file: {sinogram_npy_path}")

    # 1. Load the 3D sinogram volume
    print(f"Loading sinogram volume: {sinogram_npy_path}")
    sinogram_volume = np.load(sinogram_npy_path)  # Shape: (Slices, Detector_Pixels, Angles)
    
    num_slices, detector_size, num_angles = sinogram_volume.shape
    print(f"Loaded Shape: {num_slices} Slices | {detector_size} Detector Pixels | {num_angles} Angles")

    # 2. Re-create matching projection angle array (0 to 180 degrees)
    theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)

    # Output directory for reconstructed PNG slices
    png_out_dir = os.path.join(output_dir, "reconstructed_slices_png")
    os.makedirs(png_out_dir, exist_ok=True)

    reconstructed_slices = []
    print("\nReconstructing 3D Volume slice-by-slice with iradon()...")

    # 3. Perform Inverse Radon Transform for each slice
    for idx in range(num_slices):
        sino_2d = sinogram_volume[idx]
        
        # Perform Filtered Back Projection (FBP)
        recon_2d = iradon(sino_2d, theta=theta, circle=True)
        reconstructed_slices.append(recon_2d)

        # Normalize slice for PNG visual save (0 - 255)
        slice_norm = (recon_2d - recon_2d.min()) / (recon_2d.max() - recon_2d.min() + 1e-8) * 255.0
        img = Image.fromarray(slice_norm.astype(np.uint8))
        img.save(os.path.join(png_out_dir, f"reconstructed_slice_{idx:03d}.png"))

    # 4. Stack all reconstructed 2D slices into a 3D volume: Shape (Z, Y, X)
    reconstructed_3d_volume = np.stack(reconstructed_slices, axis=0)

    # 5. Save the complete 3D array
    volume_out_path = os.path.join(output_dir, "reconstructed_3d_volume.npy")
    np.save(volume_out_path, reconstructed_3d_volume)

    print("\nReconstruction Summary:")
    print(f" - Reconstructed 3D Volume Shape: {reconstructed_3d_volume.shape} (Z, Y, X)")
    print(f" - Saved 3D Array: {volume_out_path}")
    print(f" - Saved Reconstructed PNG Slices: {png_out_dir}")

    return reconstructed_3d_volume

def plot_orthographic_views(volume_3d: np.ndarray):
    """
    Displays Axial, Coronal, and Sagittal slices from the center of the reconstructed 3D volume.
    """
    z_center = volume_3d.shape[0] // 2
    y_center = volume_3d.shape[1] // 2
    x_center = volume_3d.shape[2] // 2
    aspect_ratio = 800 / volume_3d.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial View (X-Y plane)
    axes[0].imshow(volume_3d[z_center, :, :], cmap='gray')
    axes[0].set_title(f"Axial Slice (Z = {z_center})")
    axes[0].axis('off')

    # Coronal View (X-Z plane)
    axes[1].imshow(volume_3d[:, y_center, :], cmap='gray', aspect = aspect_ratio)
    axes[1].set_title(f"Coronal Slice (Y = {y_center})")
    axes[1].axis('off')

    # Sagittal View (Y-Z plane)
    axes[2].imshow(volume_3d[:, :, x_center], cmap='gray', aspect = aspect_ratio)
    axes[2].set_title(f"Sagittal Slice (X = {x_center})")
    axes[2].axis('off')

    plt.suptitle("3D Reconstructed Volume Multi-Planar Views", fontsize=14)
    plt.tight_layout()
    # === LINE TO SAVE ANALYSIS IMAGE ===
    save_path = os.path.join(output_directory, "multi_planar_views.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved analysis plot to: {save_path}")
    plt.show()

# =========================================================================
# RUN RECONSTRUCTION
# =========================================================================
if __name__ == "__main__":
    sinogram_file = "./sinogram_outputs/sinogram_volume.npy"
    output_directory = "./reconstruction_output"

    volume_3d = reconstruct_3d_ct_volume(
        sinogram_npy_path=sinogram_file,
        output_dir=output_directory
    )

    # Plot 3D orthographic views
    plot_orthographic_views(volume_3d)
