import os
import glob
import numpy as np
import pydicom
from skimage.transform import radon
from PIL import Image
import matplotlib.pyplot as plt

def process_dicoms_to_sinograms(dicom_dir: str, output_dir: str, num_angles: int = 180):
    """
    Reads DICOM files from dicom_dir, calculates a 2D sinogram for each slice,
    saves sinogram images, and outputs a 3D array of all sinograms.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Locate and sort DICOM files
    dcm_files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
    if not dcm_files:
        print(f"No DICOM files found in directory: {dicom_dir}")
        return

    # Define projection angles (0 to 180 degrees)
    theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)
    
    sinogram_volume = []
    print(f"Processing {len(dcm_files)} DICOM files into sinograms...")

    for path in dcm_files:
        base_name = os.path.splitext(os.path.basename(path))[0]
        
        # Read DICOM file and extract image array
        ds = pydicom.dcmread(path)
        img_data = ds.pixel_array.astype(np.float32)

        # Apply Rescale Slope / Intercept if available in DICOM header
        slope = getattr(ds, 'RescaleSlope', 1.0)
        intercept = getattr(ds, 'RescaleIntercept', 0.0)
        hu_image = img_data * slope + intercept

        # Forward Radon Transform to generate 2D Sinogram
        sinogram = radon(hu_image, theta=theta, circle=True)
        sinogram_volume.append(sinogram)

        # Normalize sinogram array to 8-bit image range (0–255) and save as PNG
        sino_norm = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min() + 1e-8) * 255.0
        sino_img = Image.fromarray(sino_norm.astype(np.uint8))
        sino_png_path = os.path.join(output_dir, f"{base_name}_sinogram.png")
        sino_img.save(sino_png_path)
        
        print(f"Generated: {base_name}.dcm -> {os.path.basename(sino_png_path)}")

    # Stack all slice sinograms into a 3D volume array: (Slices, Detector_Pixels, Angles)
    sinogram_volume = np.stack(sinogram_volume, axis=0)
    
    # Save complete 3D array for downstream CT reconstruction algorithms
    volume_path = os.path.join(output_dir, "sinogram_volume.npy")
    np.save(volume_path, sinogram_volume)
    print(f"\n3D Sinogram Volume saved to: {volume_path}")
    print(f"Volume Shape: {sinogram_volume.shape} (Slices, Detector Pixels, Projection Angles)")

    # Display plot preview for the first slice sinogram
    plt.figure(figsize=(8, 5))
    plt.imshow(sinogram_volume[0], cmap='magma', aspect='auto',
               extent=[0, 180, 0, sinogram_volume[0].shape[0]])
    plt.title(f"Sinogram Preview: {os.path.basename(dcm_files[0])}")
    plt.xlabel(r"Projection Angle $\theta$ (Degrees)")
    plt.ylabel(r"Detector Position $r$ (Pixels)")
    plt.colorbar(label="Projection Intensity")
    plt.tight_layout()
    plt.show()

# Example Usage
dicom_input_folder = "./scan_dicom"
sinogram_output_folder = "./sinogram_outputs"

process_dicoms_to_sinograms(
    dicom_dir=dicom_input_folder, 
    output_dir=sinogram_output_folder, 
    num_angles=180
)
