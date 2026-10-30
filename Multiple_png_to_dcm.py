import os
import glob
import time
import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

def convert_single_png_to_dicom(png_path: str, dcm_path: str, series_uid: str, instance_num: int):
    # Load PNG image
    img = Image.open(png_path)
    img_array = np.array(img)
    
    if img_array.ndim == 3:
        img_array = img_array[:, :, 0]  # Extract grayscale channel if RGB

    # Header Meta Information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Create DICOM Dataset
    ds = FileDataset(dcm_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    # DICOM Patient & Series Attributes
    ds.Modality = 'CT'
    ds.PatientName = "Anonymous^Patient"
    ds.PatientID = "123456"
    ds.SeriesInstanceUID = series_uid
    ds.InstanceNumber = instance_num
    ds.ContentDate = time.strftime("%Y%m%d")
    ds.ContentTime = time.strftime("%H%M%S")

    # Image Dimensions & Pixel Attributes
    ds.Rows, ds.Columns = img_array.shape
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.PixelRepresentation = 0  # Unsigned integer

    if img_array.dtype == np.uint16:
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
    else:
        img_array = img_array.astype(np.uint8)
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7

    ds.PixelData = img_array.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.save_as(dcm_path)

def batch_convert_png_to_dicom(input_dir: str, output_dir: str):
    """
    Finds all .png files in input_dir and saves them as .dcm in output_dir
    with identical base names (e.g., slice_01.png -> slice_01.dcm).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a single UID for the entire slice series
    shared_series_uid = generate_uid()
    
    # Get sorted list of PNG files
    png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return

    for idx, png_path in enumerate(png_files, start=1):
        # Extract base name without extension
        base_name = os.path.splitext(os.path.basename(png_path))[0]
        
        # Build destination path with .dcm extension
        dcm_path = os.path.join(output_dir, f"{base_name}.dcm")
        
        convert_single_png_to_dicom(
            png_path=png_path, 
            dcm_path=dcm_path, 
            series_uid=shared_series_uid, 
            instance_num=idx
        )
        print(f"Converted: {os.path.basename(png_path)} -> {base_name}.dcm")

# Example Usage
input_folder = "./scan_photos"      # Directory containing .png files
output_folder = "./scan_dicom"   # Directory to output .dcm files

batch_convert_png_to_dicom(input_folder, output_folder)
