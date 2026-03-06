import os 
from pathlib import Path
import cv2
from datetime import datetime

def create_patient_folder(image_path, output_dir="results"):
    """
    Create unique folder for each patient
    Format: results/patient_name_YYYYMMDD_HHMMSS/
    """
    # Get base name without extension
    base_name = Path(image_path).stem
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create patient folder name
    patient_folder_name = f"{base_name}_{timestamp}"
    patient_folder = os.path.join(output_dir, patient_folder_name)
    
    # Create folder
    os.makedirs(patient_folder, exist_ok=True)
    
    print(f"\n?? Created patient folder: {patient_folder}")
    
    return patient_folder

def save_pipeline_results(result, image_path, output_dir="results"):
    """
    Save all intermediate results in patient-specific folder
    Automatically creates unique folder per patient
    """
    # Get base name without extension
    base_name = Path(image_path).stem
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create patient-specific folder
    patient_folder_name = f"{base_name}_{timestamp}"
    patient_folder = os.path.join(output_dir, patient_folder_name)
    os.makedirs(patient_folder, exist_ok=True)
    
    print(f"\n?? Created patient folder: {patient_folder}")
    
    # Save images (simpler names since already in patient folder)
    cv2.imwrite(f"{patient_folder}/1_original.jpg", result['input_image'])
    cv2.imwrite(f"{patient_folder}/2_mask_overlay.jpg", result['mask_overlay'])
    cv2.imwrite(f"{patient_folder}/3_cropped.jpg", result['cropped'])
    
    # Save mask
    cv2.imwrite(f"{patient_folder}/mask.png", result['mask'])
    
    print(f"?? Results saved to: {patient_folder}/")
    
    # Return patient folder path so visualization can use it
    return patient_folder
