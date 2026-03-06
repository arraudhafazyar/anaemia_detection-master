import cv2
import numpy as np


def crop_conjunctiva(image, mask, padding=20):
    """
    Crop conjunctiva - EXACT line 237-274 inference.py
    """
    print("\n  Step 2: Cropping conjunctiva...")
    
    # Line 248: Find non-zero pixels in mask
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # Line 251: Find contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        raise ValueError("No conjunctiva detected!")
    
    # Line 258: Get bounding box of all contours
    x_min, y_min = image.shape[1], image.shape[0]
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Line 267: Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    y_max = min(image.shape[0], y_max + padding)
    
    # Line 273: Crop
    cropped_image = image[y_min:y_max, x_min:x_max]
    
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    
    print(f"Cropped: {cropped_image.shape}")
    
    return cropped_image, bbox


def extract_conjunctiva(image, mask, background='black'):
    """
    Extract conjunctiva - EXACT line 277-315 inference.py
    """
    #Crop ke area segmented
    cropped_img, cropped_mask, bbox = crop_segmented_region(image, mask)
    
    if cropped_img is None:
        return None
    
    #Create output image
    if background == 'white':
        output = np.ones_like(cropped_img) * 255
    elif background == 'black':
        output = np.zeros_like(cropped_img)
    elif background == 'transparent':
        output = np.zeros((cropped_img.shape[0], cropped_img.shape[1], 4), dtype=np.uint8)
    else:
        output = np.ones_like(cropped_img) * 255
    
    #Copy only segmented area
    mask_3d = np.stack([cropped_mask] * 3, axis=2)
    
    if background == 'transparent':
        output[:, :, :3] = cropped_img
        output[:, :, 3] = (cropped_mask * 255).astype(np.uint8)
    else:
        output = np.where(mask_3d > 0.5, cropped_img, output)
    
    return output


def crop_segmented_region(image, mask, margin=10):
    """
    Helper function - EXACT line 237-274 inference.py
    """
    # Find non-zero pixels in mask
    if mask.max() <= 1:
        mask_binary = (mask > 0.5).astype(np.uint8)
    else:
        mask_binary = (mask > 127).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, None, None
    
    # Get bounding box
    x_min, y_min = image.shape[1], image.shape[0]
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Add margin
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(image.shape[1], x_max + margin)
    y_max = min(image.shape[0], y_max + margin)
    
    # Crop
    cropped_image = image[y_min:y_max, x_min:x_max]
    
    # Normalize mask untuk cropping
    if mask.max() > 1:
        mask_normalized = mask.astype(np.float32) / 255.0
    else:
        mask_normalized = mask.astype(np.float32)
    
    cropped_mask = mask_normalized[y_min:y_max, x_min:x_max]
    
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    
    return cropped_image, cropped_mask, bbox
