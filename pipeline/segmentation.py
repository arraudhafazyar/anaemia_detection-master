import segmentation_models_pytorch as smp
import torch
import cv2
import numpy as np

# POST-PROCESSING FUNCTIONS 
def smooth_mask(mask, kernel_size=5):
    mask_uint8 = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    mask_smoothed = cv2.GaussianBlur(mask_opened, (kernel_size, kernel_size), 0)
    return mask_smoothed.astype(np.float32) / 255.0

def refine_mask_edges(mask, feather_amount=3):
    mask_uint8 = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    mask_blurred = cv2.GaussianBlur(mask_eroded, (feather_amount*2 + 1, feather_amount*2 + 1), 0)
    return mask_blurred.astype(np.float32) / 255.0

# main function
def segment_conjunctiva(image, model, device, 
                        encoder_name='mobilenet_v2', encoder_weights='imagenet',
                        threshold=0.5, smooth=True):
    """
    Return:
        mask         float32 (0.0 ~ 1.0)   WAJIB dipakai untuk cropping/extract!
        mask_overlay  BGR image dengan overlay hijau (untuk display saja)
    """
    original_size = (image.shape[1], image.shape[0])
    input_size = (640, 640)

    # --- Preprocessing ---
    img_resized = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)
    img_preprocessed = preprocessing_fn(img_rgb)
    tensor = torch.from_numpy(img_preprocessed).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # --- Inference ---
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)

    if smooth:
        mask = probs[0, 0].cpu().numpy()                    # float 0~1
    else:
        mask = (probs > threshold).float()[0, 0].cpu().numpy()

    # Resize ke ukuran asli
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    if not smooth:
        mask = (mask > threshold).astype(np.float32)

    if smooth:
        mask = smooth_mask(mask, kernel_size=5)
        mask = refine_mask_edges(mask, feather_amount=3)

    # Overlay untuk ditampilkan
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0.5] = [0, 255, 0]      # hijau
    mask_overlay = cv2.addWeighted(image, 0.6, mask_colored, 0.4, 0)

    # RETURN mask sebagai FLOAT 0 to 1 (bukan uint8!!)
    return mask, mask_overlay
