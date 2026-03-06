import cv2
import torch 
from config import config
from .preprocessing import get_classification_preprocessing

def classify_anemia(cropped_image, class_model, device):
    """
    Classify anemia from cropped conjunctiva
    
    Args:
        cropped_image: Cropped conjunctiva (H, W, 3) BGR
        class_model: Classification model
        device: torch device
    
    Returns:
        dict: Result with class_name, confidence, probabilities
    """
    print("\n Step 3: Classifying anemia...")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    transform = get_classification_preprocessing()
    augmented = transform(image=image_rgb)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = class_model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
        class_id = predicted.item()
        class_name = config.CLASS_NAMES[class_id]
        confidence_value = confidence.item()
        all_probs = probs[0].cpu().numpy()
    
    result = {
        'class_name': class_name,
        'class_id': class_id,
        'confidence': confidence_value,
        'prob_anemia': all_probs[0],
        'prob_normal': all_probs[1]
    }
    
    print(f"Classification complete")
    print(f"   Prediction: {class_name}")
    print(f"   Confidence: {confidence_value*100:.2f}%")
    
    return result