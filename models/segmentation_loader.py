import torch
import segmentation_models_pytorch as smp
import os

def load_segmentation_model(model_path, device):
    """
    Load segmentation model LANGSUNG tanpa wrapper
    
    Args:
        model_path: Path to model checkpoint (.pth file)
        device: torch.device
    
    Returns:
        model: Loaded and ready model in eval mode
    """
    print(f"Loading segmentation model from: {model_path}")
    
    # Check file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # load  smp
    model = smp.Linknet(
        encoder_name='mobilenet_v2',
        encoder_weights=None,  # Will load from checkpoint
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Debug: Print checkpoint keys
    if 'model_state_dict' in checkpoint:
        print(f" Checkpoint contains 'model_state_dict'")
        state_dict = checkpoint['model_state_dict']
    else:
        print("Checkpoint is raw state_dict (no 'model_state_dict' key)")
        state_dict = checkpoint
    
    # Load weights dengan strict=True untuk detect issues
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model weights loaded successfully (strict mode)")
    except RuntimeError as e:
        print(f" Strict loading failed: {e}")
        print("   Trying to fix key mismatch...")
        
        # Fix key mismatch jika ada prefix 'model.'
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key.replace('model.', '', 1)  # Remove 'model.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict, strict=True)
        print("Model loaded after key fixing")
    
    # Set to eval mode
    model.eval()
    
    # Print model info
    print(f"   Device: {device}")
    print(f"   Model: LinkNet + MobileNetV2")
    
    if 'best_iou_score' in checkpoint:
        print(f"   Training Best IoU: {checkpoint['best_iou_score']:.4f}")
    if 'epoch' in checkpoint:
        print(f"   Training Epoch: {checkpoint['epoch']}")
    
    print("Segmentation model ready!\n")
    
    return model
