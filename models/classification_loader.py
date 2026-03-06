import torch 
import timm 
import os 
import torch.nn as nn

class ClassificationModel(nn.Module):
    """MobileNetV2 for anemia classification"""
    def __init__(self, num_classes=2):
        super(ClassificationModel, self).__init__()
        self.backbone = timm.create_model(
            'mobilenetv2_100',
            pretrained=False,
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.backbone(x)
    
def load_classification_model(model_path, device):
    """Load classification model"""
    print(f"Loading classification model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classification model not found: {model_path}")
    
    model = ClassificationModel(num_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Classification model loaded")
    return model