import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import config


def get_segmentation_preprocessing():
    """Preprocessing for segmentation (ImageNet normalization)"""
    return A.Compose([
        A.Resize(config.SEG_IMG_SIZE, config.SEG_IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_classification_preprocessing():
    """Preprocessing for classification (ImageNet normalization)"""
    return A.Compose([
        A.Resize(config.CLASS_IMG_SIZE, config.CLASS_IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])