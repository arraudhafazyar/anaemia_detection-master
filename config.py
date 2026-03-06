class Config:
    # Model paths
    SEGMENTATION_MODEL = "models/linknet_mobilenetv2_float32.pth"
    CLASSIFICATION_MODEL = "models/anemia_classifier_mobilenetv2_best.pth"
    
    # Image sizes
    SEG_IMG_SIZE = 640  # Segmentation input size
    CLASS_IMG_SIZE = 224  # Classification input size
    
    # Device (Raspberry Pi = CPU)
    DEVICE = "cpu"  # Raspberry Pi doesn't have GPU
    
    # Thresholds
    SEG_THRESHOLD = 0.5  # Segmentation confidence
    CLASS_THRESHOLD = 0.7  # Classification confidence for warning
    
    # Class names
    CLASS_NAMES = ['Anemia', 'Normal']
    
    # Camera settings (optional, for PiCamera)
    CAMERA_RESOLUTION = (640, 480)  # ← RESOLUSI CAPTURE (Full HD)
    USE_PICAMERA = True                # ← FLAG: Pakai PiCamera2 library
    CAMERA_FRAMERATE = 30

config = Config()