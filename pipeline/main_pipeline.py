import time 
import cv2
from .segmentation import segment_conjunctiva
from .crop import extract_conjunctiva
from .classification import classify_anemia
from utils.save_results import save_pipeline_results
from utils.visualization import print_pipeline_summary, visualize_pipeline

def main_pipeline(image_path, seg_model, class_model, device, threshold=0.5,
                                save_results=False, output_dir="results"):
    """
    Complete pipeline: Segmentation → Crop → Classification
    
    Args:
        image_path: Path to input image
        seg_model: Segmentation model
        class_model: Classification model
        device: torch device
        save_results: Save intermediate results
        output_dir: Directory to save results
    
    Returns:
        dict: Complete results with all intermediate outputs
    """
    print("\n" + "="*60)
    print(" ANEMIA DETECTION AND CLASSIFICATION PIPELINE")
    print("="*60)
    print(f"Input image: {image_path}")
    
    start_time = time.time()
    
    # Load image
    print("\n Loading image...")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    print(f" Image loaded: {image.shape}")
    
    # Step 1: Segment conjunctiva
    mask, mask_overlay = segment_conjunctiva(image, seg_model, device,
        encoder_name='mobilenet_v2',
        encoder_weights='imagenet',
        threshold=threshold,
        smooth=True
    )
    
    # Step 2: Crop conjunctiva
    cropped = extract_conjunctiva(image, mask, background='black')
    
    # Step 3: Classify anemia
    result = classify_anemia(cropped, class_model, device)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Compile results
    pipeline_result = {
        'input_image': image,
        'mask': mask,
        'mask_overlay': mask_overlay,
        'cropped': cropped,
        'classification': result,
        'processing_time': total_time
    }
    
    # Save results
    if save_results:
        # Save images and get patient folder path
        patient_folder = save_pipeline_results(pipeline_result, image_path, output_dir)
        
        # Save visualization in same patient folder
        visualize_pipeline(pipeline_result, show=False, 
                          save_path=f"{patient_folder}/4_pipeline_visualization.png")
    
    # Print summary
    try:
        print_pipeline_summary(pipeline_result)
    except Exception as e:
        print(f" Warning: Could not print detailed summary: {e}")
        # Fallback summary
        print(f"\n Processing Time: {total_time:.2f} seconds")
        print(f" Result: {result['class_name']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
    
    print("="*60)
    
    return pipeline_result
