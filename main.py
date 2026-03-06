import os
import time
import lgpio

from config import config
from models import load_classification_model, load_segmentation_model
from pipeline import main_pipeline, capture_conjunctiva
from utils import visualize_pipeline, print_pipeline_summary


# LED SETUP
chip = None  # GPIO chip handle
LED_PIN = 26  # GPIO pin number for LED

def init_gpio():
    """Initialize GPIO (panggil sekali saja)"""
    global chip
    if chip is None:
        try:
            chip = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(chip, LED_PIN, 0)
            print("✓ GPIO initialized")
        except Exception as e:
            print(f"⚠ GPIO init failed: {e}")

def led_on():
    """Turn LED ON"""
    global chip
    if chip is not None:
        try:
            lgpio.gpio_write(chip, LED_PIN, 1)
        except:
            pass

def led_off():
    """Turn LED OFF"""
    global chip
    if chip is not None:
        try:
            lgpio.gpio_write(chip, LED_PIN, 0)
        except:
            pass

def cleanup_gpio():
    """Cleanup GPIO saat selesai"""
    global chip
    if chip is not None:
        try:
            lgpio.gpio_free(chip, LED_PIN)
            lgpio.gpiochip_close(chip)
            chip = None
            print("✓ GPIO cleaned up")
        except:
            pass


# MODEL LOADING
seg_model = None
class_model = None

def load_models():
    """Load models (dipanggil sekali saat startup)"""
    global seg_model, class_model
    
    if seg_model is None or class_model is None:
        print(" Loading models...")
        seg_model = load_segmentation_model(config.SEGMENTATION_MODEL, config.DEVICE)
        class_model = load_classification_model(config.CLASSIFICATION_MODEL, config.DEVICE)
        print(" Models loaded")
    
    return seg_model, class_model


# FUNGSI UNTUK API
def run_detection():
    """
    Fungsi untuk dipanggil dari API
    
    Returns:
        dict: {
            'status': 'normal' atau 'anemia',
            'confidence': 0.95,
            'image_path': '/patient_images/xxx.jpg'
        }
        atau None jika gagal
    """
    try:
        print("=" * 60)
        print(" STARTING ANEMIA DETECTION")
        print("=" * 60)
        
        # INIT GPIO
        init_gpio()
        
        #STEP 1: LOAD MODELS 
        print("\n[1/4] Loading AI models...")
        seg_model, class_model = load_models()
        
        # STEP 2: CAPTURE IMAGE
        print("\n[2/4] Capturing conjunctiva image...")
        image_path = capture_conjunctiva(
            save_dir="patient_images",
            show_preview=False,
            show_captured=False
        )
        
        if not image_path or not os.path.exists(image_path):
            print(" Capture failed!")
            cleanup_gpio()
            return None
        
        print(f" Captured: {image_path}")
        
        # STEP 3: RUN AI PIPELINE
        print("\n[3/4] Running AI pipeline (segmentation + classification)...")
        result = main_pipeline(
            image_path,
            seg_model,
            class_model,
            config.DEVICE,
            save_results=True,
            output_dir="results"
        )
        
        # STEP 4: EXTRACT RESULT 
        print("\n[4/4] Extracting results...")
        classification = result.get('classification', {})
        
        status = classification.get('class_name', 'unknown').lower()
        confidence = classification.get('confidence', 0.0)
        
        print(f" Classification: {status.upper()}")
        print(f" Confidence: {confidence*100:.2f}%")
        
        #LED ON 5 DETIK
        print("\n LED: ON for 5 seconds...")
        led_on()
        time.sleep(5)
        led_off()
        print(" LED: OFF")
        
        cleanup_gpio()
        
        print("\n" + "=" * 60)
        print(" DETECTION COMPLETE")
        print("=" * 60)
        
        # Return format untuk API
        return {
            'status': status,
            'confidence': confidence,
            'image_path': f"/patient_images/{os.path.basename(image_path)}"
        }
        
    except Exception as e:
        print(f"\n Detection error: {e}")
        import traceback
        traceback.print_exc()
        
        led_off()
        cleanup_gpio()
        
        return None


# MAIN FUNCTION
def main():
    """Main function for manual testing"""
    try:
        print("\n" + "=" * 60)
        print(" RASPBERRY PI - ANEMIA CLASSIFICATION SYSTEM")
        print("=" * 60)
        
        # INIT GPIO 
        init_gpio()
        
        # ===== STEP 1: CAPTURE IMAGE =====
        print("\n[1/5] 📸 Capturing conjunctiva image...")
        image_path = capture_conjunctiva(
            save_dir="patient_images",
            show_preview=True,
            show_captured=True
        )
        
        if not image_path:
            print("Capture failed!")
            cleanup_gpio()
            return
        
        print(f"✓ Image captured: {image_path}")
        print(f"✓ Size: {os.path.getsize(image_path) / 1024:.1f} KB")
        
        # STEP 2: CHECK MODELS
        print("\n[2/5]  Checking models...")
        if not os.path.exists(config.SEGMENTATION_MODEL):
            print(f" Segmentation model not found: {config.SEGMENTATION_MODEL}")
            cleanup_gpio()
            return
        
        if not os.path.exists(config.CLASSIFICATION_MODEL):
            print(f"Classification model not found: {config.CLASSIFICATION_MODEL}")
            cleanup_gpio()
            return
        
        print(" Models found")
        
        # STEP 3: LOAD MODELS
        print("\n[3/5]  Loading models...")
        seg_model = load_segmentation_model(config.SEGMENTATION_MODEL, config.DEVICE)
        class_model = load_classification_model(config.CLASSIFICATION_MODEL, config.DEVICE)
        print(" Models loaded")
        
        #STEP 4: RUN PIPELINE
        print("\n[4/5]  Running AI pipeline...")
        result = main_pipeline(
            image_path,
            seg_model,
            class_model,
            config.DEVICE,
            save_results=True,
            output_dir="results"
        )
        print(" Pipeline completed")
        
        #LED ON 5 DETIK 
        print("\n LED: ON for 5 seconds (processing complete)...")
        led_on()
        time.sleep(5)
        led_off()
        print(" LED: OFF")
        
        #STEP 5: VISUALIZE
        print("\n[5/5]  Generating visualization...")
        
        visualization_path = "results/pipeline_visualization.png"
        visualize_pipeline(
            result, 
            show=True,
            save_path=visualization_path
        )
        
        if os.path.exists(visualization_path):
            print(f" Visualization saved: {visualization_path}")
        
        # PRINT SUMMARY
        print("\n" + "=" * 60)
        print(" PROCESS COMPLETE")
        print("=" * 60)
        
        classification = result.get('classification', {})
        print(f"\n Result:")
        print(f"   Status: {classification.get('class_name', 'Unknown')}")
        print(f"   Confidence: {classification.get('confidence', 0)*100:.2f}%")
        print(f"   Visualization: {visualization_path}")
        print("\n" + "=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n Interrupted by user (Ctrl+C)")
        led_off()
        cleanup_gpio()
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        led_off()
        cleanup_gpio()
    
    finally:
        cleanup_gpio()


# ENTRY POINT
if __name__ == "__main__":
    main()
