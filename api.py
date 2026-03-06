#!/usr/bin/env python3
"""
ANEMIA DETECTION API

Endpoints:
- GET  /api/health
- POST /api/camera/start
- POST /api/camera/stop
- GET  /api/video_feed
- POST /api/capture
- POST /api/analyze

"""

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import sys
import os
import time
import threading
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
try:
    from main import init_gpio, led_on, led_off, cleanup_gpio
    LED_AVAILABLE = True
except:
    LED_AVAILABLE = False

# CONFIGURATION
PROJECT_PATH = '/home/pi/fazya/anaemia_detection-master'
sys.path.insert(0, PROJECT_PATH)
os.chdir(PROJECT_PATH)

# Folder untuk menyimpan gambar
CAPTURE_DIR = os.path.join(PROJECT_PATH, 'patient_images')
RESULTS_DIR = os.path.join(PROJECT_PATH, 'results')
Path(CAPTURE_DIR).mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)

# FLASK APP
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# GLOBAL VARIABLES
picam2 = None
camera_lock = threading.Lock()
is_streaming = False
stream_thread = None

# CHECK MODULES
AI_AVAILABLE = False
SENSOR_AVAILABLE = False
CAMERA_AVAILABLE = False

# Check Camera (Picamera2)
try:
    from picamera2 import Picamera2
    # Cek apakah ada kamera yang terdeteksi
    cam_info = Picamera2.global_camera_info()
    if len(cam_info) > 0:
        CAMERA_AVAILABLE = True
        print(f" Picamera2 loaded - Found {len(cam_info)} camera(s)")
        print(f"  Camera 0: {cam_info[0].get('Model', 'Unknown')}")
    else:
        print("No camera detected")
except ImportError as e:
    print(f"Picamera2 not available: {e}")
except Exception as e:
    print(f"Camera detection error: {e}")

# Check AI modules
try:
    from config import config
    from models import load_classification_model, load_segmentation_model
    from pipeline.main_pipeline import main_pipeline
    AI_AVAILABLE = True
    print(" AI modules loaded")
except ImportError as e:
    print(f"AI modules not available: {e}")

# Check Sensor
try:
    from max30100 import MAX30100
    SENSOR_AVAILABLE = True
    print("Sensor loaded")
except ImportError as e:
    print(f"Sensor not available: {e}")

# AI MODELS
seg_model = None
class_model = None

def load_ai_models():
    global seg_model, class_model
    if not AI_AVAILABLE:
        return None, None
    if seg_model is None or class_model is None:
        print("Loading models...")
        seg_model = load_segmentation_model(config.SEGMENTATION_MODEL, config.DEVICE)
        class_model = load_classification_model(config.CLASSIFICATION_MODEL, config.DEVICE)
        print("models loaded")
    return seg_model, class_model

# CAMERA FUNCTIONS 
def init_camera():
    """Initialize Picamera2 untuk Pi Camera v3"""
    global picam2

    if not CAMERA_AVAILABLE:
        print("Camera not available")
        return False

    with camera_lock:
        # Jika sudah ada instance, return True
        if picam2 is not None:
            try:
                # Test apakah masih bisa capture
                picam2.capture_array()
                return True
            except:
                # Kamera error, coba restart
                try:
                    picam2.stop()
                    picam2.close()
                except:
                    pass
                picam2 = None

        try:
            print("Initializing camera...")

            # Buat instance baru
            picam2 = Picamera2()

            # Konfigurasi untuk streaming
            # Gunakan RGB888 - hasil test menunjukkan ini yang benar
            config_cam = picam2.create_video_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                buffer_count=4
            )

            picam2.configure(config_cam)

            # Start kamera
            picam2.start()

            # Tunggu kamera siap
            time.sleep(1)

            # Set AWB Auto dan Autofocus untuk Pi Camera v3
            try:
                picam2.set_controls({
                    "AwbEnable": True,
                    "AwbMode": 0,      # Auto White Balance
                    "AeEnable": True,  # Auto Exposure
                    "AfMode": 2,       # Continuous autofocus
                    "AfTrigger": 0     # Start autofocus
                })
                print("  AWB Auto + Autofocus enabled")
            except Exception as ctrl_err:
                print(f"  Controls warning: {ctrl_err}")

            # Tunggu AWB dan autofocus settle (penting!)
            print("  Waiting for AWB to settle...")
            time.sleep(2)

            # Test capture
            test_frame = picam2.capture_array()
            print(f"Camera initialized - Frame shape: {test_frame.shape}")

            return True

        except Exception as e:
            print(f"Camera init error: {e}")
            import traceback
            traceback.print_exc()

            # Cleanup jika gagal
            if picam2 is not None:
                try:
                    picam2.stop()
                    picam2.close()
                except:
                    pass
                picam2 = None

            return False

def stop_camera():
    """Stop dan release kamera"""
    global picam2, is_streaming

    print("Stopping camera...")

    is_streaming = False

    with camera_lock:
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
                print("Camera stopped")
            except Exception as e:
                print(f"Camera stop warning: {e}")
            finally:
                picam2 = None

def generate_frames():
    """Generator untuk MJPEG streaming"""
    global picam2, is_streaming

    print("Starting frame generation...")
    frame_count = 0

    while is_streaming:
        try:
            if picam2 is None:
                print("Camera is None, stopping stream")
                break

            # Capture frame dengan lock
            with camera_lock:
                if picam2 is None:
                    break
                frame = picam2.capture_array()

            # Tambahkan overlay text
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Live | {timestamp}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Arahkan ke konjungtiva mata",
                       (10, 460), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1)

            # Encode ke JPEG - langsung tanpa konversi
            ret, buffer = cv2.imencode('.jpg', frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Streamed {frame_count} frames")

            # ~30 FPS
            time.sleep(0.033)

        except Exception as e:
            print(f"Stream error: {e}")
            import traceback
            traceback.print_exc()
            break

    print(f"Stream ended after {frame_count} frames")

# SENSOR FUNCTION
def read_sensor(duration=10, warmup=3):
    """Baca sensor MAX30100"""
    if not SENSOR_AVAILABLE:
        return {'heart_rate': 0, 'spo2': 0, 'error': 'Sensor not available'}

    try:
        sensor = MAX30100()

        if hasattr(sensor, 'check_sensor') and not sensor.check_sensor():
            return {'heart_rate': 0, 'spo2': 0, 'error': 'Sensor not detected'}

        if hasattr(sensor, 'setup'):
            sensor.setup()

        # Try import HeartRateMonitor
        try:
            from max30100 import HeartRateMonitor
            monitor = HeartRateMonitor()
            use_monitor = True
        except:
            use_monitor = False

        print(f"Reading sensor for {duration}s...")
        start_time = time.time()
        bpm_samples = []
        spo2_samples = []

        while time.time() - start_time < duration:
            try:
                if hasattr(sensor, 'read_fifo'):
                    ir, red = sensor.read_fifo()
                else:
                    break

                if time.time() - start_time > warmup and use_monitor:
                    monitor.add_sample(ir, red)
                    if hasattr(monitor, 'is_finger_detected') and monitor.is_finger_detected():
                        bpm = monitor.calculate_bpm() if hasattr(monitor, 'calculate_bpm') else 0
                        spo2 = monitor.calculate_spo2() if hasattr(monitor, 'calculate_spo2') else 0
                        if 40 < bpm < 200:
                            bpm_samples.append(bpm)
                        if 80 < spo2 < 100:
                            spo2_samples.append(spo2)

                time.sleep(0.05)
            except:
                continue

        if hasattr(sensor, 'close'):
            sensor.close()

        # Calculate averages
        result = {'heart_rate': 0, 'spo2': 0}

        if bpm_samples:
            bpm_sorted = sorted(bpm_samples)
            trim = len(bpm_sorted) // 5
            bpm_trimmed = bpm_sorted[trim:-trim] if trim > 0 and len(bpm_sorted) > 2 else bpm_sorted
            if bpm_trimmed:
                result['heart_rate'] = int(sum(bpm_trimmed) / len(bpm_trimmed))

        if spo2_samples:
            spo2_sorted = sorted(spo2_samples)
            trim = len(spo2_sorted) // 5
            spo2_trimmed = spo2_sorted[trim:-trim] if trim > 0 and len(spo2_sorted) > 2 else spo2_sorted
            if spo2_trimmed:
                result['spo2'] = int(sum(spo2_trimmed) / len(spo2_trimmed))

        return result

    except Exception as e:
        return {'heart_rate': 0, 'spo2': 0, 'error': str(e)}

# API ROUTES
@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint: GET /api/health"""
    return jsonify({
        'status': 'success',
        'ai_available': AI_AVAILABLE,
        'sensor_available': SENSOR_AVAILABLE,
        'camera_available': CAMERA_AVAILABLE,
        'camera_initialized': picam2 is not None,
        'is_streaming': is_streaming
    })


@app.route('/api/camera/start', methods=['POST'])
def camera_start():
    """Endpoint: POST /api/camera/start"""
    global is_streaming

    if not CAMERA_AVAILABLE:
        return jsonify({
            'success': False,
            'message': 'Camera not available'
        }), 503

    if is_streaming:
        stop_camera()
        time.sleep(0.5)

    # Init camera
    if not init_camera():
        return jsonify({
            'success': False,
            'message': 'Failed to initialize camera. Check if camera is connected properly.'
        }), 500

    is_streaming = True

    return jsonify({
        'success': True,
        'message': 'Camera started',
        'stream_url': '/api/video_feed'
    })


@app.route('/api/camera/stop', methods=['POST'])
def camera_stop():
    """Endpoint: POST /api/camera/stop"""
    stop_camera()
    return jsonify({
        'success': True,
        'message': 'Camera stopped'
    })


@app.route('/api/video_feed', methods=['GET'])
def video_feed():
    """Endpoint: GET /api/video_feed - MJPEG stream"""
    global is_streaming

    # Auto-init camera jika belum
    if picam2 is None:
        if not init_camera():
            return jsonify({
                'success': False,
                'message': 'Failed to start camera'
            }), 500

    is_streaming = True

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/capture', methods=['POST'])
def capture():
    """Endpoint: POST /api/capture"""
    global picam2

    # Auto-init camera jika belum
    if picam2 is None:
        if not init_camera():
            return jsonify({
                'success': False,
                'message': 'Camera not available'
            }), 500

    try:
        # Capture frame
        with camera_lock:
            if picam2 is None:
                return jsonify({
                    'success': False,
                    'message': 'Camera not initialized'
                }), 500

            frame = picam2.capture_array()

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conjunctiva_{timestamp}.jpg"
        filepath = os.path.join(CAPTURE_DIR, filename)

        # Frame sudah BGR888, langsung simpan
        cv2.imwrite(filepath, frame)

        print(f" Captured: {filename}")

        return jsonify({
            'success': True,
            'message': 'Image captured',
            'data': {
                'filepath': filepath,
                'filename': filename,
                'image_url': f'/patient_images/{filename}'
            }
        })

    except Exception as e:
        print(f"Capture error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Endpoint: POST /api/analyze
    Body: {"image_path": "/path/to/image.jpg"}
    """
    print("\n" + "=" * 50)
    print("🔬 ANALYSIS STARTED")
    print("=" * 50)

    data = request.get_json() or {}
    image_path = data.get('image_path')

    result = {
        'status_anemia': 'unknown',
        'confidence': 0,
        'heart_rate': 0,
        'spo2': 0,
        'image_path': '',
        'timestamp': datetime.now().isoformat()
    }
    warnings = []

    #AI Classification
    print("\n[1/2] AI Classification...")

    if AI_AVAILABLE and image_path and os.path.exists(image_path):
        try:
            seg_m, class_m = load_ai_models()

            pipeline_result = main_pipeline(
                image_path, seg_m, class_m, config.DEVICE,
                save_results=True, output_dir=RESULTS_DIR
            )

            classification = pipeline_result.get('classification', {})
            status = classification.get('class_name', 'unknown')
            if isinstance(status, str):
                status = status.lower()

            conf = classification.get('confidence', 0)
            confidence = round(conf * 100, 2) if conf <= 1 else round(conf, 2)

            result['status_anemia'] = status
            result['confidence'] = confidence
            result['image_path'] = f"/patient_images/{os.path.basename(image_path)}"

            print(f"   Status: {status}, Confidence: {confidence}%")

        except Exception as e:
            warnings.append(f"AI error: {str(e)}")
            print(f"   AI error: {e}")
            import traceback
            traceback.print_exc()
    else:
        if not AI_AVAILABLE:
            warnings.append("AI not available")
        elif not image_path:
            warnings.append("No image path provided")
        elif not os.path.exists(image_path):
            warnings.append(f"Image not found: {image_path}")
        print("   AI skipped")

    #Sensor Reading
    print("\n[2/2] Sensor Reading...")

    if SENSOR_AVAILABLE:
        sensor_result = read_sensor(duration=10, warmup=3)
        result['heart_rate'] = sensor_result.get('heart_rate', 0)
        result['spo2'] = sensor_result.get('spo2', 0)

        if 'error' in sensor_result:
            warnings.append(f"Sensor: {sensor_result['error']}")
            print(f"   {sensor_result['error']}")
        else:
            print(f"    HR: {result['heart_rate']} bpm, SpO2: {result['spo2']}%")
    else:
        warnings.append("Sensor not available")
        print("   Sensor skipped")

    # LED indicator
    if LED_AVAILABLE:
        init_gpio()
        led_on()
        time.sleep(5)
        led_off()
        cleanup_gpio()

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50 + "\n")

    response = {
        'success': True,
        'message': 'Analysis complete',
        'data': result
    }

    if warnings:
        response['warnings'] = warnings

    return jsonify(response)


# STATIC FILES
@app.route('/patient_images/<filename>')
def serve_patient_image(filename):
    """Serve gambar dari folder patient_images"""
    return send_from_directory(CAPTURE_DIR, filename)


@app.route('/results/<path:filepath>')
def serve_result(filepath):
    """Serve file dari folder results"""
    return send_from_directory(RESULTS_DIR, filepath)


# CLEANUP
import atexit

def cleanup():
    """Cleanup saat aplikasi exit"""
    print("\nCleaning up...")
    stop_camera()
    print("Cleanup done")

atexit.register(cleanup)


# MAIN
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 ANEMIA DETECTION API")
    print("=" * 50)
    print(f"   AI Available     : {'✓' if AI_AVAILABLE else '✗'}")
    print(f"   Sensor Available : {'✓' if SENSOR_AVAILABLE else '✗'}")
    print(f"   Camera Available : {'✓' if CAMERA_AVAILABLE else '✗'}")
    print("=" * 50)
    print("   Endpoints:")
    print("   GET  /api/health")
    print("   POST /api/camera/start")
    print("   POST /api/camera/stop")
    print("   GET  /api/video_feed")
    print("   POST /api/capture")
    print("   POST /api/analyze")
    print("=" * 50)
    print("   http://0.0.0.0:5000")
    print("=" * 50 + "\n")

    # Pre-load AI models
    if AI_AVAILABLE:
        try:
            load_ai_models()
        except Exception as e:
            print(f" Could not pre-load models: {e}")

    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)