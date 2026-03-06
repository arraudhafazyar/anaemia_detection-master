import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

from picamera2 import Picamera2, Preview
from time import sleep
from pathlib import Path
from datetime import datetime
import cv2

# Disable OpenCV logging 
cv2.setLogLevel(0)

# Manual override mode 
FORCE_VNC_MODE = True # True=VNC | False=Monitor | None=Auto-detect


def is_vnc_mode():
    # Manual override jika di-set
    if FORCE_VNC_MODE is not None:
        return FORCE_VNC_MODE
    
    # Cek environment variables yang di-set oleh VNC
    vnc_indicators = [
        os.environ.get('VNCDESKTOP'),
        'VNC' in os.environ.get('XDG_SESSION_TYPE', ''),
        os.environ.get('DISPLAY') == ':1',
    ]
    
    return any(vnc_indicators)


def capture_conjunctiva(save_dir="captures", show_preview=True, show_captured=True):
    """   
    Workflow:
    1. Live preview muncul di monitor
    2. User arahkan kamera ke mata pasien
    3. ENTER untuk capture
    4. Preview frozen + gambar ditampilkan dalam window OpenCV
    5. User konfirmasi: y (terima) / n (capture ulang)
    
    Args:
        save_dir (str): Folder penyimpanan
        show_preview (bool): Tampilkan live preview saat capturing
        show_captured (bool): Tampilkan hasil capture di OpenCV window untuk review
    
    Returns:
        str or None: Path file yang diterima, atau None jika dibatalkan
    """
    # folder check
    Path(save_dir).mkdir(exist_ok=True)
    
    # Deteksi mode
    vnc_mode = is_vnc_mode()
    mode_name = "VNC (OpenCV)" if vnc_mode else "MONITOR FISIK (QTGL)"
    
    # Setup kamera
    try:
        picam2 = Picamera2()
        
        # Configuration untuk still capture
        config = picam2.create_still_configuration(
            main={"size": (640, 480)},
            buffer_count=2
        )
        picam2.configure(config)
        
        qtgl_active = False
        
        if show_preview and not vnc_mode:
            # MODE MONITOR FISIK
            try:
                picam2.start_preview(Preview.QTGL)
                qtgl_active = True
                print(" Preview: QTGL")
            except Exception as e:
                print(f" QTGL failed: {e}")
                vnc_mode = True
        
        if show_preview and vnc_mode:
            print(" Preview: OpenCV")
        
        # ENABLE AUTOFOCUS 
        picam2.start()
        
        print(" Activating autofocus...")
        picam2.set_controls({
            "AfMode": 2,      # 0=Manual, 1=Auto, 2=Continuous
            "AfTrigger": 0    # Start autofocus
        })
        
        sleep(2)  # Kasih waktu autofocus bekerja
        # AKHIR AUTOFOCUS
        
        print("\n Camera ready with AUTOFOCUS")
        print("\n INSTRUKSI:")
        print("   Arahkan kamera ke KONJUNGTIVA mata pasien")
        print("   (Bagian DALAM kelopak mata BAWAH yang berwarna merah/pink)")
        
        if show_preview:
            if vnc_mode:
                print("\n  [VNC] SPACE=capture | Q=quit\n")
            else:
                print("\n️  [MONITOR] ENTER=capture\n")
        
    except Exception as e:
        print(f" Error: {e}")
        return None
    
    # Main capture loop
    attempt = 1
    temp_dir = os.path.join(save_dir, ".temp")
    Path(temp_dir).mkdir(exist_ok=True)
    
    try:
        while True:
            print(f"\n{'-'*60}")
            print(f" Attempt #{attempt}")
            print(f"{'-'*60}")
            
            # CONDITIONAL PREVIEW
            if show_preview and vnc_mode:
                # VNC MODE - OpenCV Preview
                print("\n Live Preview (OpenCV)...")
                print("Arahkan kamera, tunggu fokus, lalu tekan SPACE")
                
                preview_window = f"Live Preview #{attempt} - SPACE=capture, Q=quit"
                
                while True:
                    frame = picam2.capture_array()
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    display_frame = cv2.resize(frame_bgr, (800, 600))
                    
                    # Get autofocus metadata
                    try:
                        metadata = picam2.capture_metadata()
                        lens_pos = metadata.get("LensPosition", "N/A")
                        af_state = metadata.get("AfState", "N/A")
                    except:
                        lens_pos = "N/A"
                        af_state = "N/A"
                    
                    # Text overlay
                    cv2.putText(display_frame, f"Attempt #{attempt} | AUTOFOCUS: ON", 
                               (10, 40), cv2.FONT_HERSHEY_DUPLEX, 
                               0.8, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Lens: {lens_pos} | AF: {af_state}", 
                               (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 2)
                    cv2.putText(display_frame, "SPACE=Capture | Q=Quit", 
                               (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 0), 2)
                    
                    cv2.imshow(preview_window, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord(' '):
                        cv2.destroyWindow(preview_window)
                        print(" Preview frozen!")
                        break
                    elif key == ord('q') or key == ord('Q'):
                        print("\n Quit")
                        cv2.destroyAllWindows()
                        picam2.stop()
                        if qtgl_active:
                            try:
                                picam2.stop_preview()
                            except:
                                pass
                        return None
            else:
                # MONITOR MODE - Input biasa
                input("Tekan ENTER untuk capture... ")
            
            # Countdown
            print("\n Bersiap...", end=' ', flush=True)
            for t in [3, 2, 1]:
                print(f"{t}...", end=' ', flush=True)
                sleep(1)
            print("SNAP!\n")
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"temp_{timestamp}.jpg"
            temp_filepath = os.path.join(temp_dir, temp_filename)
            
            # Capture image
            try:
                picam2.capture_file(temp_filepath)
                print(f"Gambar dicapture")
            except Exception as e:
                print(f" Gagal: {e}")
                retry = input("\n   Coba lagi? (y/n): ").lower().strip()
                if retry != 'y':
                    break
                attempt += 1
                continue
            
            # REVIEW IMAGE
            if show_captured:
                try:
                    img = cv2.imread(temp_filepath)
                    if img is not None:
                        display_img = cv2.resize(img, (800, 600))
                        cv2.putText(display_img, "Review Kualitas", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (0, 255, 0), 2)
                        cv2.putText(display_img, "Press ANY KEY to continue...", 
                                   (10, 570), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (255, 255, 255), 1)
                        
                        review_window = "Captured Image - Review"
                        cv2.imshow(review_window, display_img)
                        
                        # Tunggu keypress dengan loop + timeout
                        print("\n Review window terbuka")
                        print("   Tekan ANY KEY untuk lanjut...")
                        
                        key_pressed = False
                        timeout_counter = 0
                        max_timeout = 300  # 30 detik
                        
                        while not key_pressed and timeout_counter < max_timeout:
                            key = cv2.waitKey(100) & 0xFF
                            if key != 255:
                                key_pressed = True
                                print("  Key detected!")
                            timeout_counter += 1
                            cv2.imshow(review_window, display_img)
                        
                        if not key_pressed:
                            print(" Timeout 30s, auto-close")
                        
                        cv2.destroyAllWindows()
                        print("   Review done")
                
                except Exception as e:
                    print(f"  Cannot display: {e}")
                    print(f"   File: {temp_filepath}")
            
            decision = input("\n Bagus? (y/n): ").lower().strip()
            
            if decision == 'y':
                final_filename = f"conjunctiva_{timestamp}.jpg"
                final_filepath = os.path.join(save_dir, final_filename)
                os.rename(temp_filepath, final_filepath)
                
                cv2.destroyAllWindows()
                print("\n  DITERIMA!")
                print(f"   {final_filepath}")
                print(f"   Size: {os.path.getsize(final_filepath) / 1024:.1f} KB")
                return final_filepath
            
            elif decision == 'n':
                print(" Ditolak")
                cv2.destroyAllWindows()
                try:
                    os.remove(temp_filepath)
                except:
                    pass
                
                retry = input("\n Capture ulang? (y/n): ").lower().strip()
                if retry == 'y':
                    attempt += 1
                    continue
                else:
                    return None
            
            else:
                print("\nCancel")
                cv2.destroyAllWindows()
                try:
                    os.remove(temp_filepath)
                except:
                    pass
                return None
    
    except KeyboardInterrupt:
        print("\n\n Ctrl+C")
        cv2.destroyAllWindows()
        return None
    
    finally:
        picam2.stop()
        if qtgl_active:
            try:
                picam2.stop_preview()
            except:
                pass
        cv2.destroyAllWindows()
        
        try:
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)
        except:
            pass
        
        print(" Camera off\n")
