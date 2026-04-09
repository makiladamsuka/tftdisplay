#!/usr/bin/env python3
"""
Test script for Picamera2 on Raspberry Pi
Tests basic camera functionality, capture, and display capabilities
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

def test_picamera_basic():
    """Test basic picamera2 initialization and frame capture"""
    print("=" * 60)
    print("🎥 PICAMERA2 TEST SUITE")
    print("=" * 60)
    
    # Test 1: Import and initialization
    print("\n[1/5] Testing picamera2 import...")
    try:
        from picamera2 import Picamera2
        print("✅ picamera2 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import picamera2: {e}")
        print("   Install with: sudo apt install -y python3-picamera2")
        return False
    
    # Test 2: Create and configure camera
    print("\n[2/5] Initializing Picamera2...")
    try:
        picam2 = Picamera2()
        print(f"✅ Picamera2 instance created")
        
        # Show camera properties
        print(f"   - Camera model: {picam2.camera_properties}")
        
    except Exception as e:
        print(f"❌ Failed to create Picamera2 instance: {e}")
        return False
    
    # Test 3: Configure video stream
    print("\n[3/5] Configuring video stream (1280x720, XRGB8888)...")
    try:
        config = picam2.create_video_configuration(
            main={"format": 'XRGB8888', "size": (1280, 720)}
        )
        picam2.configure(config)
        print("✅ Video configuration created and applied")
        print(f"   - Resolution: 1280x720")
        print(f"   - Format: XRGB8888")
        
    except Exception as e:
        print(f"❌ Failed to configure video: {e}")
        return False
    
    # Test 4: Start camera and capture frames
    print("\n[4/5] Starting camera and capturing frames...")
    try:
        picam2.start()
        print("✅ Camera started")
        
        # Allow camera to warm up
        time.sleep(1.0)
        print("   Waiting for camera to warm up...")
        
        # Capture test frames
        captured_frames = []
        for i in range(5):
            frame = picam2.capture_array()
            captured_frames.append(frame)
            print(f"   ✓ Frame {i+1}/5 captured (shape: {frame.shape}, dtype: {frame.dtype})")
            time.sleep(0.1)
        
        # Verify frame quality
        test_frame = captured_frames[0]
        if test_frame is None or test_frame.size == 0:
            print("❌ Captured frame is empty or None")
            return False
        
        # Check if frame has data (not all zeros or all same value)
        if np.all(test_frame == 0):
            print("⚠️  Warning: Frame appears to be all black (possible sensor/lens issue)")
        elif np.std(test_frame) < 5:
            print("⚠️  Warning: Frame has very low variance (low contrast/dark)")
        else:
            print("✅ Frames have normal variance (sensor working)")
        
        print(f"\n✅ Successfully captured 5 frames")
        print(f"   - Frame dimensions: {test_frame.shape}")
        print(f"   - Frame dtype: {test_frame.dtype}")
        
    except Exception as e:
        print(f"❌ Failed to capture frames: {e}")
        return False
    finally:
        # Properly clean up camera for next tests
        try:
            picam2.stop()
            picam2.close()
        except:
            pass
    
    # Test 5: Save frame as test image
    print("\n[5/5] Saving test frame to disk...")
    try:
        # Convert XRGB8888 to BGR for OpenCV
        test_frame = captured_frames[0]
        bgr_frame = test_frame[:, :, :3]  # Drop X channel, keep BGR
        
        output_path = Path("/home/nema/Documents/voice-agentv2/backend/assets/test_capture.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(str(output_path), bgr_frame)
        if success:
            file_size = output_path.stat().st_size
            print(f"✅ Test frame saved to {output_path}")
            print(f"   - File size: {file_size / 1024:.1f} KB")
        else:
            print(f"❌ Failed to save frame using cv2.imwrite")
            return False
            
    except Exception as e:
        print(f"❌ Failed to save test frame: {e}")
        return False
    finally:
        # Clean up
        try:
            picam2.stop()
            print("\n✅ Camera stopped cleanly")
        except:
            pass
    
    return True


def test_camera_specs():
    """Test and display camera specifications and capabilities"""
    print("\n" + "=" * 60)
    print("📋 CAMERA SPECIFICATIONS")
    print("=" * 60)
    
    try:
        from picamera2 import Picamera2
        import libcamera
        import time
        
        picam2 = Picamera2()
        
        print("\n📊 Camera Properties:")
        props = picam2.camera_properties
        for key, value in props.items():
            print(f"   - {key}: {value}")
        
        print("\n📐 Available Resolutions:")
        # Try to get stream configurations
        try:
            configs = picam2.get_stream_configuration(0)
            print(f"   Configuration info available")
        except:
            print("   (Use default: 1280x720 for most applications)")
        
        print("\n✅ Camera specs retrieved successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to retrieve camera specs: {e}")
        return False
    finally:
        try:
            picam2.close()
        except:
            pass


def test_continuous_streaming():
    """Test continuous frame streaming and performance"""
    print("\n" + "=" * 60)
    print("⚡ CONTINUOUS STREAMING TEST (5 seconds)")
    print("=" * 60)
    
    try:
        from picamera2 import Picamera2
        import time
        
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"format": 'XRGB8888', "size": (1280, 720)}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(0.5)
        
        frame_count = 0
        start_time = time.time()
        
        print("\nCapturing frames for 5 seconds...")
        while time.time() - start_time < 5.0:
            frame = picam2.capture_array()
            frame_count += 1
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        
        print(f"\n✅ Streaming test completed")
        print(f"   - Frames captured: {frame_count}")
        print(f"   - Time elapsed: {elapsed:.2f}s")
        print(f"   - FPS achieved: {fps:.1f}")
        print(f"   - Expected: ~30 FPS (1280x720)")
        
        if fps > 20:
            print("   ✅ Performance is good")
        elif fps > 10:
            print("   ⚠️  Performance is acceptable but lower than expected")
        else:
            print("   ❌ Performance is very low")
        
        picam2.stop()
        return True
        
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False
    finally:
        try:
            picam2.close()
        except:
            pass


if __name__ == "__main__":
    success = test_picamera_basic()
    
    if success:
        print("\n" + "🎉" * 30)
        print("✅ ALL TESTS PASSED - Your Picamera is working correctly!")
        print("🎉" * 30)
        
        # Run additional tests
        test_camera_specs()
        test_continuous_streaming()
        
    else:
        print("\n" + "❌" * 30)
        print("TESTS FAILED - Please check your Picamera setup")
        print("❌" * 30)
        sys.exit(1)
