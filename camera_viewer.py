#!/usr/bin/env python3
"""
Simple HTTP Camera Viewer
Shows camera frames over HTTP while trackingeyes is running
Access at: http://localhost:8001
"""

import cv2
from picamera2 import Picamera2
import threading
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket
import time


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Picamera Feed</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: #1e1e1e;
                        color: white;
                    }
                    .container { max-width: 900px; margin: 0 auto; }
                    h1 { color: #4CAF50; }
                    img {
                        max-width: 100%;
                        border: 2px solid #4CAF50;
                        border-radius: 5px;
                        margin-top: 20px;
                    }
                    .info {
                        background-color: #333;
                        padding: 15px;
                        border-radius: 5px;
                        margin-top: 20px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎥 Picamera Live Feed</h1>
                    <img src="/stream.mjpg" width="800" alt="Camera Stream">
                    <div class="info">
                        <h3>Info</h3>
                        <p><strong>Resolution:</strong> 1280x720</p>
                        <p><strong>Format:</strong> MJPEG Stream</p>
                        <p><strong>Camera:</strong> Raspberry Pi IMX219</p>
                    </div>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            
            try:
                frame_count = 0
                while True:
                    with frame_lock:
                        if current_frame is not None:
                            _, jpeg = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            self.wfile.write(b'--FRAME\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n')
                            self.wfile.write(b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n')
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b'\r\n')
                            frame_count += 1
                    time.sleep(0.01)
            except:
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


current_frame = None
frame_lock = threading.Lock()


def capture_frames():
    global current_frame
    
    print("Initializing camera...")
    picam2 = Picamera2()
    
    config = picam2.create_video_configuration(
        main={"format": 'XRGB8888', "size": (1280, 720)}
    )
    picam2.configure(config)
    picam2.start()
    
    print("✅ Camera started")
    time.sleep(1)
    
    try:
        while True:
            frame = picam2.capture_array()
            bgr_frame = frame[:, :, :3]
            
            # Rotate frame 180 degrees
            bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_180)
            
            with frame_lock:
                current_frame = bgr_frame
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        picam2.close()


if __name__ == '__main__':
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"
    
    PORT = 8001
    
    # Start capture thread
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    
    time.sleep(2)  # Let camera warm up
    
    print("\n" + "="*60)
    print("🎥 CAMERA VIEWER SERVER")
    print("="*60)
    print(f"\n✅ Server running on:")
    print(f"   Local:  http://localhost:{PORT}")
    print(f"   Remote: http://{local_ip}:{PORT}")
    print(f"\n📱 From SSH, use:")
    print(f"   ssh -L 8001:localhost:8001 nema@<your-ip>")
    print(f"   Then open: http://localhost:8001")
    print("\n⏹️  Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    server = HTTPServer(('0.0.0.0', PORT), StreamingHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
