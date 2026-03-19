"""Simple Flask runner that keeps the server alive"""
import sys
import os

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from app import app, load_model
from werkzeug.serving import make_server
import threading

class ServerThread(threading.Thread):
    def __init__(self, app):
        threading.Thread.__init__(self)
        self.server = make_server('127.0.0.1', 5000, app, threaded=True)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

# Load model first
print("Loading model...")
load_model()
print("Model loaded!")

# Start server in background thread
server = ServerThread(app)
server.daemon = True
server.start()

print("Server running at http://localhost:5000")
print("Press Ctrl+C to stop")

# Keep main thread alive
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down...")
    server.shutdown()
