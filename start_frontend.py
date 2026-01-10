"""
Simple HTTP Server to serve the frontend.
Run this to avoid file:// protocol issues in browsers.
"""
import http.server
import socketserver
import os
import webbrowser

PORT = 3000
DIRECTORY = "frontend"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == "__main__":
    # Ensure we are in the project root
    if not os.path.exists(DIRECTORY):
        print(f"Error: {DIRECTORY} directory not found. Please run from project root.")
        exit(1)
        
    print(f"Starting Frontend Server at http://127.0.0.1:{PORT}")
    print("Press Ctrl+C to stop.")
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        # Open browser automatically
        webbrowser.open(f"http://127.0.0.1:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server...")
            httpd.server_close()
