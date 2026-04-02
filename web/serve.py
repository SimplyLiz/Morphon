#!/usr/bin/env python3
"""Dev server with no-cache headers for WASM development."""
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
port = 8080
print(f"Serving on http://localhost:{port} (no-cache)")
HTTPServer(('', port), NoCacheHandler).serve_forever()
