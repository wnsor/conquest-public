"""Lightweight HTTP server with no-cache headers — for webapp dev.

Usage:
    python scripts/serve_webapp_nocache.py [port]   # default 8001
"""
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        SimpleHTTPRequestHandler.end_headers(self)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    print(f"Serving on http://localhost:{port}/ (no-cache headers)")
    HTTPServer(("", port), NoCacheHandler).serve_forever()


if __name__ == "__main__":
    main()
