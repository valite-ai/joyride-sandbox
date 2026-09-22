"""A local HTTP server on 127.0.0.1 that replays scripted responses."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time


class LocalServer:
    """Answer each request with the next (status, headers, body) or callable."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                owner.requests.append((self.path, dict(self.headers), body))
                response = owner.responses.pop(0)
                if callable(response):
                    return response(self)
                status, headers, text = response
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(text)))
                self.end_headers()
                self.wfile.write(text)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def drip(handler):
    """Send a response one byte every 0.1 seconds for 3 seconds."""
    for byte in b"HTTP/1.1 200 OK\r\nX-Slow: " + b"x" * 5:
        handler.wfile.write(bytes([byte]))
        handler.wfile.flush()
        time.sleep(0.1)


def garbage(handler):
    handler.wfile.write(b"NOT HTTP AT ALL\r\n\r\n")
