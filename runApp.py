#!/usr/bin/env python
from http.server import HTTPServer, BaseHTTPRequestHandler
import mlflow
import os
import logging

@mlflow.trace(span_type="func", attributes={"name": "add_2"})
def add_1(x):
    return x + 2

@mlflow.trace(span_type="func")
def mybot_ask(question):
    add_1(123)
    return "Paris"

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            mybot_ask("What is the capital of France?")
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>Hello, World!</h1>')
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>404 Not Found</h1>')

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    # Log all environment variables
    logger.info("Environment Variables:")
    for key, value in os.environ.items():
        logger.info(f"{key}={value}")
    
    mlflow.set_experiment("Server app trace 2")
    server = HTTPServer(('0.0.0.0', 8888), SimpleHandler)
    print("Server running on http://localhost:8888")
    with mlflow.start_run() as run:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.shutdown()
