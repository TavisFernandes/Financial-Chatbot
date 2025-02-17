from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from datetime import datetime
import os

# Store messages in memory
messages = [
    {
        "text": "Hello! I'm your AI investment advisor. How can I help you today?",
        "is_bot": True,
        "timestamp": datetime.now().isoformat()
    }
]

class InvestmentAdvisorHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/messages':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(messages).encode())
            return
        
        # Serve static files
        if self.path == '/':
            self.path = '/templates/index.html'
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path == '/api/messages':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            # Add user message
            user_message = {
                "text": data['text'],
                "is_bot": False,
                "timestamp": datetime.now().isoformat()
            }
            messages.append(user_message)
            
            # Add bot response
            bot_response = {
                "text": "I'm analyzing market conditions to provide you with personalized investment recommendations. Please note that this is a demo interface.",
                "is_bot": True,
                "timestamp": datetime.now().isoformat()
            }
            messages.append(bot_response)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode())
            return

def run(server_class=HTTPServer, handler_class=InvestmentAdvisorHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Server running on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run()