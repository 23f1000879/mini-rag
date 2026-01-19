"""
Frontend Flask Server for Mini RAG Application
Serves the HTML frontend interface
"""

from flask import Flask, render_template_string, send_from_directory
import os

app = Flask(__name__)

# Read the HTML file
HTML_FILE = 'index.html'

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        with open(HTML_FILE, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return """
        <h1>Error: index.html not found</h1>
        <p>Please make sure index.html is in the same directory as frontend_app.py</p>
        """, 404

if __name__ == '__main__':
    print("🎨 Starting Frontend Server")
    print("📱 Open your browser and go to: http://localhost:3000")
    print("⚙️  Make sure the backend is running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=3000)