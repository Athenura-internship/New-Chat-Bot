from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from chatbot import chatbot_response
import os
from whitenoise import WhiteNoise

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')
# Use WhiteNoise to serve static files efficiently
app.wsgi_app = WhiteNoise(app.wsgi_app, root='.', prefix='')
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message")

        if not message:
            return jsonify({"response": "Please enter a message."})

        response = chatbot_response(message)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": "Oops! Something went wrong. Please try again or contact support@athenura.in."})

@app.route("/health")
def health():
    return {"status": "ok"}, 200

if __name__ == "__main__":
    # Use PORT environment variable if available, otherwise default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)