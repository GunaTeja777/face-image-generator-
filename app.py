from flask import Flask, render_template, jsonify
from generator import generate_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET'])
def generate():
    img_data = generate_image()
    return jsonify({'image': img_data})

if __name__ == '__main__':
    app.run(debug=True)
