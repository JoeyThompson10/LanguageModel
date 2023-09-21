from flask import Flask, request, jsonify, render_template
# Import your script here
from test import NgramModel, Preprocessor

app = Flask(__name__)

n = 3
model = NgramModel(n)
# Sample training data. Replace with your own
sample_data = "This is a sample text for training the ngram model. This text can be replaced with any other text for better results."
model.train(Preprocessor.preprocess_text(sample_data))

@app.route('/')
def index():
    return open('index.html').read()

@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.get_json(force=True)  # get JSON data from request
    seed = data.get('seed', '')
    num_words = data.get('num_words', 50)
    generated_text = model.generate_text(seed, int(num_words))
    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(debug=True)
