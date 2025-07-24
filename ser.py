from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify

model = SentenceTransformer("./models/embodel/RoSBERTa")
app = Flask(__name__)

@app.route('/', methods=['POST'])  # Changed to root endpoint
def embed():
    data = request.json
    
    # Handle both TEI and standard formats
    if 'inputs' in data:
        texts = data['inputs']
    elif isinstance(data, list):
        texts = data
    else:
        return jsonify({"error": "Invalid input format"}), 400
    
    # Convert to list if single string
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = model.encode(texts, convert_to_tensor=True)
    return jsonify(embeddings.tolist())  # Convert tensor to list

if __name__ == '__main__':
    app.run(host='localhost', port=8080)
