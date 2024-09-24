##########
# For asking queries use this command : curl -X POST http://127.0.0.1:5000/query -H "Content-Type: application/json" -d '{"query": "machine learning"}'
# write query inside "query"
##########
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize Flask App
app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can use other models as well, such as "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Sample document corpus (a basic retrieval system)
document_corpus = {
    'doc1': 'Artificial intelligence is transforming industries with new capabilities in automation.',
    'doc2': 'Machine learning models are trained using vast amounts of data to make predictions.',
    'doc3': 'Deep learning is a subset of machine learning with neural networks to process data.',
    'doc4': 'Machine learning (ML) is a branch of artificial intelligence (AI) and computer science that focuses on using data and algorithms to enable AI to imitate the way that humans learn, gradually improving its accuracy.'
}

# Simple retrieval function (based on keyword matching)
def retrieve_documents(query):
    relevant_docs = []
    for doc_id, content in document_corpus.items():
        if query.lower() in content.lower():
            relevant_docs.append(content)
    return relevant_docs

# Function to generate a response using GPT-2
def generate_response(query, retrieved_docs):
    prompt = f"Answer the following query based on these documents: {retrieved_docs}\n\nQuery: {query}\nAnswer:"
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate output
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the RAG API! Use the /query endpoint to interact with the API."})
# API endpoint for querying
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Step 1: Retrieve relevant documents
    retrieved_docs = retrieve_documents(query)

    if not retrieved_docs:
        return jsonify({"error": "No relevant documents found"}), 404

    # Step 2: Generate a response using retrieved documents
    response = generate_response(query, retrieved_docs)

    return jsonify({"query": query, "response": response})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
