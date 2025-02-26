import faiss
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM

import yaml
import os

with open('params.yaml') as stream:
    config = yaml.safe_load(stream)

# Load the GGUF model e.g https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF
model = r"mistral-7b-v0.1.Q4_K_M.gguf"
model_path = os.path.join(config['config']['models_path'], model)
pdf_directory = config['config']['pdfs_path']
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-v0.1-GGUF", model_file=model_path, model_type="mistral", gpu_layers=0)

# Example corpus of text documents
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Tokyo is the capital of Japan.",
    "AI is revolutionizing the tech industry."
]

# Use SentenceTransformer to encode the text documents into embeddings
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
embeddings = encoder.encode(documents)

# Create a FAISS index and add the document embeddings
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Function to perform similarity search using FAISS
def search_similar_documents(query, k=3):
    query_embedding = encoder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [(documents[i], distances[0][i]) for i in indices[0]]

# Ask a question and find similar documents
question = "What is the capital of France?"
similar_documents = search_similar_documents(question)

print(f"Question: {question}")
print("Similar Documents:")
for doc, dist in similar_documents:
    print(f"- {doc} (Distance: {dist})")
