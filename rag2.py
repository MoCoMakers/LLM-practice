import glob
import faiss
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

import yaml
import os

# Load configuration from YAML file
with open('params.yaml') as stream:
    config = yaml.safe_load(stream)

# Load the GGUF model, e.g., https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF
model_path = os.path.join(config['config']['models_path'], "mistral-7b-v0.1.Q4_K_M.gguf")
pdf_directory = config['config']['pdfs_path']
llm = Llama(
    model_path=model_path,
    # n_gpu_layers=-1, # Uncomment to use GPU acceleration
    # seed=1337, # Uncomment to set a specific seed
    n_ctx=10000, # Uncomment to increase the context window, default 2048
    )

# Initialize the SentenceTransformer for embeddings
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")

# Function to split text into smaller chunks
def split_paragraphs(rawText):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(rawText)

# Function to load PDFs and extract text
def load_pdfs(pdfs):
    text_chunks = []
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            raw = page.extract_text()
            chunks = split_paragraphs(raw)
            text_chunks += chunks
    return text_chunks

# Use glob to find all PDFs in the specified directory
pdfs = glob.glob(f"{pdf_directory}/*.pdf")

# Load the PDF documents
documents = load_pdfs(pdfs)

# Create embeddings for the documents
corpus_embeddings = encoder.encode(documents, show_progress_bar=True, convert_to_tensor=True)

# Create a FAISS index and add the document embeddings
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings.cpu().numpy())

# Function to perform similarity search using FAISS
def search_similar_documents(query, k=3):
    query_embedding = encoder.encode([query], show_progress_bar=True, convert_to_tensor=True)
    query_embedding = query_embedding.cpu().numpy()
    distances, indices = index.search(query_embedding, k)

    # Ensure we don't access out of bounds indices
    num_results = min(len(indices[0]), k)
    result_documents = [(documents[i], distances[0][idx]) for idx, i in enumerate(indices[0][:num_results])]
    
    return result_documents, query_embedding

# Function to generate LLM answer based on the retrieved documents
def generate_llm_answer(query, similar_documents, query_embedding):
    context = " ".join([doc for doc, _ in similar_documents])
    embeddings_str = " ".join(map(str, query_embedding.flatten()))
    input_text = f"Question: {query}\nEmbeddings: {embeddings_str}\nContext: {context}\nAnswer:"
    llm_output = llm(input_text, max_tokens=100)
    llm_answer = llm_output['choices'][0]['text'].strip()
    return llm_answer

# Ask a question, find similar documents, and generate an LLM answer
question = "What is delta S?"
similar_documents, query_embedding = search_similar_documents(question)
llm_answer = generate_llm_answer(question, similar_documents, query_embedding)

print(f"Question: {question}")
print("Similar Documents:")
for doc, dist in similar_documents:
    print(f"- {doc} (Distance: {dist})")

print("\nLLM Answer:")
print(llm_answer)
