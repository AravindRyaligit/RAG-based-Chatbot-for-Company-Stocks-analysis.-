from transformers import AutoTokenizer, AutoModel
import torch
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "stock-chatbot"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def embed_text(texts):
    encoded_input = embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        model_output = embed_model(**encoded_input)
        
    token_embeddings = model_output.last_hidden_state
    embeddings = token_embeddings.mean(dim=1)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()

def retrieve_docs(query, top_k=5):
    query_embedding = embed_text([query])[0].tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

def generate_answer_gemini(prompt, max_tokens=150):
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

def construct_prompt(question, docs):
    context = "\n".join(docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return prompt

if __name__ == "__main__":
    user_question = "what industry is Astral Ltd?"
    print("Query:", user_question)    
    retrieved_docs = retrieve_docs(user_question)    
    prompt = construct_prompt(user_question, retrieved_docs)    
    response = generate_answer_gemini(prompt)
    print("Chatbot:", response)