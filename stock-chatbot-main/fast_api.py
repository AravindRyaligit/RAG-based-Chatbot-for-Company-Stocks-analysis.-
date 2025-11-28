from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "stock-chatbot"
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )

index = pc.Index(INDEX_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

class Query(BaseModel):
    question: str

def embed_text(texts):
    encoded_input = embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = embed_model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def retrieve_docs(query, top_k=5):
    query_embedding = embed_text([query])[0].tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

def construct_prompt(question, docs):
    context = "\n".join(docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return prompt

def generate_answer_api(prompt):
    system_prompt = (
        "You are a knowledgeable and helpful stock market assistant. "
        "Use only the provided context to answer the question. If the context does not contain "
        "sufficient information, say 'I don't know based on the provided context.' "
        "Be concise and clear in your responses."
    )
    full_prompt = f"{system_prompt}\n\n{prompt}"
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(full_prompt)
    return response.text.strip()

@app.post("/chat/")
async def chat(query: Query):
    retrieved_docs = retrieve_docs(query.question)
    prompt = construct_prompt(query.question, retrieved_docs)
    try:
        answer = generate_answer_api(prompt)
    except Exception as e:
        return {"error": str(e)}
    return {"answer": answer}
