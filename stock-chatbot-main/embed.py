from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import uuid
from tqdm import tqdm

DATA_PATH = "/home/mt/Downloads/code/llm_rag/dataset.txt" 

PINECONE_API_KEY = "pcsk_7QDKFB_9snwC9WBmGi1gLdZYSfg9V7yzxhibDjbbhthZCq3A4JjC5cqBkeAm9RtpoARrVe"       
PINECONE_ENVIRONMENT = "us-east-1"             
INDEX_NAME = "stock-chatbot"                    

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    return docs


def embed_documents(docs):
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def upload_to_pinecone(documents, embeddings, index_name, api_key, region):
    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=len(embeddings[0]),
            metric="cosine",
            spec=ServerlessSpec(
                cloud="gcp", region=region.split("-")[0]
            )
        )

    index = pc.Index(index_name)

    vectors = []
    for doc, emb in tqdm(zip(documents, embeddings), total=len(documents)):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": emb.tolist(),
            "metadata": {"text": doc}
        })

    index.upsert(vectors)
    print(f"✅ Uploaded {len(vectors)} documents to index: {index_name}")


if __name__ == "__main__":
    documents = load_documents(DATA_PATH)
    embeddings = embed_documents(documents)
    upload_to_pinecone(
        documents,
        embeddings,
        INDEX_NAME,
        PINECONE_API_KEY,
        PINECONE_ENVIRONMENT
    )
