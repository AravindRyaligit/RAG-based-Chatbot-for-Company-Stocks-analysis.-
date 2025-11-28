# Stock Chatbot 📈

A powerful RAG (Retrieval-Augmented Generation) chatbot designed to answer questions about Indian stock market data (Nifty 500). Built with **FastAPI**, **Pinecone**, and **Google Gemini**.

## 🚀 Features

-   **Intelligent Retrieval**: Uses Pinecone vector database to find relevant stock information.
-   **Generative AI**: Powered by Google's Gemini 2.0 Flash model for accurate and natural responses.
-   **FastAPI Backend**: High-performance, asynchronous API.
-   **Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for efficient text embedding.
-   **Secure**: Environment variable configuration for API keys.

## 🛠️ Tech Stack

-   **Backend**: Python, FastAPI
-   **Vector DB**: Pinecone
-   **LLM**: Google Gemini (via `google-generativeai`)
-   **Embeddings**: HuggingFace Transformers (`sentence-transformers`)
-   **Frontend**: HTML/JS (Simple Interface)

## 📋 Prerequisites

-   Python 3.8+
-   A [Google Gemini API Key](https://aistudio.google.com/)
-   A [Pinecone API Key](https://www.pinecone.io/)

## ⚙️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/stock-chatbot.git
    cd stock-chatbot
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    GOOGLE_API_KEY=your_google_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    ```

## 🏃‍♂️ Usage

### Running the Backend
Start the FastAPI server:
```bash
python -m uvicorn fast_api:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### Using the Chatbot
Open `index.html` in your web browser to interact with the chatbot.

### Running the Standalone Script
You can also test the logic directly via the terminal:
```bash
python main.py
```

## 📂 Project Structure

-   `fast_api.py`: Main backend application.
-   `main.py`: Standalone script for testing logic.
-   `requirements.txt`: Python dependencies.
-   `index.html`: Simple frontend interface.
-   `.env`: Configuration file (not committed).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Cc License

This project is licensed under the MIT License.

![Screenshot 2025-11-28 153433](https://github.com/user-attachments/assets/188f4f9b-0625-4939-8fca-cf53d5509d3e)
