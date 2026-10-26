# 🎓 AI Placement Preparation Assistant

An AI-powered placement preparation chatbot built with **Streamlit, LangChain, Google Gemini, FAISS, and Retrieval-Augmented Generation (RAG)**. Upload placement preparation PDFs and ask questions grounded in their content.

## ✨ Features

- Upload and process PDF study materials.
- Extract text and split documents into overlapping chunks.
- Generate embeddings using Google Gemini.
- Retrieve relevant chunks using FAISS vector similarity search.
- Generate context-grounded answers using Gemini Flash-Lite.
- Display source filenames and page numbers.
- Maintain chat history during the session.
- Return a fallback response when the retrieved material does not answer the question.
- Display retrieval and answer-generation progress.

## 🛠️ Technology Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- Gemini Embeddings
- FAISS
- PyPDF

## ⚙️ Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Mainuddin123/AI-Placement-Preparation-Assisstant.git
   cd AI-Placement-Preparation-Assisstant
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your own Gemini API key:

   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   ```

   Obtain a key from [Google AI Studio](https://aistudio.google.com/apikey). Never commit your real API key.

5. Run the application:

   ```bash
   streamlit run app.py
   ```

6. Upload a PDF, click **Process Documents**, and ask a question.

## 🔎 How It Works

1. **Document loading:** Extracts text and metadata from uploaded PDFs.
2. **Chunking:** Splits document text into manageable pieces.
3. **Embedding:** Converts chunks into vector representations.
4. **Indexing:** Stores vectors in a FAISS index.
5. **Retrieval:** Finds the top three relevant chunks for a question.
6. **Generation:** Sends the retrieved context and question to Gemini.
7. **Citations:** Displays source filenames and page numbers.

## 🔐 Security

- Store API keys in `.env` locally or the deployment platform's secrets manager.
- Keep `.env` excluded from Git using `.gitignore`.
- Do not upload private documents or credentials to a public repository.

## 👨‍💻 Author

**Khaja Mainuddin**

## 📄 License

No license has been specified yet.
