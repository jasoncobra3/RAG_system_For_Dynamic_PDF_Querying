# 📄 RAG System for Dynamic PDF Querying

An AI-powered Retrieval-Augmented Generation (RAG) system that allows you to query PDF documents dynamically. It extracts, embeds, and retrieves context from PDFs using **ChromaDB** and **HuggingFaceEmbeddings**, and generates intelligent answers with **ChatGroq LLMs**.  

The system is simple to run, lightweight, and demonstrates how LLMs can be combined with vector databases for accurate context-aware Q&A over documents.

---

## 🚀 Features

- 📄 Upload and parse PDF documents  
- 🔍 Chunk and embed PDF content using **HuggingFaceEmbeddings**  
- 🤖 Perform semantic retrieval on relevant chunks using **ChromaDB**
- 💬 Answer queries using **Groq LLMs**  
- 🗂️ Sample `temp.pdf` included for quick testing  
- ⚡ Minimal setup with `requirements.txt`

---

## 🧰 Tech Stack

- Python  
- ChatGroq API
- HuggingFaceEmbeddings
- ChromaDB (Vector Database)  
- LangChain  
- PyPDF or similar PDF parser  

---

## 📦 Installation

1. **Clone the Repo**
   ```bash
   git clone https://github.com/jasoncobra3/RAG_system_For_Dynamic_PDF_Querying.git
   cd RAG_system_For_Dynamic_PDF_Querying

2. **Create Virtual Environment**
   ```bash
    python -m venv venv
   
3. **Activate the Virtual Environment**
   ```bash
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    venv/bin/activate

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

---
## 🔐 Setup
1. **Create a `.env` file in root folder with**
   ```env
    GROQ_API_KEY=your_groq_api_key_here
   ```

---

##  🚀Run the App
   **Run the Script in Terminal**
   ```bash
     python app.py
   ```

---

## 📁 Project Structure
```
├── app.py
├── requirement.txt
├── temp.pdf
├── chroma_db/
├── .env
└── README.md

```

---

##  🤝 Contributing

Pull requests are welcome!
For changes, please open an issue first to discuss what you’d like to change.
