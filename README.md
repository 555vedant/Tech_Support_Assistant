# 🛠️ TaskFlow Technical Support RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) based chatbot built using:
- 🧠 LangChain for chaining LLM + retrieval
- 🔍 HuggingFace Transformers for embeddings and LLM (FLAN-T5)
- 🗃️ Pinecone for vector database search
- 🌐 Streamlit for an interactive frontend

---

## 💻 Features
- Ask technical questions about TaskFlow (a fictional cloud-based project management tool)
- Uses a CSV of past support issues and responses
- Retrieves relevant solutions using vector similarity search
- Responds concisely using a language model

---

## 📂 Folder Structure

```
.
├── RAG_ChatBot.py           # Backend logic (RAG setup)
├── app.py                   # Streamlit frontend
├── data/
│   └── tech_support_vedant.csv  # Your dataset
├── .env                     # API keys (not committed)
├── .gitignore
├── sample_io.txt            # Example input/output
└── README.md
```

---

## 🛠 Setup Instructions

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Set up your `.env`**:

```env
PINECONE_API_KEY=your_pinecone_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

3. **Run the chatbot**:

```bash
streamlit run app.py --server.fileWatcherType none
```

---

## 📘 Notes
- Make sure your `.env` and `tech_support_vedant.csv` are present.
- Do **not** commit `.env` — it contains sensitive keys.
- Dataset should include these columns:
  - `Customer_Issue`
  - `Tech_Response`
  - `Issue_Category`

---

## 👤 Author
**Vedant Kasar**

---

## 🔐 License
This project is for educational/demo purposes only. Keys should be kept secret and rotated if leaked.
