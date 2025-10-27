# 🧠  RAG Chatbot (PDF + Web + Voice) # 

A fully local, voice-enabled, retrieval-augmented chatbot built using **LangChain**, **LangGraph**, **Streamlit**, and **Ollama**. It can answer questions from both PDF documents and live web search using **Tavily**, with input via microphone or keyboard.

---


![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-2ca5a5?logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.0.x-ffb347?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-ff4b4b?logo=streamlit&logoColor=white)
![LangChain Community](https://img.shields.io/badge/langchain--community-Active-blueviolet)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM%20Server-272727?logo=llama&logoColor=white)
![LangChain Nomic](https://img.shields.io/badge/langchain--nomic-Integrated-8a2be2)
![Nomic Local](https://img.shields.io/badge/nomic%5Blocal%5D-Vector%20Embedding-593196)
![PyPDF2](https://img.shields.io/badge/PyPDF2-3.0%2B-3776ab?logo=python&logoColor=white)
![FPDF](https://img.shields.io/badge/FPDF-PDF%20Generator-007acc?logo=python&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e?logo=scikit-learn&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-Cache%20and%20Parallelism-blue?logo=python&logoColor=white)
![Typing Extensions](https://img.shields.io/badge/typing--extensions-Backport-green?logo=python&logoColor=white)
![PyPDF](https://img.shields.io/badge/pypdf-3.0%2B-008080?logo=python&logoColor=white)
![Python Dotenv](https://img.shields.io/badge/python--dotenv-Env%20Management-lightgrey?logo=python&logoColor=white)

---

## 🎥 Video Demo



https://github.com/user-attachments/assets/0faecd12-116d-47b6-bd3c-ad453746b4a7



---

## 🧠 How It Works/flow chart

![Untitled-2024-11-15-1903](https://github.com/user-attachments/assets/9aade3aa-8887-4908-828f-575de3f00fc0)


---
---

## 📁 Folder Structure

```
rag_chatbot_with_speach/
├── rag/
│   ├── ollama_llm.py
│   ├── graph_workflow.py
│   ├── prompts.py
│   ├── vectorstore.py
│   ├── embeddings.py
│   ├── loaders.py
│   ├── tavily_search.py
│   ├── router.py
│   └── utils.py
├── voice.py
├── run_once.py
├── frontend/
│   └── app.py
├── data/
│   ├── documents/        # Place PDFs here
│   └── structured.db     # Optional SQLite data
├── .env                  # Env variables (Tavily key)
├── requirements.txt
```

---

## 🔧 Setup Instructions

### 1. Clone and Setup

```bash
git clone https://github.com/your-username/rag_chatbot_with_speach.git
cd rag_chatbot_with_speach
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Add your `.env` file

Create a `.env` file in the root:

```env
TAVILY_API_KEY=your_tavily_api_key_here
```

---

## 📄 Embed PDF Documents

Put your files in `data/documents/` and run:

```bash
python run_once.py
```

---

## 💬 Run the Chatbot

```bash
streamlit run frontend/app.py
```

Then open [http://localhost:8501](http://localhost:8501)

---

## 🎙 Input Modes

- 🎤 **Voice input** via microphone
- ⌨️ **Keyboard input** via Streamlit

---

## 🌐 Web Search Fallback

If the PDF doesn’t contain the answer, the bot:
- Uses **Tavily API** for web search
- Returns a result like:  
  *"According to the web..."*  
If it comes from PDF:  
  *"According to the file provided..."*



---


## 🧰 Tech Stack

| Component         | Description                      |
|------------------|----------------------------------|
| LangChain        | RAG & orchestration              |
| LangGraph        | Graph-based query logic          |
| Ollama           | Local LLM (Mistral, LLaMA etc.)  |
| Streamlit        | Frontend web UI                  |
| Tavily API       | Live web search fallback         |
| Nomic Embeddings | Document vectorizer              |
| Pyttsx3          | Text-to-speech output            |
| SpeechRecognition| Speech-to-text input             |

---
