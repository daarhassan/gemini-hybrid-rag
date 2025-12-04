**Gemini Hybrid RAG**

**Gemini-Hybrid-RAG** is a research agent that helps you analyze documents. It uses a "Retrieval-Augmented Generation" (RAG) approach, but adds a few smarter features. Instead of just searching for text, it uses a hybrid search system (combining meaning and keywords) and an agent that can plan its actions.

It effectively reads PDF, CSV, and text files, allows you to ask questions about them, and performs calculations when necessary.

**Main Features**

- **Hybrid Search:** It searches your documents in two ways at once: looking for concepts (using FAISS) and looking for exact words (using BM25). It combines these results to find the most relevant information.
- **Smart Agent:** Powered by **Google Gemini 2.0 Flash**, the system doesn't just guess. It can decide to search multiple times or use a calculator if your question requires math.
- **Calculator Tool:** It has a built-in calculator so it doesn't make math errors when analyzing financial or statistical data from your files.
- **File Support:** Works with PDFs, CSVs, and text files.
- **Memory:** The bot remembers what you said earlier in the conversation while it is running.

**Installation**

**Prerequisites**

- **Python 3.11** (Recommended for compatibility).
- **Google Gemini API Key**.

**Steps**

- **Clone the project**
- git clone <https://github.com/daarhassan/gemini-hybrid-rag.git>
- cd gemini-hybrid-rag
- Install libraries

Run this command to install the necessary packages:

pip install langchain-google-genai langchain langgraph langchain-community faiss-cpu pypdf rank_bm25

**Configuration**

You need to provide your Google API Key for the AI to work.

Option 1: Paste it in the code (Easiest)

Open rag_agent_gemini.py, find the HARDCODED_KEY line near the top, and paste your key inside the quotes:

HARDCODED_KEY = "Your-Key-Here"

Option 2: Enter it when running

If you leave the HARDCODED_KEY variable empty, the script will ask you to type your key securely when you start it.

**How to Run**

- **Start the script**
- python rag_agent_gemini.py
- Load a file

The script will ask for a file path. Paste the path to your PDF, CSV, or text file. It will take a moment to read and index it.

- Chat

Once loaded, you can ask questions like:

- - "What is the summary of this document?"
    - "Calculate the total cost listed in the project table."

**Note:** The script creates a folder named vector_store_db_gemini to store the indexed data while running. This is automatically deleted when you type exit to close the program.
