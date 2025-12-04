import os
import sys
import getpass
import pickle
import numpy as np
import faiss
import re
import csv
import math
import warnings
import sqlite3
import hashlib
import time
import traceback
import shutil
import atexit
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict

# Suppress known Pydantic/LangChain warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core._api.deprecation")

import pypdf
from rank_bm25 import BM25Okapi  
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver 
from langchain_text_splitters import RecursiveCharacterTextSplitter

# CLI Colors
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Configuration
API_KEY = "AIzaSyAJomzd4F9N0D_j-Dv19z5NuHo-YLyPaH0" # Paste key here if needed
DB_DIR = "vector_store_db" 
INDEX_FILE = os.path.join(DB_DIR, "index.faiss")
METADATA_FILE = os.path.join(DB_DIR, "metadata.pkl")
MODEL_NAME = "gemini-2.0-flash"

if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY
elif not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass(f"{TermColors.YELLOW}Enter Google API Key:{TermColors.ENDC} ")

def cleanup_resources():
    """Wipe temp database on exit."""
    try:
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
            print(f"\n{TermColors.YELLOW}Session data cleared.{TermColors.ENDC}")
    except Exception as e:
        print(f"Cleanup error: {e}")

atexit.register(cleanup_resources)

class HybridVectorStore:
    """
    Combines FAISS (dense vector search) and BM25 (keyword search) 
    using Reciprocal Rank Fusion.
    """
    def __init__(self):
        # Force REST transport to avoid GRPC issues on some Windows envs
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            transport="rest" 
        )
        self.dimension = 768 
        self.documents = []    
        self.metadatas = []
        self.file_hashes = set()
        self.index = None      
        self.bm25 = None       
        
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            self.load_index()
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            if not os.path.exists(DB_DIR):
                os.makedirs(DB_DIR)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def is_duplicate(self, file_path: str) -> bool:
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash in self.file_hashes
        except:
            return False

    def add_documents(self, chunks: List[str], metadatas: List[Dict], file_path: str):
        try:
            with open(file_path, "rb") as f:
                self.file_hashes.add(hashlib.md5(f.read()).hexdigest())
        except: pass

        print(f"{TermColors.CYAN}Indexing {len(chunks)} chunks...{TermColors.ENDC}")
        batch_size = 50
        
        for i in range(0, len(chunks), batch_size):
            batch_text = chunks[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            
            # Simple retry mechanism for API rate limits
            for attempt in range(3):
                try:
                    embeddings = self.embeddings_model.embed_documents(batch_text)
                    break
                except Exception as e:
                    if attempt == 2: print(f"{TermColors.FAIL}Embedding failed: {e}{TermColors.ENDC}")
                    time.sleep(2)

            embedding_matrix = np.array(embeddings).astype('float32')
            self.index.add(embedding_matrix)
            
            self.documents.extend(batch_text)
            self.metadatas.extend(batch_meta)
            print(f"Processed {min(i+batch_size, len(chunks))}/{len(chunks)}...", end='\r')
        
        print(f"\n{TermColors.CYAN}Building BM25 index...{TermColors.ENDC}")
        tokenized_corpus = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save_index()
        print(f"{TermColors.GREEN}Indexing complete.{TermColors.ENDC}")

    def save_index(self):
        faiss.write_index(self.index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadatas": self.metadatas,
                "bm25": self.bm25,
                "hashes": self.file_hashes
            }, f)

    def load_index(self):
        print(f"{TermColors.CYAN}Loading index...{TermColors.ENDC}")
        self.index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadatas = data["metadatas"]
            self.bm25 = data.get("bm25")
            self.file_hashes = data.get("hashes", set())
            
            if self.bm25 is None and self.documents:
                # Rebuild BM25 if loading from an older index version
                tokenized_corpus = [self._tokenize(doc) for doc in self.documents]
                self.bm25 = BM25Okapi(tokenized_corpus)

    def keyword_search(self, query: str, k: int = 20) -> List[int]:
        if not self.bm25: return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        return np.argsort(scores)[::-1][:k].tolist()

    def vector_search(self, query: str, k: int = 20) -> Tuple[List[int], List[float]]:
        query_emb = self.embeddings_model.embed_query(query)
        vector = np.array([query_emb]).astype('float32')
        distances, indices = self.index.search(vector, k)
        return indices[0].tolist(), distances[0].tolist()

    def reciprocal_rank_fusion(self, keyword_indices: List[int], vector_indices: List[int], k: int = 60):
        """Standard RRF algorithm to merge rank lists."""
        ranks = defaultdict(float)
        for rank, doc_idx in enumerate(keyword_indices):
            ranks[doc_idx] += 1 / (rank + k)
        for rank, doc_idx in enumerate(vector_indices):
            ranks[doc_idx] += 1 / (rank + k)
        return sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    def hybrid_search(self, query: str, k: int = 5) -> str:
        # Fetch a larger pool of candidates before re-ranking
        pool_k = k * 3
        keyword_ids = self.keyword_search(query, k=pool_k)
        vector_ids, _ = self.vector_search(query, k=pool_k)
        fused_results = self.reciprocal_rank_fusion(keyword_ids, vector_ids)
        
        results = []
        for doc_idx, score in fused_results[:k]:
            if doc_idx < 0 or doc_idx >= len(self.documents): continue
            
            doc_text = self.documents[doc_idx]
            meta = self.metadatas[doc_idx]
            src = meta.get('source', 'Unknown')
            # Handle different metadata keys for CSV vs PDF
            loc = f"Page {meta.get('page')}" if 'page' in meta else f"Row {meta.get('row')}"
            
            results.append(f"[SOURCE: {src} | {loc} | Score: {score:.4f}]\n{doc_text}")
            
        return "\n\n---\n\n".join(results)

# Global store for tool access
vector_store = HybridVectorStore()

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the document knowledge base. 
    Use this to find ANY information, data, numbers, or facts.
    """
    return vector_store.hybrid_search(query, k=8) 

@tool
def calculator(expression: str) -> str:
    """
    Safe calculator. Input: a math string (e.g. '4500 * 2200').
    """
    try:
        allowed_names = {"sum": sum, "min": min, "max": max, "abs": abs, "round": round, "math": math}
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of '{name}' is not allowed")
        return str(eval(code, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"Error: {e}"

# --- LangGraph Setup ---

tools = [search_knowledge_base, calculator]
tool_node = ToolNode(tools)

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME, 
    temperature=0, 
    max_retries=2,
    transport="rest"
)
llm_with_tools = llm.bind_tools(tools)

# Strict prompt to force tool usage for data analysis
SYSTEM_PROMPT = """You are a Research & Data Analyst.

PROTOCOL:
1. **Retrieve First**: If asked for metrics (e.g., profit, headcount), you MUST call `search_knowledge_base` first. Never guess.
2. **Calculate Second**: Once you have the raw numbers from search, use the `calculator` tool. Do not do mental math for comparisons or aggregations.
3. **Reasoning**: If specific data is missing, try a broader search term.
4. **Context**: Assume names and projects mentioned exists within the documents provided.

Answer concisely with citations [SOURCE]."""

def agent_node(state: MessagesState):
    messages = state["messages"]
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

def process_file(file_path: str):
    if vector_store.is_duplicate(file_path):
        print(f"{TermColors.YELLOW}File '{os.path.basename(file_path)}' is already indexed.{TermColors.ENDC}")
        return

    print(f"{TermColors.BLUE}Processing {file_path}...{TermColors.ENDC}")
    ext = os.path.splitext(file_path)[1].lower()
    all_chunks = []
    all_metadatas = []
    
    try:
        # Larger chunk size helps keep table rows together
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=300
        )

        if ext == '.pdf':
            reader = pypdf.PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    chunks = text_splitter.split_text(text)
                    all_chunks.extend(chunks)
                    all_metadatas.extend([{"page": i+1, "source": os.path.basename(file_path)} for _ in chunks])

        elif ext == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                batch_rows = []
                batch_idx = []
                for i, row in enumerate(reader):
                    row_str = ", ".join([f"{k}: {v}" for k, v in row.items() if v])
                    batch_rows.append(row_str)
                    batch_idx.append(i+1)
                    if len(batch_rows) >= 10: 
                        all_chunks.append("\n".join(batch_rows))
                        all_metadatas.append({"row": f"{batch_idx[0]}-{batch_idx[-1]}", "source": os.path.basename(file_path)})
                        batch_rows = []
                        batch_idx = []
                if batch_rows:
                    all_chunks.append("\n".join(batch_rows))
                    all_metadatas.append({"row": f"{batch_idx[0]}-{batch_idx[-1]}", "source": os.path.basename(file_path)})

        elif ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = text_splitter.split_text(text)
                all_chunks.extend(chunks)
                all_metadatas.extend([{"page": "1", "source": os.path.basename(file_path)} for _ in chunks])

        if all_chunks:
            vector_store.add_documents(all_chunks, all_metadatas, file_path)
        else:
            print(f"{TermColors.FAIL}No text extracted.{TermColors.ENDC}")

    except Exception as e:
        print(f"{TermColors.FAIL}Error processing file: {e}{TermColors.ENDC}")

def main():
    print(f"\n{TermColors.HEADER}Local RAG Agent [{MODEL_NAME}]{TermColors.ENDC}")
    
    if len(vector_store.documents) == 0:
        fpath = input("Path to file (PDF/CSV/TXT): ").strip().replace('"', '')
        if os.path.exists(fpath): process_file(fpath)
    else:
        print(f"{TermColors.GREEN}System loaded ({len(vector_store.documents)} chunks).{TermColors.ENDC}")
        if input("Add another file? (y/n): ").lower() == 'y':
            fpath = input("Path to file: ").strip().replace('"', '')
            if os.path.exists(fpath): process_file(fpath)

    print(f"\n{TermColors.BOLD}Ready. Type 'exit' to quit.{TermColors.ENDC}")
    config = {"configurable": {"thread_id": "session_1"}}
    
    while True:
        try:
            user_input = input(f"\n{TermColors.BOLD}User:{TermColors.ENDC} ")
            if user_input.lower() in ["exit", "quit"]: break
            
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            for event in graph.stream(inputs, config=config):
                for key, val in event.items():
                    if key == "agent":
                        msg = val["messages"][-1]
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"   {TermColors.YELLOW}Tool: {tc['name']}({tc['args']}){TermColors.ENDC}")
                        else:
                            content = msg.content
                            # Parse Gemini's occasional list response
                            if isinstance(content, list):
                                text_parts = [part['text'] for part in content if isinstance(part, dict) and 'text' in part]
                                print(f"\n{TermColors.GREEN}Agent:{TermColors.ENDC} {' '.join(text_parts) if text_parts else content}")
                            else:
                                print(f"\n{TermColors.GREEN}Agent:{TermColors.ENDC} {content}")
                    elif key == "tools":
                        print(f"   {TermColors.BLUE}Tool executed.{TermColors.ENDC}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            if "404" in str(e) and "models/" in str(e):
                print(f"\n{TermColors.FAIL}Error: Model not found.{TermColors.ENDC}")
            else:
                print(f"\n{TermColors.FAIL}Error occurred:{TermColors.ENDC}")
                traceback.print_exc() 

if __name__ == "__main__":
    main()