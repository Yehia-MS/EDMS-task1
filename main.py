# File: app.py
# Description: A complete and functional Agentic RAG application using Google Gemini.

import os
import re
import json
import sqlite3
import shutil
import io
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta

# --- Third-Party Libraries ---
import streamlit as st
from PyPDF2 import PdfReader, errors
from dotenv import load_dotenv
from st_audiorec import st_audiorec
from gtts import gTTS # Google Text-to-Speech library
import pytz # Added for timezone-aware calculations

# --- LangChain & Google AI ---
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

# Load environment variables and initialize Google AI client
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    st.error(f"Failed to configure Google AI. Please ensure your GOOGLE_API_KEY is set. Error: {e}")
    st.stop()


# --- Configuration ---
from dataclasses import dataclass

@dataclass
class AppConfig:
    """Centralized configuration for the application."""
    APP_DATA_DIR: Path = Path("app_data")
    ARCHIVE_SUBDIR: str = "chat_archives"
    DOCS_DIR: str = "docs"
    DB_FILENAME: str = "memory.db"
    EMBEDDING_MODEL: str = "models/embedding-001"
    GENERATIVE_MODEL: str = "gemini-1.5-flash"
    # GENERATIVE_MODEL: str = "gemini-1.5-pro"
    TEMP_AUDIO_FILENAME: str = "temp_audio.wav"

config = AppConfig()

# ---------- Memory Manager with Archiving ----------
class MemoryManager:
    """Persistent conversation memory stored in SQLite with archiving."""
    def __init__(self, db_path: Path, archive_dir: Path):
        self.db_path = db_path
        self.archive_dir = archive_dir
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.conn = self._get_connection()
        self._create_table()

    def _get_connection(self):
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def _create_table(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        self.conn.commit()

    def add(self, role: str, content: str):
        db_role = 'user' if role == 'user' else 'model'
        self.conn.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)",
            (db_role, content)
        )
        self.conn.commit()

    def get_last(self, n: int = 50) -> List[Dict[str, str]]:
        cur = self.conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (n,)
        )
        rows = cur.fetchall()[::-1]
        return [{"role": r if r in ['user', 'model'] else 'user', "content": c} for r, c in rows]

    def _clear_current_session(self):
        self.conn.close()
        if self.db_path.exists():
            os.remove(self.db_path)
        self.conn = self._get_connection()
        self._create_table()

    def archive_session(self) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages")
        if cursor.fetchone()[0] < 2:
            return False
        existing = self.archive_dir.glob("memory*.db")
        indices = [int(re.search(r'(\d+)', f.name).group(1)) for f in existing if re.search(r'(\d+)', f.name)]
        next_index = max(indices) + 1 if indices else 1
        archive_path = self.archive_dir / f"memory{next_index}.db"
        self.conn.close()
        shutil.copy2(self.db_path, archive_path)
        self._clear_current_session()
        return True


# ---------- PDF Loader & Retriever ----------
@st.cache_resource
def init_vectorstore(pdf_dir: str = config.DOCS_DIR) -> FAISS:
    """Load all PDFs by page, embed and index with FAISS using Google Embeddings"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Embedding model failed to initialize. Check your API Key. Error: {e}")
        return None

    docs: List[Document] = []
    Path(pdf_dir).mkdir(exist_ok=True)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        st.error(f"No PDF files found in the '{config.DOCS_DIR}' directory. Please add your PDFs there.")
        return None

    with st.spinner(f"Loading {len(pdf_files)} PDF(s)..."):
        for fn in pdf_files:
            path = os.path.join(pdf_dir, fn)
            try:
                reader = PdfReader(path)
                for i, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": fn, "page": i}
                        ))
            except (errors.PdfReadError, Exception) as e:
                st.warning(f"Skipping PDF '{fn}': {e}")
                continue
    if not docs:
        st.error("Could not extract any text from the PDFs. The agent will not have any context.")
        return None

    st.success(f"Successfully loaded {len(docs)} pages from {len(pdf_files)} PDF(s).")
    return FAISS.from_documents(docs, embeddings)

# ---------- Agentic RAG Logic with Gemini ----------
class AgenticRAG:
    """The main RAG agent, using Google Gemini."""
    def __init__(self):
        self.memory = MemoryManager(
            db_path=config.APP_DATA_DIR / config.DB_FILENAME,
            archive_dir=config.APP_DATA_DIR / config.ARCHIVE_SUBDIR
        )
        self.vstore = init_vectorstore()
        self.llm_client = genai.GenerativeModel(config.GENERATIVE_MODEL)
        os.makedirs("fetched_data", exist_ok=True)
        config.APP_DATA_DIR.mkdir(exist_ok=True)

    def _handle_api_call(self, api_call_func, *args, **kwargs):
        """A wrapper for API calls to handle quota errors gracefully."""
        try:
            return api_call_func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str and "quota" in error_str:
                st.session_state.quota_exceeded = True
                st.rerun()
            raise e

    def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribes audio data using Gemini."""
        temp_audio_path = config.APP_DATA_DIR / config.TEMP_AUDIO_FILENAME
        with open(temp_audio_path, "wb") as f:
            f.write(audio_data)
        try:
            audio_file = genai.upload_file(path=temp_audio_path)
            response = self._handle_api_call(
                self.llm_client.generate_content,
                ["Transcribe this audio accurately.", audio_file]
            )
            genai.delete_file(audio_file.name)
            return response.text.strip()
        finally:
            if temp_audio_path.exists():
                os.remove(temp_audio_path)

    def rewrite_query(self, query: str) -> str:
        """Refine user question into a concise retrieval query using Gemini."""
        prompt = f"You are an expert query rewriter. Convert the following user question into a keyword-based search query. Remove stop words and conversational phrases.\n\nOriginal question: '{query}'\n\nRewritten query:"
        try:
            response = self._handle_api_call(self.llm_client.generate_content, prompt)
            return response.text.strip().replace('"', '')
        except Exception:
            return query

    def fetch_local(self, query: str, top_k: int) -> List[Document]:
        """Retrieve top_k relevant pages from local PDFs."""
        if not query or not self.vstore:
            st.warning("Vector store not initialized. Cannot perform search.")
            return []
        return self.vstore.similarity_search(query, k=top_k)

    def generate_answer(self, query: str, docs: List[Document], history: List[Dict[str, str]]) -> str:
        """Produce final answer with citations using Gemini."""
        if not docs:
            return "There is no information about that in the provided documents. Therefore, I cannot answer your question."

        system_prompt = "You are a helpful assistant. Synthesize an answer based ONLY on the provided documents. Cite sources for every piece of information using the format `[filename, page <page_number>]`. If the documents do not contain the answer, you MUST state that the information was not found."
        docs_text = "\n---\n".join(
            f"Context from `[{d.metadata['source']}, page {d.metadata['page']}]`:\n{d.page_content}"
            for d in docs
        )
        user_content = f"Available Documents:\n{docs_text}\n\nUser Query: {query}"

        gemini_formatted_history = [
            {'role': msg['role'], 'parts': [msg['content']]} for msg in history
        ]
        gemini_formatted_history.append({'role': 'user', 'parts': [f"{system_prompt}\n\n{user_content}"]})

        try:
            response = self._handle_api_call(self.llm_client.generate_content, gemini_formatted_history)
            answer = response.text
        except Exception as e:
            # This block will now primarily catch non-quota errors
            st.error(f"An error occurred while generating the answer: {e}")
            return "Sorry, I encountered an error while trying to generate a response."

        cites = re.findall(r"\[([^,\]]+),\s*page\s*(\d+)\]", answer, re.IGNORECASE)
        unique = sorted(list(set([f"{file.strip().replace('`', '')}, page {pg}" for file, pg in cites])))
        if unique:
            sources_list = "\n".join(f"- {u}" for u in unique)
            answer += f"\n\n**Sources:**\n{sources_list}"
        return answer

    def handle(self, query: str) -> Tuple[str, List[Document], str]:
        """Handles the main logic, returning the answer, docs, and rewritten query."""
        self.memory.add("user", query)

        refined_query = self.rewrite_query(query)
        docs = self.fetch_local(refined_query, top_k=4)

        history = self.memory.get_last(10)
        final_answer = self.generate_answer(refined_query, docs, history)
        self.memory.add("model", final_answer)

        return final_answer, docs, refined_query

# ---------- UI Helper Functions ----------
def get_time_until_reset() -> timedelta:
    """Calculates the time remaining until the next midnight Pacific Time."""
    pt = pytz.timezone('US/Pacific')
    now_pt = datetime.now(pt)
    midnight_pt = now_pt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return midnight_pt - now_pt

def play_audio_response(text: str):
    """Generates and plays audio using gTTS."""
    with st.spinner("Generating audio..."):
        try:
            clean_text = re.sub(r'\[[^\]]+\]|\*\*Sources:\*\*|\n-.*|`', '', text).strip()
            if not clean_text: return
            tts = gTTS(text=clean_text, lang='en')
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            st.audio(audio_buffer, format='audio/mp3', autoplay=True)
        except Exception as e:
            st.error(f"Audio generation failed: {e}")

def display_archive(db_path: Path):
    """Renders a read-only view of an archived chat."""
    st.subheader(f"Viewing Archive: {db_path.name}")
    try:
        conn = sqlite3.connect(str(db_path))
        messages = conn.execute("SELECT role, content FROM messages ORDER BY id ASC").fetchall()
        conn.close()
        for role, content in messages:
            st.chat_message("assistant" if role == 'model' else role).write(content)
    except Exception as e:
        st.error(f"Could not load archive: {e}")

def render_sidebar(agent: AgenticRAG) -> Tuple[bool, Any, str]:
    """Renders the full sidebar UI."""
    with st.sidebar:
        st.header("⚙️ Controls")
        audio_enabled = st.toggle("🔊 Audio Responses", value=False)
        if st.button("📦 Archive & Start New Chat", use_container_width=True):
            if agent.memory.archive_session():
                st.success("Chat archived!"); st.rerun()
            else:
                st.warning("Not enough messages to archive.")
        if st.button("🔄 Reload Docs & Clear Cache", use_container_width=True, type="primary"):
            st.cache_resource.clear(); st.success("Cache cleared!"); st.rerun()
        st.markdown("---")
        st.header("🎙️ Voice Input")
        wav_audio_data = st_audiorec()
        st.markdown("---")
        st.header("📂 Archives")
        archive_dir = config.APP_DATA_DIR / config.ARCHIVE_SUBDIR
        archive_files = sorted([f for f in archive_dir.glob("*.db")], reverse=True)
        archive_choice = st.selectbox("View an archive", ["None"] + [f.name for f in archive_files])
    return audio_enabled, wav_audio_data, archive_choice

# ---------- Main Streamlit Application ----------
def main():
    st.set_page_config(page_title="Agentic RAG with Gemini", layout="wide")
    st.title("⚡️ Agentic RAG ")

    # --- NEW: Quota Management UI ---
    if st.session_state.get('quota_exceeded', False):
        time_left = get_time_until_reset()
        if time_left.total_seconds() <= 0:
            st.session_state.quota_exceeded = False
            st.rerun()
        else:
            hours, remainder = divmod(int(time_left.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            st.error(
                f"🚨 **Daily API Quota Exceeded!**\n\n"
                f"You have used all your free API requests for today. "
                f"The session will resume in **{hours}h {minutes}m {seconds}s**.\n\n"
                f"*(Quota resets at midnight Pacific Time. For higher limits, check your Google AI plan and billing details.)*"
            )
            st.stop() # Stop further execution of the page until quota resets

    if "agent" not in st.session_state:
        st.session_state.agent = AgenticRAG()

    agent = st.session_state.agent
    audio_enabled, wav_audio_data, archive_choice = render_sidebar(agent)

    if archive_choice != "None":
        display_archive(config.APP_DATA_DIR / config.ARCHIVE_SUBDIR / archive_choice)
        st.stop()

    for msg in agent.memory.get_last(50):
        st.chat_message("assistant" if msg["role"] == "model" else msg["role"]).write(msg["content"])

    if wav_audio_data:
        with st.spinner("Transcribing audio..."):
            st.session_state.user_prompt = agent.transcribe_audio(wav_audio_data)
        st.rerun()
    
    # Disable chat input if quota is exceeded (double check for safety)
    chat_input_disabled = st.session_state.get('quota_exceeded', False)
    prompt = st.chat_input(
        "Ask me anything about your documents...",
        disabled=chat_input_disabled,
        key="main_chat_input"
    )

    if "user_prompt" in st.session_state and st.session_state.user_prompt:
        prompt = st.session_state.pop("user_prompt")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                final_answer, retrieved_docs, rewritten_query = agent.handle(prompt)

            st.info(f"**Rewritten Query for Search:** `{rewritten_query}`")

            if retrieved_docs:
                with st.expander("📚 Retrieved Document Sections"):
                    for doc in retrieved_docs:
                        st.markdown(f"**Source:** {doc.metadata['source']}, Page {doc.metadata['page']}")
                        st.markdown(f"> {doc.page_content[:300].strip()}...")
            else:
                st.warning("No relevant document sections were found for this query.")
            
            st.markdown(final_answer)

            if audio_enabled:
                play_audio_response(final_answer)

if __name__ == "__main__":
    main()