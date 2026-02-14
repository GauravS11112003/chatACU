"""
Hybrid RAG Chatbot — Company Regulations
Uses local (Ollama) or cloud (Google Gemini) LLM with a single local vector store (Chroma + Ollama embeddings).
"""

import os
import tempfile
from pathlib import Path
from typing import List

import chromadb
import streamlit as st
from chromadb.config import Settings as ChromaSettings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

# OCR imports
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

# -----------------------------------------------------------------------------
# Tesseract Configuration (Windows compatibility)
# -----------------------------------------------------------------------------
# Try to find Tesseract executable
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Tesseract-OCR\tesseract.exe",
]

def configure_tesseract():
    """Configure pytesseract with the correct path on Windows."""
    # First, check if tesseract is already accessible
    try:
        pytesseract.get_tesseract_version()
        return True  # Already configured
    except:
        pass
    
    # Try common installation paths
    for path in TESSERACT_PATHS:
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                pytesseract.get_tesseract_version()
                return True
            except:
                continue
    
    return False

# Configure on module load
TESSERACT_AVAILABLE = configure_tesseract()

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "regulations"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 4
OLLAMA_MODEL = "deepseek-r1:8b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
GEMINI_MODEL = "gemini-2.5-flash"


# -----------------------------------------------------------------------------
# Cached resources (single Chroma client to avoid "already exists" error)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_embeddings():
    """Load Ollama embeddings once (used for vector store only)."""
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)


@st.cache_resource
def get_chroma_client():
    """Single persistent Chroma client so all uses share the same settings."""
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


# -----------------------------------------------------------------------------
# Enhanced PDF processing with OCR support
# -----------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    """
    Extract text from PDF with automatic OCR fallback for scanned documents.
    Returns list of Document objects with page-level metadata.
    """
    documents = []
    
    # Check if OCR is available
    if not TESSERACT_AVAILABLE:
        st.warning(
            "⚠️ Tesseract OCR is not installed or not found. "
            "Scanned PDFs will not be processed. "
            "Only text-based PDFs will work.\n\n"
            "**To enable OCR for scanned documents:**\n"
            "1. Download from: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe\n"
            "2. Install to: `C:\\Program Files\\Tesseract-OCR\\`\n"
            "3. Restart this app\n\n"
            "See INSTALL_OCR.md for detailed instructions."
        )
    
    try:
        # Open PDF with PyMuPDF for better handling
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        # Progress tracking
        progress_bar = st.progress(0, text="Processing PDF pages...")
        
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            
            # Try extracting text first (fast for text-based PDFs)
            text = page.get_text().strip()
            
            # If no text or very little text, likely a scanned page - use OCR
            if len(text) < 50 and TESSERACT_AVAILABLE:  # Threshold for "mostly empty"
                try:
                    # Convert page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(
                        img,
                        lang='eng',
                        config='--psm 6 --oem 3'  # Assume uniform text block, use LSTM OCR
                    ).strip()
                    
                    if text:
                        st.info(f"📷 Page {page_num + 1}: Used OCR (scanned document)")
                    
                except Exception as ocr_error:
                    st.warning(f"OCR failed on page {page_num + 1}: {ocr_error}")
                    text = ""
            
            # Only add pages with content
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path,
                            "page": page_num + 1,
                            "total_pages": total_pages,
                        }
                    )
                )
            
            # Update progress
            progress_bar.progress(
                (page_num + 1) / total_pages,
                text=f"Processing page {page_num + 1} of {total_pages}..."
            )
        
        progress_bar.empty()
        pdf_document.close()
        
        return documents
        
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")
        return []


def process_pdf(pdf_path: str) -> Chroma | None:
    """
    Load PDF, split into chunks, and store in persistent Chroma.
    Supports both text-based and scanned PDFs with OCR.
    Returns the vector store or None on failure.
    """
    # Extract text with OCR support
    documents = extract_text_from_pdf(pdf_path)

    if not documents:
        st.warning("No text could be extracted from the PDF. Check if the file is valid.")
        return None
    
    # Show extraction stats
    total_chars = sum(len(doc.page_content) for doc in documents)
    st.success(f"✓ Extracted {len(documents)} pages, {total_chars:,} characters total")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    client = get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Collection may not exist yet

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME,
    )
    return vector_store


def get_vector_store() -> Chroma | None:
    """Load existing Chroma vector store from disk if it exists."""
    if not Path(CHROMA_DIR).exists():
        return None
    try:
        return Chroma(
            client=get_chroma_client(),
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
        )
    except Exception:
        return None


# -----------------------------------------------------------------------------
# LLM and RAG response
# -----------------------------------------------------------------------------
def get_llm(mode: str, api_key: str | None = None):
    """Return the configured LLM for the selected mode."""
    if mode == "Local (Privacy Mode)":
        return ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
    else:  # Cloud (Power Mode)
        key = (api_key or "").strip() or os.environ.get("GOOGLE_API_KEY")
        if not key:
            return None
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=key,
            temperature=0.2,
        )


def get_llm_response(
    llm,
    question: str,
    vector_store: Chroma,
    top_k: int = TOP_K_CHUNKS,
):
    """
    Retrieve top_k chunks from the vector store, build a context prompt, and stream the LLM response.
    """
    docs = vector_store.similarity_search(question, k=top_k)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    prompt = f"""You are a helpful assistant that answers questions based only on the following context from company regulations. If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    return llm.stream(prompt)


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Hybrid RAG — Company Regulations", layout="wide")
    st.title("Hybrid RAG Chatbot")
    st.caption("Ask questions about your uploaded Company Regulations PDF.")

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False
    if "current_pdf_name" not in st.session_state:
        st.session_state.current_pdf_name = None

    # ----- Sidebar -----
    with st.sidebar:
        st.header("Configuration")
        
        # Show OCR status
        if TESSERACT_AVAILABLE:
            st.success("✓ OCR enabled (scanned docs supported)")
        else:
            st.warning("⚠️ OCR disabled (only text PDFs)")

        # PDF upload
        uploaded_file = st.file_uploader("Upload PDF (Company Regulations)", type=["pdf"])
        if uploaded_file is not None:
            if st.session_state.current_pdf_name != uploaded_file.name:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                with st.spinner("Processing PDF and building vector store..."):
                    try:
                        vs = process_pdf(tmp_path)
                        if vs is not None:
                            st.session_state.vector_store_ready = True
                            st.session_state.current_pdf_name = uploaded_file.name
                            st.success(f"Loaded: {uploaded_file.name}")
                        else:
                            st.session_state.vector_store_ready = False
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.vector_store_ready = False
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
            else:
                st.info(f"Using existing index: {uploaded_file.name}")
        else:
            st.session_state.current_pdf_name = None
            # Check if we have an existing DB from a previous run
            if not st.session_state.vector_store_ready:
                vs_existing = get_vector_store()
                if vs_existing is not None:
                    try:
                        if vs_existing._collection.count() > 0:
                            st.session_state.vector_store_ready = True
                            st.info("Using existing vector store from disk.")
                    except Exception:
                        pass

        st.divider()

        # Model selector
        mode = st.radio(
            "Reasoning Engine",
            options=["Local (Privacy Mode)", "Cloud (Power Mode)"],
            index=0,
            help="Local uses Ollama (deepseek-r1:8b), Cloud uses Google Gemini (gemini-1.5-pro).",
        )

        api_key = None
        if mode == "Cloud (Power Mode)":
            api_key = st.text_input(
                "Google API Key",
                type="password",
                placeholder="Enter GOOGLE_API_KEY",
                help="Required for Gemini. You can also set GOOGLE_API_KEY in your environment.",
            )

    # ----- Main chat -----
    vector_store = get_vector_store() if st.session_state.vector_store_ready else None
    if not vector_store and uploaded_file is None:
        st.info("Upload a PDF in the sidebar to get started.")
    elif not st.session_state.vector_store_ready:
        st.info("Upload a PDF in the sidebar to build the knowledge base.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the regulations..."):
        if not st.session_state.vector_store_ready and not vector_store:
            st.warning("Please upload a PDF first.")
            st.stop()

        vs = vector_store or get_vector_store()
        if vs is None:
            st.warning("Vector store not ready. Please upload a PDF again.")
            st.stop()

        llm = None
        if mode == "Local (Privacy Mode)":
            llm = get_llm(mode)
        else:
            llm = get_llm(mode, api_key)
            if llm is None:
                st.error("Please enter your Google API Key in the sidebar to use Cloud (Power Mode).")
                st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                stream = get_llm_response(llm, prompt, vs)
                response_placeholder = st.empty()
                full_response = []
                for chunk in stream:
                    if hasattr(chunk, "content") and chunk.content:
                        full_response.append(chunk.content)
                        response_placeholder.markdown("".join(full_response))
                final = "".join(full_response)
            except Exception as e:
                final = f"Sorry, an error occurred: {e}"
                if "Ollama" in str(e) or "connection" in str(e).lower():
                    st.warning("Is Ollama running? Try: `ollama run deepseek-r1:8b`")
                response_placeholder.markdown(final)

        st.session_state.messages.append({"role": "assistant", "content": final})

    st.session_state.vector_store_ready = (
        st.session_state.vector_store_ready or (get_vector_store() is not None)
    )


if __name__ == "__main__":
    main()
