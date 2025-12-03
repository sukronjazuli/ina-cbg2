import streamlit as st
import os
import sys
import json
from dotenv import load_dotenv

# Configure page to hide menu at the very beginning
st.set_page_config(
    page_title="Analisis Dokumen AI dengan RAG",
    page_icon="📄",
    menu_items=None
)

# Pustaka LangChain & Komponen AI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- TAHAP 0: KONFIGURASI DAN VALIDASI LINGKUNGAN ---
def setup_environment():
    """Memuat API key dari Streamlit secrets atau .env file."""
    api_key = None
    
    # Coba ambil dari Streamlit secrets (untuk deployment)
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    except:
        pass
    
    # Jika tidak ada di secrets, coba dari .env file (untuk development lokal)
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
    
    # Validasi API key
    if not api_key:
        st.error("❌ **ERROR: GOOGLE_API_KEY tidak ditemukan!**")
        
        tab1, tab2 = st.tabs(["☁️ Streamlit Cloud", "💻 Development Lokal"])
        
        with tab1:
            st.markdown("**Untuk deployment di Streamlit Cloud:**")
            st.code('GOOGLE_API_KEY = "your_api_key_here"', language="toml")
            st.caption("1. Buka Settings > Secrets\n2. Paste kode di atas\n3. Ganti dengan API key Anda\n4. Save")
        
        with tab2:
            st.markdown("**Untuk development lokal:**")
            st.code('GOOGLE_API_KEY=your_api_key_here', language="text")
            st.caption("1. Buat file .env di root folder\n2. Paste kode di atas\n3. Ganti dengan API key Anda\n4. Restart app")
        
        st.link_button("🔑 Dapatkan API Key", "https://makersuite.google.com/app/apikey")
        st.stop()
    
    # Set environment variable agar bisa digunakan oleh library
    os.environ["GOOGLE_API_KEY"] = api_key

# --- TAHAP 1: INGESTION (PEMUATAN & PEMISAHAN DOKUMEN) ---
@st.cache_data
def load_and_split_documents(file_path):
    """Memuat teks dan memecahnya menjadi potongan-potongan (chunks)."""
    with st.spinner("Mengurai file dokumen..."):
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            st.success(f"Berhasil memecah dokumen menjadi {len(chunks)} bagian.")
            return chunks
        except Exception as e:
            st.error(f"ERROR saat memuat/memecah dokumen: {e}")
            st.stop()

# --- TAHAP 2: EMBEDDING & INDEXING (MEMBUAT DAN MENYIMPAN VEKTOR) ---
@st.cache_resource
def index_documents(_chunks):
    """Mengkonversi chunks menjadi vektor dan menyimpannya di Vector Store (FAISS)."""
    with st.spinner("Membuat Embedding dan Indexing dengan FAISS..."):
        try:
            embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            # Menggunakan FAISS sebagai pengganti ChromaDB
            db = FAISS.from_documents(_chunks, embedding_model)
            st.success("Vektor telah tersimpan di Vector Database (FAISS).")
            return db
        except Exception as e:
            st.error(f"ERROR saat membuat index FAISS: {e}")
            st.error("Pastikan 'faiss-cpu' terinstal dengan benar.")
            st.stop()

# --- TAHAP 3: GENERATION (TANYA JAWAB MENGGUNAKAN GEMINI API) ---
def run_qa_chain(db, query):
    """Menjalankan rantai RAG (Retrieval-Augmented Generation) menggunakan Gemini."""
    with st.spinner("Menjalankan RAG Chain (Retrieval & Generation)..."):
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, convert_system_message_to_human=True)

            # Menggunakan 'from_chain_type' yang merupakan metode standar
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": 5})
            )

            st.info(f"Mengajukan pertanyaan: '{query}'")
            # Menggunakan metode invoke yang lebih modern dengan instruksi bahasa Indonesia
            response = qa_chain.invoke(f"Jawab dalam bahasa Indonesia: {query}")
            return response['result']
        except Exception as e:
            st.error(f"ERROR saat menjalankan Q&A: {e}")
            st.stop()

# --- MAIN STREAMLIT APP ---
def main():
    # Hide Streamlit menu and footer
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}
    .css-1rs6os {visibility: hidden;}
    .css-1lsmgbg {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none;}
    .viewerBadge_link__1S137 {display: none;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.title("📄 Analisis Dokumen AI dengan RAG")
    st.markdown("Aplikasi untuk menganalisis dokumen menggunakan Retrieval-Augmented Generation (RAG) dengan Gemini AI.")

    # Setup environment
    setup_environment()

    # History file
    history_file = "chat_history.json"

    # Load history from file if exists
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                st.session_state.history = json.load(f)
        except:
            st.session_state.history = []
    else:
        st.session_state.history = []

    # File configuration
    text_file = "contoh_dokumen_extracted_extracted.txt"

    if not os.path.exists(text_file):
        st.error(f"ERROR: File '{text_file}' tidak ditemukan. Silakan ganti nama file atau letakkan file Anda.")
        st.stop()

    # Load and process document
    try:
        document_chunks = load_and_split_documents(text_file)
        vector_db = index_documents(document_chunks)

        # Display chat history
        for chat in st.session_state.history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])

        # Chat input
        pertanyaan_user = st.chat_input("Ajukan pertanyaan Anda tentang dokumen:")

        if pertanyaan_user:
            if pertanyaan_user.strip():
                # Run Q&A
                final_answer = run_qa_chain(vector_db, pertanyaan_user)

                # Add to history
                st.session_state.history.append({"question": pertanyaan_user, "answer": final_answer})

                # Save history to file
                with open(history_file, "w", encoding="utf-8") as f:
                    json.dump(st.session_state.history, f, ensure_ascii=False, indent=4)

                # Display new chat messages
                with st.chat_message("user"):
                    st.write(pertanyaan_user)
                with st.chat_message("assistant"):
                    st.write(final_answer)
            else:
                st.warning("Silakan masukkan pertanyaan terlebih dahulu.")

    except Exception as e:
        st.error(f"TERJADI KESALAHAN FATAL SELAMA EKSEKUSI: {e}")
        st.error("Periksa kembali instalasi pustaka, koneksi internet, dan API Key Anda.")

if __name__ == "__main__":
    main()
