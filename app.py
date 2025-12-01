import streamlit as st
import os
import sys
import json
import uuid
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict

# Configure page to hide menu at the very beginning
st.set_page_config(
    page_title="Analisis Dokumen AI dengan RAG",
    page_icon="📄",
    layout="wide",
    menu_items=None
)

# Pustaka LangChain & Komponen AI
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# --- USER SESSION MANAGEMENT ---
def get_or_create_user_id():
    """Membuat atau mengambil user ID unik dengan persistensi menggunakan file."""
    if 'user_id' not in st.session_state:
        # Coba load dari file lokal dulu
        user_id = load_user_id_from_file()
        
        if not user_id:
            # Jika tidak ada, buat user_id baru
            user_id = str(uuid.uuid4())
            save_user_id_to_file(user_id)
        
        st.session_state.user_id = user_id
        st.session_state.session_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return st.session_state.user_id

def load_user_id_from_file():
    """Load user ID dari file lokal."""
    user_id_file = ".streamlit_user_id"
    if os.path.exists(user_id_file):
        try:
            with open(user_id_file, 'r', encoding='utf-8') as f:
                user_id = f.read().strip()
                if user_id:  # Pastikan tidak kosong
                    return user_id
        except Exception as e:
            print(f"Error loading user ID: {e}")
    return None

def save_user_id_to_file(user_id):
    """Simpan user ID ke file lokal."""
    user_id_file = ".streamlit_user_id"
    try:
        with open(user_id_file, 'w', encoding='utf-8') as f:
            f.write(user_id)
        return True
    except Exception as e:
        print(f"Error saving user ID: {e}")
        return False

def reset_user_session():
    """Reset user session - hapus file user_id dan buat yang baru."""
    user_id_file = ".streamlit_user_id"
    if os.path.exists(user_id_file):
        try:
            os.remove(user_id_file)
        except:
            pass
    
    # Clear session state
    if 'user_id' in st.session_state:
        del st.session_state.user_id
    if 'current_conversation_id' in st.session_state:
        del st.session_state.current_conversation_id
    if 'current_messages' in st.session_state:
        del st.session_state.current_messages
    if 'conversation_title' in st.session_state:
        del st.session_state.conversation_title

def get_user_conversations(user_id):
    """Mengambil daftar semua percakapan user."""
    history_dir = "user_histories"
    user_dir = os.path.join(history_dir, user_id)
    
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
        return []
    
    conversations = []
    for filename in os.listdir(user_dir):
        if filename.endswith('.json'):
            conv_id = filename.replace('.json', '')
            filepath = os.path.join(user_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('messages'):
                        conversations.append({
                            'id': conv_id,
                            'title': data.get('title', 'Percakapan Baru'),
                            'created': data.get('created', ''),
                            'updated': data.get('updated', ''),
                            'message_count': len(data.get('messages', []))
                        })
            except Exception as e:
                print(f"Error loading conversation {filename}: {e}")
                continue
    
    # Sort by updated time (newest first)
    conversations.sort(key=lambda x: x.get('updated', ''), reverse=True)
    return conversations

def load_conversation(user_id, conversation_id):
    """Memuat percakapan tertentu."""
    history_dir = "user_histories"
    filepath = os.path.join(history_dir, user_id, f"{conversation_id}.json")
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return None
    return None

def save_conversation(user_id, conversation_id, title, messages):
    """Menyimpan percakapan."""
    history_dir = "user_histories"
    user_dir = os.path.join(history_dir, user_id)
    
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    filepath = os.path.join(user_dir, f"{conversation_id}.json")
    
    # Load existing data to preserve created time
    existing_data = load_conversation(user_id, conversation_id)
    created_time = existing_data.get('created') if existing_data else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = {
        'id': conversation_id,
        'title': title,
        'created': created_time,
        'updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'messages': messages
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def delete_conversation(user_id, conversation_id):
    """Menghapus percakapan."""
    history_dir = "user_histories"
    filepath = os.path.join(history_dir, user_id, f"{conversation_id}.json")
    
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

def create_new_conversation():
    """Membuat percakapan baru."""
    st.session_state.current_conversation_id = str(uuid.uuid4())
    st.session_state.current_messages = []
    st.session_state.conversation_title = "Percakapan Baru"

def generate_title_from_first_question(question):
    """Generate judul dari pertanyaan pertama."""
    # Ambil maksimal 50 karakter pertama
    title = question.strip()[:25]
    if len(question.strip()) > 50:
        title += "..."
    return title

def initialize_conversation_state(user_id):
    """Inisialisasi state percakapan saat aplikasi dimulai atau di-refresh."""
    if 'current_conversation_id' not in st.session_state:
        # Cek apakah ada percakapan tersimpan
        conversations = get_user_conversations(user_id)
        if conversations and len(conversations) > 0:
            # Load percakapan terakhir
            last_conv = conversations[0]
            conv_data = load_conversation(user_id, last_conv['id'])
            if conv_data:
                st.session_state.current_conversation_id = last_conv['id']
                st.session_state.current_messages = conv_data.get('messages', [])
                st.session_state.conversation_title = conv_data.get('title', 'Percakapan Baru')
            else:
                create_new_conversation()
        else:
            # Tidak ada percakapan, buat baru
            create_new_conversation()

# --- TAHAP 0: KONFIGURASI DAN VALIDASI LINGKUNGAN ---
# def setup_environment():
#     """Memuat .env dan memvalidasi keberadaan kunci API."""
#     load_dotenv()
#     if os.getenv("GOOGLE_API_KEY") is None:
#         st.error("❌ ERROR: Variabel lingkungan GOOGLE_API_KEY tidak ditemukan.")
#         st.error("Pastikan Anda telah membuat file .env dan memasukkan kunci API Anda di sana.")
#         st.stop()

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

        # --- SMART JSON CHUNKING ---
def load_json_database(json_file: str) -> Dict:
    """Load JSON database"""
    with st.spinner(f"📄 Memuat database JSON: {json_file}..."):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            st.error(f"❌ ERROR membaca JSON: {e}")
            st.stop()

def create_smart_chunks(json_data: Dict) -> List[Document]:
    """
    SMART CHUNKING: Satu case = satu chunk utuh
    Jadi context tidak pernah terputus!
    """
    with st.spinner("🧩 Membuat Smart Chunks (per case)..."):
        documents = []
        
        for case in json_data['cases']:
            # Format case menjadi text yang kaya konteks
            chunk_text = f"""ID: {case['id']}
KATEGORI: {case['kategori'].upper().replace('_', ' ')}

DIAGNOSA: {case['diagnosa']}
KODE ICD-10: {', '.join(case['kode_diagnosa']) if case['kode_diagnosa'] else 'Tidak ada'}

PROSEDUR: {case['prosedur'] if case['prosedur'] else 'Tidak ada prosedur khusus'}

ASPEK KODING:
{case['aspek_koding']}

PERHATIAN KHUSUS: {case['perhatian_khusus'] if case['perhatian_khusus'] else 'Tidak ada'}

Keywords: {', '.join(case['keywords'])}"""
            
            # Buat Document dengan metadata kaya
            doc = Document(
                page_content=chunk_text,
                metadata={
                    'id': case['id'],
                    'diagnosa': case['diagnosa'],
                    'kode': case['kode_diagnosa'],
                    'kategori': case['kategori'],
                    'keywords': case['keywords']
                }
            )
            
            documents.append(doc)
        
        return documents

# --- EMBEDDING & VECTOR STORE ---
@st.cache_resource
def create_vector_store(_documents: List[Document]):
    """Buat vector store dari documents"""
    with st.spinner("🔍 Membuat Vector Store dengan embeddings..."):
        try:
            embedding_model = SentenceTransformerEmbeddings(
                model_name="paraphrase-multilingual-mpnet-base-v2"
            )
            db = FAISS.from_documents(_documents, embedding_model)
            return db
        except Exception as e:
            st.error(f"❌ ERROR membuat vector store: {e}")
            st.stop()

# --- HYBRID SEARCH WITH METADATA FILTER ---
def hybrid_search(db, query: str, json_data: Dict, k: int = 8) -> List[Document]:
    """
    HYBRID SEARCH: Semantic + Metadata Filter
    """
    import re
    
    # Deteksi kode ICD-10 dalam query
    kode_pattern = r'\b[A-Z]\d{2}(?:\.\d+)?\b'
    kode_found = re.findall(kode_pattern, query.upper())
    
    if kode_found:
        # Filter by kode spesifik
        filtered_docs = []
        for case in json_data['cases']:
            if any(kode in case['kode_diagnosa'] for kode in kode_found):
                chunk_text = f"""ID: {case['id']}
DIAGNOSA: {case['diagnosa']}
KODE ICD-10: {', '.join(case['kode_diagnosa'])}
PROSEDUR: {case['prosedur']}
ASPEK KODING: {case['aspek_koding']}
PERHATIAN KHUSUS: {case['perhatian_khusus']}"""
                doc = Document(
                    page_content=chunk_text,
                    metadata={'id': case['id'], 'kode': case['kode_diagnosa']}
                )
                filtered_docs.append(doc)
        
        if filtered_docs:
            return filtered_docs[:k]
    
    # Semantic search biasa
    results = db.similarity_search(query, k=k)
    return results

# --- MULTI-QUERY GENERATION (LEVEL 5) ---
def generate_multi_queries(llm, original_query: str) -> List[str]:
    """Generate variasi query untuk perspective berbeda"""
    multi_query_prompt = f"""Buat 2 variasi pertanyaan dari pertanyaan ini (satu per baris, tanpa numbering):

{original_query}

Variasi:"""
    
    try:
        response = llm.invoke(multi_query_prompt)
        queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        return [original_query] + queries[:2]
    except:
        return [original_query]

# --- RECIPROCAL RANK FUSION (LEVEL 5) ---
def reciprocal_rank_fusion(results_list: List[List[Document]], k: int = 60) -> List[Document]:
    """Gabungkan hasil dari multiple queries dengan RRF scoring"""
    fused_scores = {}
    doc_objects = {}
    
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc.metadata.get('id', doc.page_content[:50])
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_objects[doc_id] = doc
            
            fused_scores[doc_id] += 1 / (k + rank + 1)
    
    # Sort by score
    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return documents in order
    return [doc_objects[doc_id] for doc_id, score in sorted_ids]

# --- ADVANCED RAG CHAIN (LEVEL 5 FULL POWER) ---
def run_advanced_rag(db, json_data: Dict, query: str) -> str:
    """
    LEVEL 5 RAG: Multi-Query + Fusion + Hybrid Search + Smart Chunking
    """
    with st.spinner("🤖 AI Level 5: Multi-Query Fusion + Smart Chunking..."):
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.1,
                convert_system_message_to_human=True
            )
            
            # STEP 1: Generate multiple queries
            queries = generate_multi_queries(llm, query)
            
            # STEP 2: Search dengan setiap query
            all_results = []
            for q in queries:
                results = hybrid_search(db, q, json_data, k=6)
                all_results.append(results)
            
            # STEP 3: Reciprocal Rank Fusion
            fused_docs = reciprocal_rank_fusion(all_results)
            
            # STEP 4: Ambil top 5 hasil terbaik
            top_docs = fused_docs[:5]
            
            # STEP 5: Build context
            context = "\n\n---\n\n".join([doc.page_content for doc in top_docs])
            
            # STEP 6: Generate answer dengan prompt fokus
            prompt = f"""Kamu adalah asisten ahli koding ICD-10 dan INA-CBG.

DATABASE STRUCTURE:
Setiap case punya: Diagnosa, Kode ICD-10, Prosedur, Aspek Koding, Perhatian Khusus

KONTEKS (sudah di-filter dengan Multi-Query Fusion + Smart Chunking):
{context}

PERTANYAAN: {query}

ATURAN JAWAB:
1. Analisis pertanyaan - apa yang ditanya?
2. Cari info SPESIFIK dari konteks
3. Jawab LANGSUNG dan FOKUS
4. Sebutkan kode ICD-10 jika relevan
5. Rujuk "Aspek Koding" untuk detail teknis
6. Sebutkan "Perhatian Khusus" jika ada
7. Jika tidak ada info: "Tidak ditemukan dalam database"
8. Gunakan struktur jelas (bullet/numbering) jika perlu

Jawaban (ringkas & akurat):"""
            
            response = llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            st.error(f"❌ ERROR: {e}")
            return f"Maaf, terjadi kesalahan: {str(e)}"
        
        # --- MAIN STREAMLIT APP ---
def main():
    # Hide Streamlit menu and footer - UI/UX LENGKAP DARI KODE LAMA
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
    
    /* Center content container for better readability */
    [data-testid="stAppViewContainer"] > section > div {
        max-width: 50rem !important;
        margin: 0 auto !important;
    }
    
    /* Override Streamlit default background to pure dark like ChatGPT */
    .stApp {
        background-color: #212121 !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #212121 !important;
    }
    
    [data-testid="stHeader"] {
        background-color: #212121 !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #fff !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #fff !important;
    }
    
    /* Main content background */
    .main .block-container {
        background-color: #212121 !important;
    }
    
    .main {
        background-color: #212121 !important;
    }
    
    /* Chat input container background */
    [data-testid="stChatInputContainer"] {
        background-color: #212121 !important;
    }
    
    [data-testid="stChatInputContainer"] > div {
        background-color: #212121 !important;
    }
    
    /* Chat input box itself */
    [data-testid="stChatInput"] {
        background-color: #212121 !important;
    }
    
    [data-testid="stChatInput"] > div {
        background-color: #212121 !important;
    }
    
    [data-testid="stChatInput"] > div > div {
        background-color: #212121 !important;
    }
    
    /* Text input field */
    [data-testid="stChatInput"] textarea {
        background-color: #0e1117 !important;
        color: #ffffff !important;
        border-color: #0e1117 !important;
    }
    
    /* Chat input container */
    [data-testid="stChatInput"] > div {
        background-color: #0e1117 !important;
    }
    
    /* All nested divs in chat input */
    [data-testid="stChatInput"] div {
        background-color: #0e1117 !important;
    }
    
    /* Base input wrapper */
    [data-testid="stChatInput"] [data-baseweb="base-input"] {
        background-color: #0e1117 !important;
    }
    
    /* Input root */
    [data-testid="stChatInput"] [data-baseweb="input"] {
        background-color: #0e1117 !important;
    }
    
    /* Placeholder text color */
    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Submit button (send button) - more specific selector */
    [data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"],
    [data-testid="stChatInput"] button,
    [data-testid="stChatInput"] button[kind="primary"],
    [data-testid="stChatInput"] button[type="submit"] {
        background-color: #ff4b4b !important;
        background: #ff4b4b !important;
        color: #ffffff !important;
        border-color: #ff4b4b !important;
    }
    
    /* Submit button hover */
    [data-testid="stChatInput"] button:hover,
    [data-testid="stChatInput"] button[type="submit"]:hover {
        background-color: #e03e3e !important;
        background: #e03e3e !important;
        border-color: #e03e3e !important;
    }
    
    /* Bottom section background */
    .stBottom {
        background-color: #212121 !important;
    }
    
    div[data-testid="stBottom"] {
        background-color: #212121 !important;
    }
    
    div[data-testid="stBottom"] > div {
        background-color: #212121 !important;
    }
    
    /* All possible containers */
    section[data-testid="stMain"] {
        background-color: #212121 !important;
    }
    
    div[data-testid="stVerticalBlock"] {
        background-color: #212121 !important;
    }
    
    div[data-testid="column"] {
        background-color: #212121 !important;
    }
    
    /* Force all elements to use #212121 background */
    * {
        background-color: inherit;
    }
    
    html, body {
        background-color: #212121 !important;
    }
    
    /* Root elements */
    #root, [data-testid="stApp"] {
        background-color: #212121 !important;
    }
    
    /* Specific containers that need #212121 - NO WILDCARD */
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"],
    section[data-testid="stMain"],
    .main,
    .block-container,
    [data-testid="stVerticalBlock"],
    [data-testid="column"] {
        background-color: #212121 !important;
    }
    
    /* Exception for Streamlit buttons - restore original background */
    button[kind="primary"],
    button[kind="secondary"],
    .stButton > button {
        background-color: initial !important;
        background: initial !important;
    }
    
    /* Chat input styling - force blue color */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] *:not(button) {
        background-color: #0e1117 !important;
        background: #303030 !important;
    }
    
    /* Submit button (send button) - MUST BE AFTER wildcard to override */
    [data-testid="stChatInput"] button {
        background-color: #ff4b4b !important;
        background: #ff4b4b !important;
        color: #ffffff !important;
        border-color: #ff4b4b !important;
    }
    
    /* Submit button hover */
    [data-testid="stChatInput"] button:hover {
        background-color: #e03e3e !important;
        background: #e03e3e !important;
        border-color: #e03e3e !important;
    }
    
    /* Override any black backgrounds */
    *[style*="background-color: rgb(0, 0, 0)"],
    *[style*="background-color: #000"],
    *[style*="background-color: black"],
    *[style*="background: rgb(0, 0, 0)"],
    *[style*="background: #000"],
    *[style*="background: black"] {
        background-color: #212121 !important;
        background: #212121 !important;
    }
    
    body {
        background-color: #212121 !important;
    }
    
    /* Streamlit bottom bar area */
    [class*="stBottom"] {
        background-color: #212121 !important;
    }
    
    /* Chat input parent containers */
    div[data-baseweb="base-input"] {
        background-color: #212121 !important;
    }
    
    /* All divs in bottom area */
    .stBottom > * {
        background-color: #212121 !important;
    }
    
    /* Area behind chat input */
    .stChatFloatingInputContainer {
        background-color: #212121 !important;
    }
    
    div[class*="stChatFloating"] {
        background-color: #212121 !important;
    }
    
    /* Bottom floating area */
    div[data-testid="InputInstructions"] {
        background-color: #212121 !important;
    }
    
    /* Streamlit bottom container */
    section.main > div {
        background-color: #212121 !important;
    }
    
    /* All elements behind input */
    [data-testid="stChatInput"]::before,
    [data-testid="stChatInput"]::after {
        background-color: #212121 !important;
    }
    
    /* Parent of chat input */
    [data-testid="stChatInput"]:parent {
        background-color: #212121 !important;
    }
    
    /* Custom styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 1rem;
    }
    
    /* User message styling */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background-color: #2b5278 !important;
        color: white !important;
    }
    
    /* Assistant message styling */  
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background-color: #f7f7f8 !important;
        color: #000000 !important;
    }
    
    /* Professional conversation list styling */
    .conversation-item {
        transition: all 0.2s ease;
        border-radius: 8px;
        margin-bottom: 4px;
    }
    
    .conversation-item:hover {
        background-color: rgba(255, 75, 75, 0.05);
        transform: translateX(2px);
    }
    
    /* Reduce spacing between items */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.3rem !important;
    }
    
    /* Custom button styling for conversations */
    div[data-testid="column"] > div > div > button {
        text-align: left !important;
        padding: 0.5rem 0.75rem !important;
        font-size: 0.875rem !important;
        border: none !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        height: 38px !important;
        line-height: 1.2 !important;
        display: flex !important;
        align-items: center !important;
    }
    
    /* Active conversation styling */
    div[data-testid="column"] > div > div > button[disabled] {
        background-color: rgba(255, 75, 75, 0.1) !important;
        border-left: 3px solid #ff4b4b !important;
        font-weight: 500 !important;
    }
    
    /* Delete button styling - smaller */
    div[data-testid="column"]:last-child button {
        padding: 0.4rem 0.5rem !important;
        font-size: 0.75rem !important;
        height: 38px !important;
        min-width: 38px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Adjust chat input container position - less margin */
    .stChatInputContainer {
        margin-bottom: 0rem !important;
    }

    /* Sidebar background - all elements inside */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] * {
        background-color: #181818 !important;
    }

    /* Exception for buttons - keep original color */
    section[data-testid="stSidebar"] button[kind="primary"],
    section[data-testid="stSidebar"] button[kind="secondary"],
    section[data-testid="stSidebar"] .stButton > button {
        background-color: initial !important;
        background: initial !important;
    }
    
    /* Mobile responsive styling for chat bubbles */
    @media (max-width: 768px) {
        /* User message - compact like ChatGPT mobile */
        .user-message-container {
            padding: 0 0.75rem !important;
        }
        .user-message-container > div {
            max-width: 80% !important;
            padding: 0.75rem 1rem !important;
            font-size: 1rem !important;
            line-height: 1.5 !important;
        }
        
        /* Assistant message - almost full width like ChatGPT mobile */
        .assistant-message-container {
            padding: 0 0.75rem !important;
        }
        .assistant-message-container > div {
            max-width: 100% !important;
            width: 100% !important;
            padding: 0.75rem 0.5rem !important;
            font-size: 1.0625rem !important;
            line-height: 1.7 !important;
        }
    }
    

    /* MOBILE FIX - Horizontal layout paksa */
    # section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"],
    # section[data-testid="stSidebar"] .row-widget,
    # section[data-testid="stSidebar"] .stHorizontalBlock {
    #     display: flex !important;
    #     flex-direction: row !important;
    #     gap: 0.5rem !important;
    #     align-items: center !important;
    # }
    
    /* First column expand - WIDTH LEBIH BESAR */
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:nth-child(1),
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(1) {
        flex: 1 !important;
        min-width: 0 !important;
        width: calc(100% - 3rem) !important;
    }
    
    /* Second column fixed - LEBIH KECIL */
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:nth-child(2),
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) {
        width: 2.5rem !important;
        flex: 0 0 2.5rem !important;
        min-width: 2.0rem !important;
        max-width: 2.0rem !important;
    }
    
    /* BUTTON - Pastikan text horizontal */
    # section[data-testid="stSidebar"] .stButton button {
    #     height: auto !important;
    #     min-height: 2.5rem !important;
    #     padding: 0.5rem 0.75rem !important;
    #     font-size: 0.9rem !important;
    #     line-height: 1.3 !important;
    #     width: 100% !important;
    #     word-break: keep-all !important;
    # }
    
    /* Mobile specific */
    @media screen and (max-width: 768px) {
        section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
            flex-direction: row !important;
            flex-wrap: nowrap !important;
        }
    }
    

    /* Desktop styling - ChatGPT like */
    @media (min-width: 769px) {
        /* Center chat container with max-width for readability */
        .block-container {
            max-width: 46rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        /* User message - small bubble right side */
        .user-message-container {
            padding: 0 0 !important;
            justify-content: flex-end !important;
        }
        .user-message-container > div {
            max-width: 70% !important;
            background-color: #2f2f2f !important;
            padding: 0.8rem 1.2rem !important;
            border-radius: 1.25rem !important;
        }
        
        /* Assistant message - full width within container */
        .assistant-message-container {
            padding: 0 0 !important;
            justify-content: flex-start !important;
        }
        .assistant-message-container > div {
            max-width: 100% !important;
            width: 100% !important;
            background-color: transparent !important;
            padding: 1rem 0 !important;
            border-radius: 0 !important;
        }
    }

    /* Tombol > sidebar - background biru */
    [data-testid="collapsedControl"] {
        background-color: #3b82f6 !important;
        border-radius: 0.6rem !important;
    }

    [data-testid="collapsedControl"]:hover {
        background-color: #2563eb !important;
    }
    
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Setup environment
    setup_environment()

    # Get or create unique user ID
    user_id = get_or_create_user_id()

    # Initialize conversation state
    initialize_conversation_state(user_id)

    # Sidebar untuk riwayat percakapan - UI LENGKAP DARI KODE LAMA
    with st.sidebar:
        # Header dengan styling
        st.markdown("### 💬 Riwayat Chat")
        
        # Tombol percakapan baru
        if st.button("➕ Percakapan Baru", use_container_width=True, type="primary"):
            create_new_conversation()
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Daftar percakapan
        conversations = get_user_conversations(user_id)
        
        if conversations:
            # Header untuk list
            col_h1, col_h2 = st.columns([5, 2])
            with col_h1:
                st.markdown(f"<small style='color: #666;'>📋 {len(conversations)} histori tersimpan</small>", unsafe_allow_html=True)
            
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            
            # Scrollable conversation list
            for idx, conv in enumerate(conversations):
                col1, col2 = st.columns([7, 1])
                
                with col1:
                    # Tombol untuk membuka percakapan
                    is_active = conv['id'] == st.session_state.current_conversation_id
                    
                    # Tampilkan judul (CSS akan handle truncation)
                    display_title = conv['title']
                    
                    if st.button(
                        display_title,
                        key=f"conv_{conv['id']}",
                        use_container_width=True,
                        disabled=is_active,
                        type="primary" if is_active else "secondary"
                    ):
                        # Load conversation
                        conv_data = load_conversation(user_id, conv['id'])
                        if conv_data:
                            st.session_state.current_conversation_id = conv['id']
                            st.session_state.current_messages = conv_data.get('messages', [])
                            st.session_state.conversation_title = conv_data.get('title', 'Percakapan Baru')
                            st.rerun()
                
                with col2:
                    # Tombol hapus - icon lebih kecil
                    if st.button("🗑", key=f"del_{conv['id']}", help="Hapus"):
                        if delete_conversation(user_id, conv['id']):
                            if conv['id'] == st.session_state.current_conversation_id:
                                remaining_convs = get_user_conversations(user_id)
                                if remaining_convs:
                                    first_conv = remaining_convs[0]
                                    conv_data = load_conversation(user_id, first_conv['id'])
                                    if conv_data:
                                        st.session_state.current_conversation_id = first_conv['id']
                                        st.session_state.current_messages = conv_data.get('messages', [])
                                        st.session_state.conversation_title = conv_data.get('title', 'Percakapan Baru')
                                else:
                                    create_new_conversation()
                            st.rerun()
                
                # Mini spacing between items
                # if idx < len(conversations) - 1:
                #     st.markdown("<div style='height: 0px;'></div>", unsafe_allow_html=True)
        
        else:
            # Empty state
            st.markdown("""
            <div style='text-align: center; padding: 2rem 1rem; color: #666;'>
                <p style='font-size: 2rem; margin: 0;'>💭</p>
                <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Belum ada percakapan</p>
                <p style='font-size: 0.8rem; color: #999;'>Mulai chat untuk membuat percakapan baru</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Divider dengan styling
        st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
        
        # Session info dan tips
        st.markdown(f"<small style='color: #999;'>🔐 Session: <code>{user_id[:12]}...</code></small>", unsafe_allow_html=True)
        
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        
        # Tips dengan expander yang lebih menarik
        with st.expander("💡 Tips Penggunaan"):
            st.markdown("""
            **Cara Kerja Persistensi:**
            
            - 🔖 Histori tersimpan di komputer ini
            - 🔄 Refresh halaman = histori tetap ada
            - 💻 Ganti komputer = histori berbeda
            - 🌐 Browser berbeda di komputer sama = histori SAMA
            
            **Fitur:**
            
            - ✅ Histori otomatis tersimpan
            - ✅ Percakapan tidak hilang saat refresh
            - ✅ Bisa buat banyak percakapan
            - ✅ Level 5 RAG: Multi-Query Fusion
            - ✅ Smart Chunking: Context tidak terputus
            """)
        
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        
        # Reset button dengan warning style
        if st.button("🔄 Reset Session", use_container_width=True, help="Mulai sebagai user baru (hapus semua histori)"):
            reset_user_session()
            st.rerun()

    # Main content area
    st.markdown("""
    <h1 style='font-size: 1.8rem; font-weight: 700; margin-top: -4rem; margin-bottom: 0.5rem;'>
        📄 Analisis Panduan Manual Verifikasi Klaim INA-CBG Edisi 2 Dengan AI
    </h1>
    """, unsafe_allow_html=True)

    # Header dengan judul percakapan
    col_title, col_info = st.columns([3, 1])
    with col_title:
        st.markdown(f"### 💬 {st.session_state.conversation_title}")
    with col_info:
        st.caption(f"📊 {len(st.session_state.current_messages)} pesan")

    st.divider()

    # File configuration - PAKAI JSON SEKARANG!
    json_file = "medical_database_structured.json"

    if not os.path.exists(json_file):
        st.error(f"❌ ERROR: File '{json_file}' tidak ditemukan.")
        st.info("📝 Jalankan dulu 'convert_txt_to_json.py' untuk convert database Anda ke JSON.")
        st.stop()

    # Load and process JSON database
    try:
        # Load & process JSON dengan SMART CHUNKING
        with st.spinner("⚙️ Memproses database JSON..."):
            json_data = load_json_database(json_file)
            documents = create_smart_chunks(json_data)
            vector_db = create_vector_store(documents)

        # st.success(f"✅ Database loaded: {json_data['metadata']['total_cases']} cases dari {len(json_data['metadata']['categories'])} kategori")
        st.markdown(f"""
        <div style="background-color: #1e3a5f; color: #fff; padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            ✅ Database loaded: {json_data['metadata']['total_cases']} cases dari {len(json_data['metadata']['categories'])} kategori
        </div>
        """, unsafe_allow_html=True)

        # Display chat messages - ChatGPT style layout
        if len(st.session_state.current_messages) == 0:
            st.markdown("""
            <p style="color: #fff; background: transparent; padding: 0.75rem 1rem; margin: 1rem 0;">
                👋 Selamat datang! Database siap dianalisis. Tanya tentang diagnosa, kode ICD-10, atau aspek koding.
            </p>
            """, unsafe_allow_html=True)
        
        for msg in st.session_state.current_messages:
            # Escape HTML characters properly to prevent rendering issues
            import html
            question_text = html.escape(msg["question"]).replace("\n", "<br>")
            answer_text = html.escape(msg["answer"]).replace("\n", "<br>")
            
            # User message - RIGHT aligned compact like ChatGPT
            st.markdown(f"""
            <div class="user-message-container" style="display: flex; justify-content: flex-end; margin-bottom: 1.5rem; padding: 0 1rem;">
                <div style="background-color: #2f2f2f; color: #ececec; padding: 0.75rem 1rem; border-radius: 1.25rem; max-width: 70%; text-align: left; word-wrap: break-word; font-size: 1rem; line-height: 1.6;">
                    {question_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant message - LEFT aligned full width like ChatGPT
            st.markdown(f"""
            <div class="assistant-message-container" style="display: flex; justify-content: flex-start; margin-bottom: 2rem; padding: 0 1rem;">
                <div style="background-color: transparent; color: #ececec; padding: 0.75rem 0.5rem; border-radius: 0; max-width: 100%; width: 100%; text-align: left; word-wrap: break-word; line-height: 1.75; font-size: 1.0625rem; font-weight: 400;">
                    {answer_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Chat input
        pertanyaan_user = st.chat_input("💭 Ajukan pertanyaan Anda ...")
        
        # Add powered by link
        st.markdown("""
        <style>
        div[style*="position: fixed"][style*="bottom: 0"] {
            background-color: #212121 !important;
            background: #212121 !important;
        }
        
        div[style*="position: fixed"] a {
            pointer-events: auto !important;
        }
        div[style*="position: fixed"] a:hover {
            color: #93c5fd !important;
            text-decoration: underline !important;
        }
        
        [data-testid="stSidebar"] {
            z-index: 9999 !important;
        }
        </style>
        <div style="position: fixed; bottom: 0; left: 0; right: 0; text-align: center; padding: 0.7rem; background: rgba(18,18,18,0.95); z-index: 999; border-top: 1px solid rgba(255,255,255,0.1);">
            <p style="font-size: 0.85rem; color: #e0e0e0; margin: 0; font-weight: 400;">
                Powered by: <a href="https://rekam-medis.id" target="_blank" style="color: #60a5fa; text-decoration: none; font-weight: 600; cursor: pointer; transition: color 0.2s;">https://rekam-medis.id</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

        if pertanyaan_user:
            if pertanyaan_user.strip():
                # Properly escape HTML characters using html module
                import html
                question_escaped = html.escape(pertanyaan_user).replace("\n", "<br>")
                
                # Display user message immediately - RIGHT side like ChatGPT
                st.markdown(f"""
                <div class="user-message-container" style="display: flex; justify-content: flex-end; margin-bottom: 1.5rem; padding: 0 1rem;">
                    <div style="background-color: #2f2f2f; color: #ececec; padding: 0.75rem 1rem; border-radius: 1.25rem; max-width: 70%; text-align: left; word-wrap: break-word; font-size: 1rem; line-height: 1.6;">
                        {question_escaped}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Run ADVANCED RAG (LEVEL 5)
                final_answer = run_advanced_rag(vector_db, json_data, pertanyaan_user)
                
                # Properly escape HTML characters in answer
                answer_escaped = html.escape(final_answer).replace("\n", "<br>")

                # Display assistant message - LEFT side full width like ChatGPT
                st.markdown(f"""
                <div class="assistant-message-container" style="display: flex; justify-content: flex-start; margin-bottom: 2rem; padding: 0 1rem;">
                    <div style="background-color: transparent; color: #ececec; padding: 0.75rem 0.5rem; border-radius: 0; max-width: 100%; width: 100%; text-align: left; word-wrap: break-word; line-height: 1.75; font-size: 1.0625rem; font-weight: 400;">
                        {answer_escaped}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Add to messages (without timestamp display)
                chat_entry = {
                    "question": pertanyaan_user,
                    "answer": final_answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.current_messages.append(chat_entry)

                # Update title if first message
                if len(st.session_state.current_messages) == 1:
                    st.session_state.conversation_title = generate_title_from_first_question(pertanyaan_user)

                # Save conversation
                save_conversation(
                    user_id,
                    st.session_state.current_conversation_id,
                    st.session_state.conversation_title,
                    st.session_state.current_messages
                )
                
                st.rerun()
            else:
                st.warning("⚠️ Silakan masukkan pertanyaan terlebih dahulu.")

    except Exception as e:
        st.error(f"❌ TERJADI KESALAHAN FATAL: {e}")
        st.error("🔧 Periksa kembali instalasi pustaka, koneksi internet, dan API Key Anda.")
        
        with st.expander("🐛 Detail Error untuk Debugging"):
            st.code(str(e))

if __name__ == "__main__":
    main()
