import streamlit as st
import os
import json
import uuid
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict
import re
import html

# --- KONFIGURASI HALAMAN (STYLE KODE 1) ---
st.set_page_config(
    page_title="Analisis Dokumen Medis AI (JSON)",
    page_icon="📄",
    layout="wide",
    menu_items=None
)

# ==============================================================================
# BAGIAN 1: LOGIKA & OTAK PROGRAM (MURNI DARI KODE 2)
# ==============================================================================

# Pustaka LangChain & Komponen AI
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# --- USER SESSION MANAGEMENT (SAMA UTK KEDUA KODE) ---
def get_or_create_user_id():
    if 'user_id' not in st.session_state:
        user_id = load_user_id_from_file()
        if not user_id:
            user_id = str(uuid.uuid4())
            save_user_id_to_file(user_id)
        st.session_state.user_id = user_id
        st.session_state.session_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return st.session_state.user_id

def load_user_id_from_file():
    user_id_file = ".streamlit_user_id"
    if os.path.exists(user_id_file):
        try:
            with open(user_id_file, 'r', encoding='utf-8') as f:
                user_id = f.read().strip()
                if user_id: return user_id
        except: pass
    return None

def save_user_id_to_file(user_id):
    try:
        with open(".streamlit_user_id", 'w', encoding='utf-8') as f:
            f.write(user_id)
        return True
    except: return False

def reset_user_session():
    try: os.remove(".streamlit_user_id")
    except: pass
    keys = ['user_id', 'current_conversation_id', 'current_messages', 'conversation_title']
    for k in keys:
        if k in st.session_state: del st.session_state[k]

def get_user_conversations(user_id):
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
                            'updated': data.get('updated', ''),
                        })
            except: continue
    conversations.sort(key=lambda x: x.get('updated', ''), reverse=True)
    return conversations

def load_conversation(user_id, conversation_id):
    filepath = os.path.join("user_histories", user_id, f"{conversation_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
        except: return None
    return None

def save_conversation(user_id, conversation_id, title, messages):
    history_dir = "user_histories"
    user_dir = os.path.join(history_dir, user_id)
    if not os.path.exists(user_dir): os.makedirs(user_dir)
    filepath = os.path.join(user_dir, f"{conversation_id}.json")
    existing = load_conversation(user_id, conversation_id)
    created = existing.get('created') if existing else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        'id': conversation_id,
        'title': title,
        'created': created,
        'updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'messages': messages
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def delete_conversation(user_id, conversation_id):
    filepath = os.path.join("user_histories", user_id, f"{conversation_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

def create_new_conversation():
    st.session_state.current_conversation_id = str(uuid.uuid4())
    st.session_state.current_messages = []
    st.session_state.conversation_title = "Percakapan Baru"

def generate_title_from_first_question(question):
    title = question.strip()[:20]
    if len(question.strip()) > 50: title += "..."
    return title

def initialize_conversation_state(user_id):
    if 'current_conversation_id' not in st.session_state:
        conversations = get_user_conversations(user_id)
        if conversations:
            last_conv = conversations[0]
            conv_data = load_conversation(user_id, last_conv['id'])
            if conv_data:
                st.session_state.current_conversation_id = last_conv['id']
                st.session_state.current_messages = conv_data.get('messages', [])
                st.session_state.conversation_title = conv_data.get('title', 'Percakapan Baru')
            else: create_new_conversation()
        else: create_new_conversation()

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

def load_json_database(json_file: str) -> Dict:
    with st.spinner(f"📄 Memuat database JSON: {json_file}..."):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"❌ ERROR membaca JSON: {e}")
            st.stop()

def create_smart_chunks(json_data: Dict) -> List[Document]:
    with st.spinner("🧩 Membuat Smart Chunks (per case)..."):
        documents = []
        for case in json_data['cases']:
            chunk_text = f"""ID: {case['id']}
KATEGORI: {case['kategori'].upper().replace('_', ' ')}
DIAGNOSA: {case['diagnosa']}
KODE ICD-10: {', '.join(case['kode_diagnosa']) if case['kode_diagnosa'] else 'Tidak ada'}
PROSEDUR: {case['prosedur'] if case['prosedur'] else 'Tidak ada prosedur khusus'}
ASPEK KODING:
{case['aspek_koding']}
PERHATIAN KHUSUS: {case['perhatian_khusus'] if case['perhatian_khusus'] else 'Tidak ada'}
Keywords: {', '.join(case['keywords'])}"""
            doc = Document(
                page_content=chunk_text,
                metadata={
                    'id': case['id'], 'diagnosa': case['diagnosa'],
                    'kode': case['kode_diagnosa'], 'kategori': case['kategori'],
                    'keywords': case['keywords']
                }
            )
            documents.append(doc)
        return documents

@st.cache_resource
def create_vector_store(_documents: List[Document]):
    with st.spinner("🔍 Membuat Vector Store dengan embeddings..."):
        try:
            embedding_model = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
            return FAISS.from_documents(_documents, embedding_model)
        except Exception as e:
            st.error(f"❌ ERROR: {e}"); st.stop()

def hybrid_search(db, query: str, json_data: Dict, k: int = 8) -> List[Document]:
    kode_pattern = r'\b[A-Z]\d{2}(?:\.\d+)?\b'
    kode_found = re.findall(kode_pattern, query.upper())
    if kode_found:
        filtered_docs = []
        for case in json_data['cases']:
            if any(kode in case['kode_diagnosa'] for kode in kode_found):
                chunk_text = f"ID: {case['id']}\nDIAGNOSA: {case['diagnosa']}\nKODE: {', '.join(case['kode_diagnosa'])}\nPROSEDUR: {case['prosedur']}\nASPEK KODING: {case['aspek_koding']}"
                doc = Document(page_content=chunk_text, metadata={'id': case['id'], 'kode': case['kode_diagnosa']})
                filtered_docs.append(doc)
        if filtered_docs: return filtered_docs[:k]
    return db.similarity_search(query, k=k)

def generate_multi_queries(llm, original_query: str) -> List[str]:
    try:
        response = llm.invoke(f"Buat 2 variasi pertanyaan dari: {original_query}")
        queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        return [original_query] + queries[:2]
    except: return [original_query]

def reciprocal_rank_fusion(results_list: List[List[Document]], k: int = 60) -> List[Document]:
    fused_scores = {}
    doc_objects = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc.metadata.get('id', doc.page_content[:50])
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_objects[doc_id] = doc
            fused_scores[doc_id] += 1 / (k + rank + 1)
    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_objects[doc_id] for doc_id, score in sorted_ids]

def run_advanced_rag(db, json_data: Dict, query: str) -> str:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        queries = generate_multi_queries(llm, query)
        all_results = []
        for q in queries:
            all_results.append(hybrid_search(db, q, json_data, k=6))
        fused_docs = reciprocal_rank_fusion(all_results)
        context = "\n\n---\n\n".join([doc.page_content for doc in fused_docs[:5]])
        prompt = f"Kamu asisten ICD-10. DATA: {context}\nPERTANYAAN: {query}\nJAWABAN RINGKAS:"
        response = llm.invoke(prompt)
        return response.content
    except Exception as e: return f"Error: {str(e)}"

# ==============================================================================
# BAGIAN 2: TAMPILAN VISUAL (MURNI DARI KODE 1)
# ==============================================================================

def main():
    # --- CSS STYLE MURNI DARI KODE 1 (BARIS 124 - 186) ---
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
    
    [data-testid="stAppViewContainer"] > section > div {
        max-width: 50rem !important; margin: 0 auto !important;
    }
    
    .stApp { background-color: #212121 !important; }
    [data-testid="stAppViewContainer"] { background-color: #212121 !important; }
    [data-testid="stHeader"] { background-color: #212121 !important; }
    
    section[data-testid="stSidebar"] { background-color: #fff !important; }
    section[data-testid="stSidebar"] > div { background-color: #fff !important; }
    
    .main .block-container { background-color: #212121 !important; }
    .main { background-color: #212121 !important; }
    
    [data-testid="stChatInputContainer"] { background-color: #212121 !important; }
    [data-testid="stChatInputContainer"] > div { background-color: #212121 !important; }
    [data-testid="stChatInput"] { background-color: #212121 !important; }
    [data-testid="stChatInput"] > div { background-color: #212121 !important; }
    [data-testid="stChatInput"] > div > div { background-color: #212121 !important; }
    
    [data-testid="stChatInput"] textarea {
        background-color: #0e1117 !important; color: #ffffff !important;
        border-color: #0e1117 !important;
    }
    
    [data-testid="stChatInput"] > div { background-color: #0e1117 !important; }
    [data-testid="stChatInput"] div { background-color: #0e1117 !important; }
    [data-testid="stChatInput"] [data-baseweb="base-input"] { background-color: #0e1117 !important; }
    [data-testid="stChatInput"] [data-baseweb="input"] { background-color: #0e1117 !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: rgba(255, 255, 255, 0.7) !important; }
    
    [data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"],
    [data-testid="stChatInput"] button,
    [data-testid="stChatInput"] button[kind="primary"],
    [data-testid="stChatInput"] button[type="submit"] {
        background-color: #ff4b4b !important; background: #ff4b4b !important;
        color: #ffffff !important; border-color: #ff4b4b !important;
    }
    
    [data-testid="stChatInput"] button:hover,
    [data-testid="stChatInput"] button[type="submit"]:hover {
        background-color: #e03e3e !important; background: #e03e3e !important;
        border-color: #e03e3e !important;
    }
    
    .stBottom { background-color: #212121 !important; }
    div[data-testid="stBottom"] { background-color: #212121 !important; }
    div[data-testid="stBottom"] > div { background-color: #212121 !important; }
    section[data-testid="stMain"] { background-color: #212121 !important; }
    div[data-testid="stVerticalBlock"] { background-color: #212121 !important; }
    div[data-testid="column"] { background-color: #212121 !important; }
    * { background-color: inherit; }
    html, body { background-color: #212121 !important; }
    #root, [data-testid="stApp"] { background-color: #212121 !important; }
    
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"],
    section[data-testid="stSidebar"], section[data-testid="stMain"], .main, .block-container,
    [data-testid="stVerticalBlock"], [data-testid="column"] {
        background-color: #212121 !important;
    }
    
    button[kind="primary"], button[kind="secondary"], .stButton > button {
        background-color: initial !important; background: initial !important;
    }
    
    [data-testid="stChatInput"], [data-testid="stChatInput"] *:not(button) {
        background-color: #0e1117 !important; background: #303030 !important;
    }
    
    [data-testid="stChatInput"] button {
        background-color: #ff4b4b !important; background: #ff4b4b !important;
        color: #ffffff !important; border-color: #ff4b4b !important;
    }
    [data-testid="stChatInput"] button:hover {
        background-color: #e03e3e !important; background: #e03e3e !important;
        border-color: #e03e3e !important;
    }
    
    *[style*="background-color: rgb(0, 0, 0)"], *[style*="background-color: #000"],
    *[style*="background-color: black"], *[style*="background: rgb(0, 0, 0)"],
    *[style*="background: #000"], *[style*="background: black"] {
        background-color: #212121 !important; background: #212121 !important;
    }
    
    body { background-color: #212121 !important; }
    [class*="stBottom"] { background-color: #212121 !important; }
    div[data-baseweb="base-input"] { background-color: #212121 !important; }
    .stBottom > * { background-color: #212121 !important; }
    .stChatFloatingInputContainer { background-color: #212121 !important; }
    div[class*="stChatFloating"] { background-color: #212121 !important; }
    div[data-testid="InputInstructions"] { background-color: #212121 !important; }
    section.main > div { background-color: #212121 !important; }
    [data-testid="stChatInput"]::before, [data-testid="stChatInput"]::after { background-color: #212121 !important; }
    [data-testid="stChatInput"]:parent { background-color: #212121 !important; }
    
    .stChatMessage { padding: 1rem; border-radius: 1rem; }
    [data-testid="stChatMessage"][data-testid*="user"] { background-color: #2b5278 !important; color: white !important; }
    [data-testid="stChatMessage"][data-testid*="assistant"] { background-color: #f7f7f8 !important; color: #000000 !important; }
    
    .conversation-item { transition: all 0.2s ease; border-radius: 8px; margin-bottom: 4px; }
    .conversation-item:hover { background-color: rgba(255, 75, 75, 0.05); transform: translateX(2px); }
    div[data-testid="stVerticalBlock"] > div { gap: 0.2rem !important; }
    
    div[data-testid="column"] > div > div > button {
        text-align: left !important; padding: 0.5rem 0.75rem !important;
        font-size: 0.875rem !important; border: none !important; box-shadow: none !important;
        transition: all 0.2s ease !important; white-space: nowrap !important;
        overflow: hidden !important; text-overflow: ellipsis !important;
        height: 38px !important; line-height: 1.2 !important; display: flex !important; align-items: center !important;
    }
    
    div[data-testid="column"] > div > div > button[disabled] {
        background-color: rgba(255, 75, 75, 0.1) !important;
        border-left: 3px solid #ff4b4b !important; font-weight: 500 !important;
    }
    
    div[data-testid="column"]:last-child button {
        padding: 0.4rem 0.5rem !important; font-size: 0.75rem !important;
        height: 38px !important; min-width: 38px !important; display: flex !important;
        align-items: center !important; justify-content: center !important;
    }
    
    .stChatInputContainer { margin-bottom: 0rem !important; }
    
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div, section[data-testid="stSidebar"] * {
        background-color: #181818 !important;
    }
    
    section[data-testid="stSidebar"] button[kind="primary"],
    section[data-testid="stSidebar"] button[kind="secondary"],
    section[data-testid="stSidebar"] .stButton > button {
        background-color: initial !important; background: initial !important;
    }
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    setup_environment()
    user_id = get_or_create_user_id()
    initialize_conversation_state(user_id)

    # --- SIDEBAR (VISUAL DARI KODE 1, BARIS 186-209) ---
    with st.sidebar:
        st.markdown("### 💬 Riwayat Chat")
        if st.button("➕ Percakapan Baru", use_container_width=True, type="primary"):
            create_new_conversation(); st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        conversations = get_user_conversations(user_id)
        
        if conversations:
            col_h1, col_h2 = st.columns([5, 2])
            with col_h1: st.markdown(f"<small style='color: #666;'>📋 {len(conversations)} percakapan tersimpan</small>", unsafe_allow_html=True)
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            
            for idx, conv in enumerate(conversations):
                col1, col2 = st.columns([7, 1])
                with col1:
                    is_active = conv['id'] == st.session_state.current_conversation_id
                    display_title = conv['title']
                    if st.button(display_title, key=f"conv_{conv['id']}", use_container_width=True, disabled=is_active, type="primary" if is_active else "secondary"):
                        conv_data = load_conversation(user_id, conv['id'])
                        if conv_data:
                            st.session_state.current_conversation_id = conv['id']
                            st.session_state.current_messages = conv_data.get('messages', [])
                            st.session_state.conversation_title = conv_data.get('title', 'Percakapan Baru')
                            st.rerun()
                with col2:
                    if st.button("🗑", key=f"del_{conv['id']}", help="Hapus"):
                        if delete_conversation(user_id, conv['id']):
                            if conv['id'] == st.session_state.current_conversation_id:
                                remaining = get_user_conversations(user_id)
                                if remaining:
                                    first = remaining[0]
                                    d = load_conversation(user_id, first['id'])
                                    if d:
                                        st.session_state.current_conversation_id = first['id']
                                        st.session_state.current_messages = d.get('messages', [])
                                        st.session_state.conversation_title = d.get('title', '')
                                else: create_new_conversation()
                            st.rerun()
                if idx < len(conversations) - 1: st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)
        else:
            st.markdown("""<div style='text-align: center; padding: 2rem 1rem; color: #666;'>
                <p style='font-size: 2rem; margin: 0;'>💭</p>
                <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Belum ada percakapan</p>
                <p style='font-size: 0.8rem; color: #999;'>Mulai chat untuk membuat percakapan baru</p>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
        st.markdown(f"<small style='color: #999;'>🔐 Session: <code>{user_id[:12]}...</code></small>", unsafe_allow_html=True)
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        with st.expander("💡 Tips Penggunaan"):
            st.markdown("""**Cara Kerja Persistensi:**\n- 🔖 Histori tersimpan di komputer ini\n- 🔄 Refresh halaman = histori tetap ada""")
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset Session", use_container_width=True, help="Mulai sebagai user baru"):
            reset_user_session(); st.rerun()

    # --- MAIN HEADER (VISUAL DARI KODE 1, BARIS 209) ---
    st.markdown("""
    <h1 style='font-size: 1.8rem; font-weight: 700; margin-top: -4rem; margin-bottom: 0.5rem;'>
        📄 Analisis Database Medis AI (Structured JSON)
    </h1>
    """, unsafe_allow_html=True)

    col_title, col_info = st.columns([3, 1])
    with col_title: st.markdown(f"### 💬 {st.session_state.conversation_title}")
    with col_info: st.caption(f"📊 {len(st.session_state.current_messages)} pesan")
    st.divider()

    # --- LOGIKA LOADING DATA (KODE 2) ---
    json_file = "medical_database_structured.json"
    if not os.path.exists(json_file):
        st.error(f"❌ ERROR: File '{json_file}' tidak ditemukan."); st.stop()

    try:
        with st.spinner("⚙️ Memuat database JSON..."):
            json_data = load_json_database(json_file)
            documents = create_smart_chunks(json_data)
            vector_db = create_vector_store(documents)

        # --- LOOP TAMPILAN PESAN (VISUAL DARI KODE 1, BARIS 212-219) ---
        if len(st.session_state.current_messages) == 0:
            st.markdown("""<p style="color: #fff; background: transparent; padding: 0.75rem 1rem; margin: 1rem 0;">👋 Selamat datang! Silakan tanya diagnosa.</p>""", unsafe_allow_html=True)
        
        for msg in st.session_state.current_messages:
            q_text = html.escape(msg["question"]).replace("\n", "<br>")
            a_text = html.escape(msg["answer"]).replace("\n", "<br>")
            
            # User Message (Style Kode 1)
            st.markdown(f"""
            <div class="user-message-container" style="display: flex; justify-content: flex-end; margin-bottom: 1.5rem; padding: 0 1rem;">
                <div style="background-color: #2f2f2f; color: #ececec; padding: 0.75rem 1rem; border-radius: 1.25rem; max-width: 70%; text-align: left; word-wrap: break-word; font-size: 1rem; line-height: 1.6;">
                    {q_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant Message (Style Kode 1)
            st.markdown(f"""
            <div class="assistant-message-container" style="display: flex; justify-content: flex-start; margin-bottom: 2rem; padding: 0 1rem;">
                <div style="background-color: transparent; color: #ececec; padding: 0.75rem 0.5rem; border-radius: 0; max-width: 100%; width: 100%; text-align: left; word-wrap: break-word; line-height: 1.75; font-size: 1.0625rem; font-weight: 400;">
                    {a_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- FOOTER & INPUT STYLE (VISUAL DARI KODE 1, BARIS 220-237) ---
        # Ini bagian penting yang sebelumnya terlewat (media queries untuk bubble chat)
        st.markdown("""
        <style>
        div[style*="position: fixed"][style*="bottom: 0"] { background-color: #212121 !important; background: #212121 !important; }
        div[style*="position: fixed"] a { pointer-events: auto !important; }
        div[style*="position: fixed"] a:hover { color: #93c5fd !important; text-decoration: underline !important; }
        [data-testid="stSidebar"] { z-index: 9999 !important; }
        
        /* Mobile responsive styling for chat bubbles (KODE 1) */
        @media (max-width: 768px) {
            .user-message-container { padding: 0 0.75rem !important; }
            .user-message-container > div { max-width: 80% !important; padding: 0.75rem 1rem !important; font-size: 1rem !important; line-height: 1.5 !important; }
            .assistant-message-container { padding: 0 0.75rem !important; }
            .assistant-message-container > div { max-width: 100% !important; width: 100% !important; padding: 0.75rem 0.5rem !important; font-size: 1.0625rem !important; line-height: 1.7 !important; }
        }
        
        /* Desktop styling (KODE 1) */
        @media (min-width: 769px) {
            .block-container { max-width: 46rem !important; padding-left: 2rem !important; padding-right: 2rem !important; }
            .user-message-container { padding: 0 0 !important; justify-content: flex-end !important; }
            .user-message-container > div { max-width: 70% !important; background-color: #2f2f2f !important; padding: 0.8rem 1.2rem !important; border-radius: 1.25rem !important; }
            .assistant-message-container { padding: 0 0 !important; justify-content: flex-start !important; }
            .assistant-message-container > div { max-width: 100% !important; width: 100% !important; background-color: transparent !important; padding: 1rem 0 !important; border-radius: 0 !important; }
        }
        </style>
        <div style="position: fixed; bottom: 0; left: 0; right: 0; text-align: center; padding: 0.7rem; background: rgba(18,18,18,0.95); z-index: 999; border-top: 1px solid rgba(255,255,255,0.1);">
            <p style="font-size: 0.85rem; color: #e0e0e0; margin: 0; font-weight: 400;">
                Powered by: <a href="https://rekam-medis.id" target="_blank" style="color: #60a5fa; text-decoration: none; font-weight: 600;">https://rekam-medis.id</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- INPUT & EKSEKUSI (LOGIKA KODE 2, TAMPILAN KODE 1) ---
        pertanyaan_user = st.chat_input("💭 Tanya tentang diagnosa...")
        
        if pertanyaan_user and pertanyaan_user.strip():
            # Render User Message (Immediate - Kode 1 Style)
            q_escaped = html.escape(pertanyaan_user).replace("\n", "<br>")
            st.markdown(f"""
            <div class="user-message-container" style="display: flex; justify-content: flex-end; margin-bottom: 1.5rem; padding: 0 1rem;">
                <div style="background-color: #2f2f2f; color: #ececec; padding: 0.75rem 1rem; border-radius: 1.25rem; max-width: 70%; text-align: left; word-wrap: break-word; font-size: 1rem; line-height: 1.6;">
                    {q_escaped}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Run Logic (Kode 2)
            final_answer = run_advanced_rag(vector_db, json_data, pertanyaan_user)
            
            # Render Assistant Message (Kode 1 Style)
            a_escaped = html.escape(final_answer).replace("\n", "<br>")
            st.markdown(f"""
            <div class="assistant-message-container" style="display: flex; justify-content: flex-start; margin-bottom: 2rem; padding: 0 1rem;">
                <div style="background-color: transparent; color: #ececec; padding: 0.75rem 0.5rem; border-radius: 0; max-width: 100%; width: 100%; text-align: left; word-wrap: break-word; line-height: 1.75; font-size: 1.0625rem; font-weight: 400;">
                    {a_escaped}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Save History
            st.session_state.current_messages.append({
                "question": pertanyaan_user, "answer": final_answer, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            if len(st.session_state.current_messages) == 1:
                st.session_state.conversation_title = generate_title_from_first_question(pertanyaan_user)
            save_conversation(user_id, st.session_state.current_conversation_id, st.session_state.conversation_title, st.session_state.current_messages)
            st.rerun()

    except Exception as e:
        st.error(f"❌ ERROR: {e}")
        with st.expander("🐛 Detail Error"): st.code(str(e))

if __name__ == "__main__":
    main()
