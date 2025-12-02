import streamlit as st
import os
import json
import uuid
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict

# Configure page
st.set_page_config(
    page_title="Analisis Dokumen AI dengan RAG",
    page_icon="📄",
    layout="wide",
    menu_items=None
)

# Pustaka LangChain & Komponen AI
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# --- USER SESSION MANAGEMENT (SAMA SEPERTI SEBELUMNYA) ---
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
                if user_id:
                    return user_id
        except Exception as e:
            print(f"Error loading user ID: {e}")
    return None

def save_user_id_to_file(user_id):
    user_id_file = ".streamlit_user_id"
    try:
        with open(user_id_file, 'w', encoding='utf-8') as f:
            f.write(user_id)
        return True
    except Exception as e:
        print(f"Error saving user ID: {e}")
        return False

def reset_user_session():
    user_id_file = ".streamlit_user_id"
    if os.path.exists(user_id_file):
        try:
            os.remove(user_id_file)
        except:
            pass
    if 'user_id' in st.session_state:
        del st.session_state.user_id
    if 'current_conversation_id' in st.session_state:
        del st.session_state.current_conversation_id
    if 'current_messages' in st.session_state:
        del st.session_state.current_messages
    if 'conversation_title' in st.session_state:
        del st.session_state.conversation_title

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
                            'created': data.get('created', ''),
                            'updated': data.get('updated', ''),
                            'message_count': len(data.get('messages', []))
                        })
            except Exception as e:
                print(f"Error loading conversation {filename}: {e}")
                continue
    conversations.sort(key=lambda x: x.get('updated', ''), reverse=True)
    return conversations

def load_conversation(user_id, conversation_id):
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
    history_dir = "user_histories"
    user_dir = os.path.join(history_dir, user_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    filepath = os.path.join(user_dir, f"{conversation_id}.json")
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
    history_dir = "user_histories"
    filepath = os.path.join(history_dir, user_id, f"{conversation_id}.json")
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
    if len(question.strip()) > 50:
        title += "..."
    return title

def initialize_conversation_state(user_id):
    if 'current_conversation_id' not in st.session_state:
        conversations = get_user_conversations(user_id)
        if conversations and len(conversations) > 0:
            last_conv = conversations[0]
            conv_data = load_conversation(user_id, last_conv['id'])
            if conv_data:
                st.session_state.current_conversation_id = last_conv['id']
                st.session_state.current_messages = conv_data.get('messages', [])
                st.session_state.conversation_title = conv_data.get('title', 'Percakapan Baru')
            else:
                create_new_conversation()
        else:
            create_new_conversation()

# --- ENVIRONMENT SETUP ---
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

# --- MULTI-QUERY GENERATION ---
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

# --- RECIPROCAL RANK FUSION ---
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

# --- ADVANCED RAG CHAIN ---
def run_advanced_rag(db, json_data: Dict, query: str) -> str:
    """
    LEVEL 5 RAG: Multi-Query + Fusion + Hybrid Search
    """
    with st.spinner("🤖 AI Level 5: Multi-Query Fusion + Smart Chunking..."):
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.1
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

# --- MAIN APP ---
def main():
    # CSS sama seperti sebelumnya
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {visibility: hidden;}
    
    .stApp {
        background-color: #212121 !important;
    }
    
    section[data-testid="stSidebar"] * {
        background-color: #181818 !important;
    }
    
    section[data-testid="stSidebar"] button[kind="primary"],
    section[data-testid="stSidebar"] button[kind="secondary"],
    section[data-testid="stSidebar"] .stButton > button {
        background-color: initial !important;
        background: initial !important;
    }
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    setup_environment()
    user_id = get_or_create_user_id()
    initialize_conversation_state(user_id)
    
    # Sidebar (sama seperti sebelumnya - disingkat untuk hemat space)
    with st.sidebar:
        st.markdown("### 💬 Riwayat Chat")
        if st.button("➕ Percakapan Baru", use_container_width=True, type="primary"):
            create_new_conversation()
            st.rerun()
        
        conversations = get_user_conversations(user_id)
        if conversations:
            for conv in conversations:
                col1, col2 = st.columns([7, 1])
                with col1:
                    is_active = conv['id'] == st.session_state.current_conversation_id
                    if st.button(conv['title'], key=f"conv_{conv['id']}", 
                                use_container_width=True, disabled=is_active,
                                type="primary" if is_active else "secondary"):
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
                                create_new_conversation()
                            st.rerun()
        
        if st.button("🔄 Reset Session", use_container_width=True):
            reset_user_session()
            st.rerun()
    
    # Main content
    st.markdown("""
    <h1 style='font-size: 1.8rem; font-weight: 700; margin-top: -4rem; margin-bottom: 0.5rem;'>
        📄 Analisis Database Medis AI (JSON Smart Chunking)
    </h1>
    """, unsafe_allow_html=True)
    
    col_title, col_info = st.columns([3, 1])
    with col_title:
        st.markdown(f"### 💬 {st.session_state.conversation_title}")
    with col_info:
        st.caption(f"📊 {len(st.session_state.current_messages)} pesan")
    
    st.divider()
    
    # Load JSON database
    json_file = "medical_database_structured.json"
    
    if not os.path.exists(json_file):
        st.error(f"❌ File '{json_file}' tidak ditemukan!")
        st.info("💡 Jalankan dulu script 'convert_txt_to_json.py' untuk convert database Anda.")
        st.stop()
    
    try:
        # Load & process JSON
        json_data = load_json_database(json_file)
        documents = create_smart_chunks(json_data)
        vector_db = create_vector_store(documents)
        
        st.success(f"✅ Database loaded: {json_data['metadata']['total_cases']} cases dari {len(json_data['metadata']['categories'])} kategori")
        
        # Display messages
        if len(st.session_state.current_messages) == 0:
            st.markdown("""
            <p style="color: #fff; padding: 0.75rem 1rem; margin: 1rem 0;">
                👋 Database siap! Tanya tentang diagnosa, kode ICD-10, atau aspek koding.
            </p>
            """, unsafe_allow_html=True)
        
        for msg in st.session_state.current_messages:
            import html
            question_text = html.escape(msg["question"]).replace("\n", "<br>")
            answer_text = html.escape(msg["answer"]).replace("\n", "<br>")
            
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 1.5rem; padding: 0 1rem;">
                <div style="background-color: #2f2f2f; color: #ececec; padding: 0.75rem 1rem; border-radius: 1.25rem; max-width: 70%;">
                    {question_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start; margin-bottom: 2rem; padding: 0 1rem;">
                <div style="color: #ececec; padding: 0.75rem 0.5rem; width: 100%; line-height: 1.75;">
                    {answer_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat input
        pertanyaan_user = st.chat_input("💭 Tanya tentang diagnosa, kode ICD-10, prosedur...")
        
        if pertanyaan_user:
            if pertanyaan_user.strip():
                import html
                question_escaped = html.escape(pertanyaan_user).replace("\n", "<br>")
                
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 1.5rem; padding: 0 1rem;">
                    <div style="background-color: #2f2f2f; color: #ececec; padding: 0.75rem 1rem; border-radius: 1.25rem; max-width: 70%;">
                        {question_escaped}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Run Advanced RAG
                final_answer = run_advanced_rag(vector_db, json_data, pertanyaan_user)
                
                answer_escaped = html.escape(final_answer).replace("\n", "<br>")
                
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin-bottom: 2rem; padding: 0 1rem;">
                    <div style="color: #ececec; padding: 0.75rem 0.5rem; width: 100%; line-height: 1.75;">
                        {answer_escaped}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                chat_entry = {
                    "question": pertanyaan_user,
                    "answer": final_answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.current_messages.append(chat_entry)
                
                if len(st.session_state.current_messages) == 1:
                    st.session_state.conversation_title = generate_title_from_first_question(pertanyaan_user)
                
                save_conversation(
                    user_id,
                    st.session_state.current_conversation_id,
                    st.session_state.conversation_title,
                    st.session_state.current_messages
                )
                
                st.rerun()
    
    except Exception as e:
        st.error(f"❌ ERROR: {e}")
        with st.expander("🐛 Detail Error"):
            st.code(str(e))

if __name__ == "__main__":
    main()
