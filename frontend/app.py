import streamlit as st
import requests
import os
import markdown

# docker style
# public API URL for browser-side usage (proxied by nginx)
API_BASE_URL = os.getenv("BACKEND_URL", "/api")

# internal URL used by server-side Python requests (must include scheme and host)
BACKEND_INTERNAL_URL = os.getenv("BACKEND_INTERNAL_URL", "http://backend:8000")

# local style
# API_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# BACKEND_INTERNAL_URL = os.getenv("BACKEND_INTERNAL_URL", "http://localhost:8000")

#logging configuration
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* CSS Variables for Light Theme (Default) */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-gradient: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #ffffff 100%);
        
        --user-msg-bg: linear-gradient(135deg, rgba(226, 232, 240, 0.8), rgba(203, 213, 225, 0.8));
        --user-msg-border: rgba(148, 163, 184, 0.4);
        --user-msg-text: #0f172a;
        
        --assistant-msg-bg: #f1f5f9;
        --assistant-msg-border: rgba(226, 232, 240, 0.8);
        --assistant-msg-text: #1e293b;
        
        --meta-text: #64748b;
        --source-bg: rgba(241, 245, 249, 0.8);
        --source-border: rgba(226, 232, 240, 0.8);
        --source-header: #475569;
        
        --status-ready-bg: rgba(34, 197, 94, 0.15);
        --status-ready-border: rgba(34, 197, 94, 0.4);
        --status-ready-text: #15803d;
        
        --status-offline-bg: rgba(239, 68, 68, 0.15);
        --status-offline-border: rgba(239, 68, 68, 0.4);
        --status-offline-text: #991b1b;
        
        --badge-bg: rgba(245, 158, 11, 0.15);
        --badge-border: rgba(245, 158, 11, 0.4);
        --badge-text: #d97706;
        
        --input-bg: #ffffff;
        --input-border: rgba(226, 232, 240, 0.8);
        --input-text: #1e293b;
        
        --card-bg: rgba(248, 250, 252, 0.9);
        --card-border: rgba(226, 232, 240, 0.8);
        --card-text: #334155;
        
        --suggestion-bg: #ffffff;
        --suggestion-border: rgba(226, 232, 240, 0.8);
        --suggestion-text: #475569;
        --suggestion-hover-bg: #f1f5f9;
        --suggestion-hover-border: rgba(6, 182, 212, 0.6);
        --suggestion-hover-text: #1e293b;
    }
    
    /* Dark Theme Variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            
            --user-msg-bg: linear-gradient(135deg, rgba(51, 65, 85, 0.8), rgba(71, 85, 105, 0.8));
            --user-msg-border: rgba(100, 116, 139, 0.4);
            --user-msg-text: #f1f5f9;
            
            --assistant-msg-bg: #1e293b;
            --assistant-msg-border: rgba(51, 65, 85, 0.8);
            --assistant-msg-text: #e2e8f0;
            
            --meta-text: #94a3b8;
            --source-bg: rgba(30, 41, 59, 0.8);
            --source-border: rgba(51, 65, 85, 0.8);
            --source-header: #cbd5e1;
            
            --status-ready-bg: rgba(34, 197, 94, 0.2);
            --status-ready-border: rgba(34, 197, 94, 0.5);
            --status-ready-text: #86efac;
            
            --status-offline-bg: rgba(239, 68, 68, 0.2);
            --status-offline-border: rgba(239, 68, 68, 0.5);
            --status-offline-text: #fca5a5;
            
            --badge-bg: rgba(245, 158, 11, 0.2);
            --badge-border: rgba(245, 158, 11, 0.5);
            --badge-text: #fbbf24;
            
            --input-bg: #1e293b;
            --input-border: rgba(51, 65, 85, 0.8);
            --input-text: #f1f5f9;
            
            --card-bg: rgba(30, 41, 59, 0.9);
            --card-border: rgba(51, 65, 85, 0.8);
            --card-text: #cbd5e1;
            
            --suggestion-bg: #1e293b;
            --suggestion-border: rgba(51, 65, 85, 0.8);
            --suggestion-text: #cbd5e1;
            --suggestion-hover-bg: #334155;
            --suggestion-hover-border: rgba(6, 182, 212, 0.6);
            --suggestion-hover-text: #f1f5f9;
        }
    }
    
    /* Main background */
    .stApp { 
        background: var(--bg-gradient);
    }
    
    /* Chat message containers */
    .user-message {
        background: var(--user-msg-bg);
        border: 1px solid var(--user-msg-border);
        border-radius: 15px;
        padding: 15px 20px;
        margin: 10px 0;
        margin-left: 20%;
        color: var(--user-msg-text);
    }

    .assistant-message {
        background: var(--assistant-msg-bg);
        border: 1px solid var(--assistant-msg-border);
        border-radius: 15px;
        padding: 15px 20px;
        margin: 10px 0;
        margin-right: 20%;
        color: var(--assistant-msg-text);
    }

    /* Message metadata */
    .message-meta {
        font-size: 0.75rem;
        color: var(--meta-text);
        margin-top: 8px;
        display: flex;
        gap: 15px;
    }

    .meta-item {
        display: flex;
        align-items: center;
        gap: 5px;
    }

    /* Source documents */
    .source-box {
        background: var(--source-bg);
        border: 1px solid var(--source-border);
        border-radius: 10px;
        padding: 12px;
        margin: 5px 0;
        font-size: 0.85rem;
    }
    
    .source-header {
        color: var(--source-header);
        font-weight: 500;
        margin-bottom: 5px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .table-badge {
        background: var(--badge-bg);
        color: var(--badge-text);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    
    .status-ready {
        background: var(--status-ready-bg);
        border: 1px solid var(--status-ready-border);
        color: var(--status-ready-text);
    }
    
    .status-offline {
        background: var(--status-offline-bg);
        border: 1px solid var(--status-offline-border);
        color: var(--status-offline-text);
    }
    
    /* Machine context badge */
    .machine-badge {
        background: var(--badge-bg);
        border: 1px solid var(--badge-border);
        color: var(--badge-text);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #f59e0b, #ea580c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(6, 182, 212, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: var(--input-bg);
        border: 1px solid var(--input-border);
        border-radius: 10px;
        color: var(--input-text);
    }
    
    .stTextArea > div > div > textarea {
        background: var(--input-bg);
        border: 1px solid var(--input-border);
        border-radius: 10px;
        color: var(--input-text);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Welcome card */
    .welcome-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin: 50px auto;
        max-width: 600px;
        color: var(--card-text);
    }
    
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 20px;
    }
    
    /* Suggestion buttons */
    .suggestion-btn {
        background: var(--suggestion-bg);
        border: 1px solid var(--suggestion-border);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.2s ease;
        color: var(--suggestion-text);
        text-align: left;
    }
    
    .suggestion-btn:hover {
        background: var(--suggestion-hover-bg);
        border-color: var(--suggestion-hover-border);
        color: var(--suggestion-hover-text);
    }
    
    /* Additional elements that need theme support */
    .stExpander {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
    }
    
    .stMarkdown {
        color: var(--card-text);
    }
    
</style>
""", unsafe_allow_html=True)
# Custom CSS for dark theme styling
st.markdown("""
<style>
    /* Main background */
    .stApp { 
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #ffffff 100%);
    }
    

    /* Chat message containers (white theme) */
.user-message {
    background: linear-gradient(135deg, rgba(226, 232, 240, 0.8), rgba(203, 213, 225, 0.8));
    border: 1px solid rgba(148, 163, 184, 0.4);
    border-radius: 15px;
    padding: 15px 20px;
    margin: 10px 0;
    margin-left: 20%;
    color: #0f172a; /* Slate-900 text */
}

.assistant-message {
    background: #f1f5f9;
    border: 1px solid rgba(226, 232, 240, 0.8);
    border-radius: 15px;
    padding: 15px 20px;
    margin: 10px 0;
    margin-right: 20%;
    color: #1e293b; /* Slate-800 text */
}

/* Message metadata (lighter text for white theme) */
.message-meta {
    font-size: 0.75rem;
    color: #64748b; /* Slate-500 */
    margin-top: 8px;
    display: flex;
    gap: 15px;
}

.meta-item {
    display: flex;
    align-items: center;
    gap: 5px;
}

    /* Source documents */
    .source-box {
        background: rgba(241, 245, 249, 0.8);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 10px;
        padding: 12px;
        margin: 5px 0;
        font-size: 0.85rem;
    }
    
    .source-header {
        color: #475569;
        font-weight: 500;
        margin-bottom: 5px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .table-badge {
        background: rgba(245, 158, 11, 0.15);
        color: #d97706;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    
    .status-ready {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #15803d;
    }
    
    .status-offline {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #991b1b;
    }
    
    /* Machine context badge */
    .machine-badge {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #d97706;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #f59e0b, #ea580c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(6, 182, 212, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: #ffffff;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 10px;
        color: #1e293b;
    }
    
    .stTextArea > div > div > textarea {
        background: #ffffff;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 10px;
        color: #1e293b;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Welcome card */
    .welcome-card {
        background: rgba(248, 250, 252, 0.9);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin: 50px auto;
        max-width: 600px;
    }
    
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 20px;
    }
    
    /* Suggestion buttons */
    .suggestion-btn {
        background: #ffffff;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #475569;
        text-align: left;
    }
    
    .suggestion-btn:hover {
        background: #f1f5f9;
        border-color: rgba(6, 182, 212, 0.6);
        color: #1e293b;
    }
    
    

""", unsafe_allow_html=True)


show_sources = True  
show_metadata = True

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "machine_context" not in st.session_state:
    st.session_state.machine_context = None
if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None




def send_message(message: str, return_sources: bool = True):
    """Send a message to the API and get response"""
    try:
        response = requests.post(
            f"{BACKEND_INTERNAL_URL}/chat",
            json={"message": message, "return_sources": return_sources},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The model might be loading."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is the server running?"}
    except Exception as e:
        return {"error": str(e)}


def rate_answer(rating: str):
    """Rate the last answer"""
    try:
        response = requests.post(
            f"{BACKEND_INTERNAL_URL}/chat/rate",
            json={"rating": rating},
            timeout=10
        )
        return response.status_code == 200
    except:
        return False


def clear_history():
    """Clear chat history on the server"""
    try:
        response = requests.post(f"{BACKEND_INTERNAL_URL}/chat/clear", timeout=10)
        return response.status_code == 200
    except:
        return False


def get_machines():
    """Get list of available machines"""
    try:
        response = requests.get(f"{BACKEND_INTERNAL_URL}/machines", timeout=10)
        if response.status_code == 200:
            return response.json().get("machines", {})
        return {}
    except:
        return {}

def get_metrics():
    """Get system metrics"""
    try:
        response = requests.get(f"{BACKEND_INTERNAL_URL}/metrics", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

def set_machine_context(machine: str, original_name: str):
    """Set the machine context on the backend"""
    try:
        
        response = requests.post(
            f"{BACKEND_INTERNAL_URL}/machines/set_context",
            json={"machine_name": [machine], "original_name": original_name},
            timeout=10
        )
        return response.status_code == 200
    except:
        return False

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">🤖 P Bot 1.0</div>', unsafe_allow_html=True)
    

    st.markdown("Technischer Dokumentationsassistent für Pfeuffer-Maschinent")
    st.markdown("---")
    

    # Actions
    
    if st.button(" + Neuer Chat", use_container_width=True):
        if clear_history():
            st.session_state.messages = []
            st.session_state.machine_context = None
            st.session_state.machine_selected = False   # ← add this line
            st.rerun()
        else:
            st.error("Failed to clear history")
    

#Include your machines here. For now, it's an empty list. You can populate it with actual machine names.
MACHINES = [] 

if not st.session_state.get("machine_selected"):
        # ── Welcome / machine selection screen ──────────────────────────────
        st.markdown("""
        <div style="max-width:520px; margin:80px auto 0; text-align:center;">
            <div style="font-size:3rem; margin-bottom:16px;">🤖</div>
            <h2 style="font-weight:500; margin-bottom:8px;">Willkommen bei P Bot – Technischer Dokumentationsassistent für Pfeuffer-Maschinen</h2>
            <p style="color:#64748b; margin-bottom:32px;">
                Bitte wählen Sie einen Maschinenkontext aus, um Ihre Sitzung zu starten.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            selected_machine = st.selectbox(
                "Machine",
                options=["— Wählen Sie eine Pfeuffer-Maschine aus —"] + MACHINES,
                label_visibility="collapsed"
            )
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            if st.button("Start session", use_container_width=True):
                if selected_machine == "— Wählen Sie eine Pfeuffer-Maschine aus —":
                    st.warning("Bitte wählen Sie eine Maschine aus, um fortzufahren.")
                else:
                    with st.spinner("Maschinenkontext wird ermittelt…"):

                        machine_routing = "".join(selected_machine.split(" ")).lower()
                        logger.info(f"Setting machine context to {selected_machine} (routing: {machine_routing})")
                        ok = set_machine_context(machine_routing, selected_machine) # Pass the original machine name for logging and for retrieval based on the original name(for query rephrasing)
                        logger.info(f"Set machine context response: {ok}")
                    if ok:
                        st.session_state.machine_context = selected_machine
                        st.session_state.machine_selected = True
                        st.rerun()
                    else:
                        st.error("Could not set machine context. Is the server running?")

else:
        # Display messages
        st.markdown(f"""
            <div style="text-align:center; margin: 8px 0;">
                <p style="color:#64748b; font-size:0.85rem;">
                    🔧 Sprechen über <strong>{st.session_state.machine_context}</strong> — 
                    Zum Umschalten drücken <em>Neuer Chat</em> in der Seitenleiste.
                </p>
            </div>
        """, unsafe_allow_html=True)
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                html_content = markdown.markdown(msg["content"], extensions=["nl2br", "sane_lists"])
                st.markdown(
                    f'<div class="user-message">{html_content}</div>',
                    unsafe_allow_html=True
                )
            else:
                # Assistant message
                html_content = markdown.markdown(msg["content"], extensions=["nl2br", "sane_lists"])
                st.markdown(
                    f'<div class="assistant-message">{html_content}</div>',
                    unsafe_allow_html=True
                )
                
                # Metadata
              
                if show_metadata and msg.get("metadata"):
                    meta = msg["metadata"]
                    if meta.get("latency"):
                        st.caption(f"⏱️ {meta['latency']:.2f}s")
                
                # Sources
                
                if show_sources and msg.get("sources"):
                    with st.expander(f"📄 View {len(msg['sources'])} source(s)"):
                        for source in msg["sources"]:
                            source_meta = source.get("metadata", {})
                            source_name = source_meta.get("section", source_meta.get("source", "Unknown"))
                            content_type = source_meta.get("content_type", "text")
                            score =  source_meta.get('rerank_score','N/A')
                            score = f'score:{score}'
                            
                            
                            st.markdown(
                                f"""<div class="source-box">
                                    <div class="source-header">📄 {source_name}</div>
                                    <div style="color: #71717a; font-size: 0.8rem;">{source['content']}</div>
                                    <div style="color: #71717a; font-size: 0.8rem;">{score}</div>
                                </div>""",
                                unsafe_allow_html=True
                            )
                
                # Rating buttons for the last assistant message
                if idx == len(st.session_state.messages) - 1 and msg.get("metadata", {}).get("used_rag"):
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        if st.button("👍", key=f"thumbs_up_{idx}"):
                            if rate_answer("right"):
                                st.success("Thanks for your feedback!")
                            else:
                                st.error("Failed to submit rating")
                    with col2:
                        if st.button("👎", key=f"thumbs_down_{idx}"):
                            if rate_answer("wrong"):
                                st.success("Thanks for your feedback!")
                            else:
                                st.error("Failed to submit rating")
                    with col3:
                        if st.button("⚠️", key = f"maybe_{idx}"):
                            if rate_answer("maybe"):
                                st.success("Thanks for your feedback!")
                            else:
                                st.error("Failed to submit rating")

        # Chat input — now only shown after session starts
        # Info text above the form
        st.markdown("""
            <style>
            [data-testid="InputInstructions"] { display: none !important; }
            </style>
        """, unsafe_allow_html=True)
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.text_input(
                    "Message",
                    placeholder="Ihre Frage.",
                    label_visibility="collapsed"
                )
            
            with col2:
                submit = st.form_submit_button("Senden", use_container_width=True)
            
            if submit and user_input.strip():
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Get response
                with st.spinner("💭"):
                    response = send_message(user_input, return_sources=show_sources)
                
                if "error" in response:
                    st.error(response["error"])
                    # Remove the user message if there was an error
                    st.session_state.messages.pop()
                else:
                    assistant_msg = {
                        "role": "assistant",
                        "content": response.get("answer", ""),
                        "sources": response.get("sources", []),
                        "metadata": {
                            "intent": response.get("intent"),
                            "machine_context": response.get("machine_context"),
                            "latency": response.get("latency"),
                            "used_rag": response.get("used_rag")
                        }
                    }
                    st.session_state.messages.append(assistant_msg)
                   
                
                st.rerun()

