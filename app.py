import streamlit as st
import time
import utils
from PIL import Image
import numpy as np
import uuid

# Set page config
st.set_page_config(page_title="Annotation Assistant", layout="wide", page_icon="✨")

# --- Premium Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    /* Global Theme */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background Gradient - "Deep Space" Theme */
    .stApp {
        background: radial-gradient(circle at top left, #1a202c, #0d1117);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }
    
    /* Hide Header and Default Elements */
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
        max_width: 1000px;
    }
    
    /* Headers */
    h1 {
        background: -webkit-linear-gradient(45deg, #60A5FA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Dotted Upload Box */
    [data-testid='stFileUploader'] section {
        border: 1px dashed #4A5568;
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        padding: 4rem 2rem;
        min-height: 300px;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
    }
    [data-testid='stFileUploader'] section:hover {
        background-color: rgba(255, 255, 255, 0.05);
        border-color: #60A5FA;
        cursor: pointer;
        box-shadow: 0 0 25px rgba(96, 165, 250, 0.15);
        transform: scale(1.01);
    }
    
    /* Buttons - "Glass" Style */
    .stButton > button {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        background: rgba(255,255,255,0.05);
        color: #E2E8F0;
        font-weight: 500;
        backdrop-filter: blur(5px);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: rgba(255,255,255,0.1);
        border-color: #60A5FA;
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Secondary/Reset Button */
    button[kind="secondary"] {
        color: #F87171 !important;
        border-color: rgba(248, 113, 113, 0.2) !important;
    }
    button[kind="secondary"]:hover {
        background: rgba(248, 113, 113, 0.1) !important;
        border-color: #F87171 !important;
        box-shadow: 0 0 10px rgba(248, 113, 113, 0.2);
    }
    
    /* Session Buttons in Sidebar */
    .session-btn {
        width: 100%;
        text-align: left;
        margin-bottom: 5px;
    }

    /* Metrics Bar - Floating "Pill" */
    .metric-pill {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        background: rgba(16, 24, 39, 0.8);
        border: 1px solid #2D3748;
        padding: 10px 24px;
        border-radius: 100px;
        margin: 20px auto; /* Centered */
        width: fit-content;
        font-size: 0.9rem;
        color: #94A3B8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        color: #34D399;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
    
    /* Reasoning Cards - Centered & Wide */
    .reasoning-container {
        margin-top: 20px;
        background: rgba(30, 41, 59, 0.3);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .reasoning-card {
        background: rgba(255,255,255,0.02);
        border-left: 3px solid #3B82F6;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
    .reasoning-label {
        font-weight: 600;
        color: #E2E8F0;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    .reasoning-text {
        font-size: 0.85rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    
    /* Input Area */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    
    /* Slider Customization */
    div[data-testid="stSlider"] > div {
        max_width: 300px;
        margin: auto;
    }
    
    /* CENTER IMAGES */
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    div[data-testid="stImage"] > img {
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "sessions" not in st.session_state:
    # Structure: { session_id: { name, history, detections, image, metrics, timestamp } }
    st.session_state.sessions = {}
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

# Helper 1: Create a new session
def create_session(name="New Chat"):
    session_id = str(uuid.uuid4())
    st.session_state.sessions[session_id] = {
        "name": name,
        "history": [],
        "detections": [],
        "image": None,
        "metrics": {},
        "created_at": time.time()
    }
    st.session_state.active_session_id = session_id
    return session_id

# Helper 2: Get active session data
def get_active_session():
    if not st.session_state.active_session_id:
        create_session()
    return st.session_state.sessions[st.session_state.active_session_id]

# Ensure at least one session exists
if not st.session_state.sessions:
    create_session()

current_session = get_active_session()

# --- Sidebar (Session Manager) ---
with st.sidebar:
    st.markdown("### 🗂️ Sessions")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        create_session()
        st.rerun()
        
    st.markdown("---")
    
    # Sort sessions by recency
    sorted_sessions = sorted(
        st.session_state.sessions.items(), 
        key=lambda x: x[1]['created_at'], 
        reverse=True
    )
    
    for s_id, s_data in sorted_sessions:
        # Hide empty "New Chat" sessions from the list unless active
        if s_data['image'] is None:
            continue
            
        is_active = (s_id == st.session_state.active_session_id)
        
        display_name = s_data['name']
        icon = "📂" if is_active else "📝"
        label = f"{icon} {display_name}"
        
        if st.button(label, key=f"sess_{s_id}", use_container_width=True, type="secondary" if not is_active else "primary"):
            st.session_state.active_session_id = s_id
            st.rerun()

# --- Model Loading ---
if not st.session_state.model_loaded:
    with st.spinner("Initializing AI Core..."):
        processor, model = utils.load_model()
        if processor and model:
            st.session_state.model_loaded = True
            st.session_state.processor = processor
            st.session_state.model = model
            st.rerun()
        else:
            st.error("Model Engine Failure.")
            st.stop()

# --- Main Workspace ---

# Header
col_logo, col_space = st.columns([6, 1])
with col_logo:
    if current_session['name'] == "New Chat":
        st.markdown("# Annotation Assistant")
    else:
        st.markdown(f"# {current_session['name']}")

# Logic
if current_session['image'] is None:
    # --- Upload State ---
    st.markdown(
        "<h3 style='text-align: center; color: #94A3B8; border: none;'>Upload an image to start this session</h3>", 
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader(
        "Upload Image", 
        type=["jpg", "png", "jpeg"], 
        key=f"uploader_{st.session_state.active_session_id}",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        current_session['image'] = image
        current_session['name'] = uploaded_file.name
        st.rerun()
        
else:
    # --- Analysis State ---
    
    # Image Controls
    img_width = st.slider("Adjust View Size", 300, 1500, 700, 50, help="Drag to resize the image view")
    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Main visual (Hero)
    display_image = current_session['image'].copy()
    
    if current_session['detections']:
        display_image = utils.draw_boxes(display_image, current_session['detections'])
    
    st.image(display_image, width=img_width)
    
    # 2. Results Actions & Metrics
    if current_session['detections']:
        # Metrics Row
        if current_session['metrics']:
            m = current_session['metrics']
            st.markdown(f"""
            <div class='metric-pill'>
                <span>Inference <span class='metric-value'>{m.get('inference_time', 0)}s</span></span>
                <span style='color: #4B5563'>|</span>
                <span>Total <span class='metric-value'>{m.get('total_time', 0)}s</span></span>
                <span style='color: #4B5563'>|</span>
                <span>Tokens <span class='metric-value'>{m.get('token_count', 0)}</span></span>
            </div>
            """, unsafe_allow_html=True)
            
        # Download Row
        c1, c2, c3 = st.columns([1, 1, 3]) # Bias to left
        with c1:
            # UPDATED: Pass usage metadata for Strict COCO compatibility
            coco_json = utils.convert_to_coco(
                current_session['detections'], 
                image_size=current_session['image'].size,
                filename=current_session['name']
            )
            st.download_button("Download JSON", coco_json, "annotations.json", "application/json", use_container_width=True)
        with c2:
            zip_buffer = utils.create_crops_zip(current_session['image'], current_session['detections'])
            st.download_button("Download ZIP", zip_buffer, "crops.zip", "application/zip", use_container_width=True)
        
        # 3. Reasoning Stream (Below)
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("### AI Insights")
        with st.container():
            st.markdown("<div class='reasoning-container'>", unsafe_allow_html=True)
            for det in current_session['detections'][::-1]:
                    label = det.get('label', 'Object')
                    reasoning = det.get('reasoning', None)
                    if not reasoning: reasoning = "Object detected based on visual features."
                    st.markdown(f"""
                    <div class='reasoning-card'>
                        <div class='reasoning-label'>{label}</div>
                        <div class='reasoning-text'>{reasoning}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Image loaded but no detections
        st.markdown(
            "<div style='text-align: center; margin-top: 20px; color: #64748B; font-style: italic;'>"
            "Waiting for instructions... Use the chat bar below."
            "</div>", 
            unsafe_allow_html=True
        )

# --- Floating Chat Bar ---
st.markdown("<br>", unsafe_allow_html=True)
prompt = st.chat_input("Describe objects to detect...")

if prompt:
    if current_session['image'] is None:
        st.error("Please upload an image first.")
    else:
        # Warning for HF Spaces Free Tier
        if "cpu" in str(st.session_state.get('device', 'cpu')):
             st.info("ℹ️ Running on CPU (Free Tier). Complex scenes may take 30-60s to analyze.")

        with st.status("Analyzing Scene...", expanded=True) as status:
            detections, updated_history, raw_text, metrics = utils.get_bounding_boxes(
                current_session['image'], 
                prompt, 
                current_session['history'], 
                st.session_state.processor, 
                st.session_state.model
            )
            
            if detections:
                current_session['detections'] = utils.smart_merge_detections(current_session['detections'], detections)
                current_session['history'] = updated_history
                current_session['metrics'] = metrics
                status.update(label="Complete", state="complete", expanded=False)
                st.rerun()
            else:
                status.update(label="No matches found.", state="error", expanded=False)
                st.toast(f"No match found.", icon="⚠️")
