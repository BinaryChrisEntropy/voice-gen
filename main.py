import streamlit as st
import requests
import io

# Must be the first Streamlit command
st.set_page_config(
    page_title="AudioXVoiceGen",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a premium dark UI
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #00FFC2 !important; /* Cyan-Green accent */
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #00FFC2;
        color: #0E1117;
        border: none;
        padding: 12px;
        font-weight: 700;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00D1A0;
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 255, 194, 0.4);
    }
    .stTextArea>div>div>textarea {
        background-color: #1E1E1E;
        color: white;
        border: 1px solid #333;
    }
    .status-container {
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 20px;
        background: #1E1E1E;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

BACKEND_URL = "http://localhost:8000"

def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/status", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        return None
    return None

def main():
    st.markdown("<h1 style='text-align: center;'>🎙️ AudioXVoiceGen</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>High-fidelity Zero-Shot Voice Cloning</p>", unsafe_allow_html=True)

    # Backend Status Sidebar
    status = check_backend_status()
    with st.sidebar:
        st.header("Backend Settings")
        if status:
            st.success(f"Connected: {status['device'].upper()} ({status['dtype']})")
        else:
            st.error("Backend Offline - Start server.py first")
        
        api_url = st.text_input("API URL", value=BACKEND_URL)

    # Main UI Tabs
    tab1, tab2 = st.tabs(["🗣️ Voice Cloning & Generation", "ℹ️ About MOSS-TTS"])

    with tab1:
        st.markdown("### 🎙️ 1. Voice Style")
        st.info("Upload a 5-10 second sample to capture the voice's characteristics (style, tone, pitch).")
        
        audio_file = st.file_uploader("Upload Audio Sample", type=["wav", "mp3", "m4a"])
        
        if audio_file:
            st.audio(audio_file)
            
            st.markdown("### 📝 2. Target Content")
            target_text = st.text_area("What should the voice say?", placeholder="Type the text you want to generate in the cloned voice...", height=150)
            
            if st.button("Generate Audio in this Style"):
                if not target_text.strip():
                    st.warning("Please enter some text.")
                elif not status:
                    st.error("Backend is not responding. Please check your FastAPI server.")
                else:
                    with st.spinner("Processing with MOSS-TTS (8B) and Whisper..."):
                        try:
                            # Prepare file and data
                            files = {"reference_audio": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                            data = {"text": target_text}
                            
                            response = requests.post(f"{BACKEND_URL}/generate", files=files, data=data)
                            
                            if response.status_code == 200:
                                st.success("Voice successfully cloned and generated!")
                                st.audio(response.content, format="audio/wav")
                                # Download button
                                st.download_button(
                                    label="Download Audio",
                                    data=response.content,
                                    file_name="cloned_voice.wav",
                                    mime="audio/wav"
                                )
                            else:
                                st.error(f"Backend Error: {response.json().get('detail', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Connection Error: {e}")

    with tab2:
        st.markdown("""
        ### Info
        This application uses the **MOSS-TTS 8B** model for high-quality speech synthesis and zero-shot voice cloning.
        """)

if __name__ == "__main__":
    main()
