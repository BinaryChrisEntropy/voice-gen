# 🎙️ AudioXVoiceGen

A high-fidelity, zero-shot voice cloning and generation platform powered by the **MOSS-TTS 8B** model. This project is optimized for deployment on high-performance compute environments like **NVIDIA DGX Spark**.

## 🚀 Features

- **Native Zero-Shot Voice Cloning**: Clone any voice with just a 5-10 second audio sample.
- **Automated Transcription**: Integrated OpenAI Whisper model to automatically transcribe reference audio for seamless cloning.
- **Premium Studio UI**: A modern, responsive Streamlit dashboard for controlling generation and monitoring backend status.
- **FastAPI Backend**: Robust asynchronous API handling model inference and audio processing.

## 💻 Hardware Requirements

The **MOSS-TTS 8B** model is a large-scale transformer and requires significant computational resources:

### GPU (Graphics Processing Unit)
- **Recommended**: NVIDIA RTX 3090, 4090, A100, or H100.
- **VRAM**: 
    - **Minimum**: 16 GB (may require quantization).
    - **Recommended**: **24 GB or more** for full-precision (`bfloat16`) and fast generation.
- **Architecture**: Pascal or newer (Ampere/Hopper recommended for FlashAttention-2 support).

### Memory & Storage
- **RAM**: 32 GB+ recommended.
- **Storage**: ~20 GB for model weights and dependencies.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd voicegen
   ```

2. **Setup Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Linux/Mac
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage

### 1. Start the FastAPI Backend
The backend must be started first to load the models into GPU memory.
```bash
python server.py
```

### 2. Start the Streamlit Frontend
Open a new terminal and run:
```bash
streamlit run main.py
```
Access the UI at `http://localhost:8501`.

## ⚙️ Technical Details

- **Model**: [OpenMOSS-Team/MOSS-TTS](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) (8B Parameters)
- **Transcription**: OpenAI Whisper (Base)
- **Optimization**: Uses `torch.bfloat16` and `flash_attention_2` where available.
- **Mode**: Uses the `continuation` strategy for zero-shot cloning.

## 📜 License
MIT
