import torch
import torchaudio
import io
import os
import uvicorn
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import Response
from transformers import AutoModel, AutoProcessor

app = FastAPI(title="AudioXVoiceGen API")

# Configuration as constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Initialize app state
app.state.model = None
app.state.processor = None

def load_models(device, dtype):
    """Loads the models and returns them instead of setting globals."""
    try:
        print(f"Loading MOSS-TTS model on {device}...")
        model_id = "OpenMOSS-Team/MOSS-TTS"
        
        # Load MOSS-TTS Processor
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        # Resolve attention implementation
        attn_implementation = "sdpa"
        if device == "cuda":
            try:
                import importlib.util
                if importlib.util.find_spec("flash_attn") is not None and dtype in {torch.float16, torch.bfloat16}:
                    major, _ = torch.cuda.get_device_capability()
                    if major >= 8:
                        attn_implementation = "flash_attention_2"
            except:
                pass

        # Load MOSS-TTS Model
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        print("All models loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

@app.on_event("startup")
async def startup_event():
    # To enable, uncomment:
    # app.state.model, app.state.processor = load_models(DEVICE, DTYPE)
    pass

@app.get("/status")
async def get_status():
    return {
        "model_loaded": app.state.model is not None,
        "device": DEVICE,
        "dtype": str(DTYPE)
    }

@app.post("/generate")
async def generate_audio(
    request: Request,
    text: str = Form(...),
    reference_audio: UploadFile = File(None)
):
    # Access models via app.state
    model = request.app.state.model
    processor = request.app.state.processor

    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    temp_path = None
    try:
        if reference_audio:
            # Save uploaded file to a temporary location for the processor
            suffix = os.path.splitext(reference_audio.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await reference_audio.read()
                tmp.write(content)
                temp_path = tmp.name
            
            # Use the HF style: reference argument in build_user_message
            conversations = [
                [processor.build_user_message(text=text, reference=[temp_path])]
            ]
            mode = "generation"
        else:
            # Direct TTS mode
            conversations = [[processor.build_user_message(text=text)]]
            mode = "generation"

        # Generate using the simpler HF approach
        with torch.no_grad():
            batch = processor(conversations, mode=mode)
            input_ids = batch["input_ids"].to(device=DEVICE)
            attention_mask = batch["attention_mask"].to(device=DEVICE)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4096,
            )

            # Decode and get audio
            decoded = processor.decode(outputs)
            generated_audio = decoded[0].audio_codes_list[0]
            
            # Export to WAV
            out_io = io.BytesIO()
            torchaudio.save(out_io, generated_audio.unsqueeze(0), processor.model_config.sampling_rate, format="wav")
            out_io.seek(0)
            
            return Response(content=out_io.read(), media_type="audio/wav")

    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
