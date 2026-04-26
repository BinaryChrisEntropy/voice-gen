import torch
import torchaudio
import whisper
import io
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor
import numpy as np

app = FastAPI(title="AudioXVoiceGen API")

# Global variables for model and processor
model = None
processor = None
whisper_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

def load_models():
    global model, processor, whisper_model
    try:
        print(f"Loading MOSS-TTS model on {device}...")
        pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS"
        
        # Load MOSS-TTS Processor
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True
        )
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        # Load MOSS-TTS Model
        # Using flash_attention_2 if possible (optimized for A100/H100 on DGX)
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if device == "cuda" else "sdpa",
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        print("Loading Whisper model for transcription...")
        whisper_model = whisper.load_model("base", device=device)
        
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # On local dev machine without 24GB VRAM, this will likely fail.
        # But we keep it as is for the DGX Spark.

@app.on_event("startup")
async def startup_event():
    # Only load models if we are not in a limited environment
    # In a real DGX, we would always load.
    #load_models()
    pass

@app.get("/status")
async def get_status():
    return {
        "model_loaded": model is not None,
        "device": device,
        "dtype": str(dtype)
    }

@app.post("/generate")
async def generate_audio(
    text: str = Form(...),
    reference_audio: UploadFile = File(None)
):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        if reference_audio:
            # VOICE CLONING MODE
            audio_bytes = await reference_audio.read()
            audio_io = io.BytesIO(audio_bytes)
            ref_audio, sr = torchaudio.load(audio_io)
            ref_audio = ref_audio[0] # Ensure single channel
            
            # Transcribe reference audio using Whisper
            # MOSS-TTS needs the transcription of the reference to 'prime' the voice
            temp_path = "temp_ref.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
            
            result = whisper_model.transcribe(temp_path)
            ref_text = result["text"]
            os.remove(temp_path)
            
            print(f"Ref Audio Transcribed: {ref_text}")
            
            # Build conversation for cloning
            conversations = [
                [
                    processor.build_user_message(text=ref_text + " " + text),
                    processor.build_assistant_message(audio_codes_list=[ref_audio])
                ]
            ]
            mode = "continuation"
        else:
            # TEXT-TO-SPEECH MODE (using a default voice or prompt)
            # For simplicity, we'll assume the user always provides a reference for cloning in this app.
            # If not, MOSS-TTS might need a default priming.
            raise HTTPException(status_code=400, detail="Reference audio is required for this implementation of cloning.")

        # Generate
        with torch.no_grad():
            batch = processor(conversations, mode=mode)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4096,
            )

            # Decode
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
