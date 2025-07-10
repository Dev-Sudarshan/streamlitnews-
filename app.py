import streamlit as st
import whisper
import subprocess
import os
import requests
import json
from tempfile import NamedTemporaryFile
import subprocess

ffmpeg_path = r"D:\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
ffmpeg_command = [ffmpeg_path, "-i", "input.mp4", "output.mp3"]
st.set_page_config(page_title="🎤 Video to News Article Generator")

st.title("📹 Video to 📰 News Article Generator")
st.markdown("Upload a video file, and this app will transcribe it and turn the transcript into a news article using GPT-4!")

# Upload video
video_file = st.file_uploader("📤 Upload your video (.mp4, .mkv, etc.)", type=["mp4", "mkv", "mov", "avi"])

if video_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    # Extract audio using ffmpeg
    audio_path = "audio.wav"
    st.info("🎧 Extracting audio...")
    ffmpeg_command = [
        ffmpeg_path, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", audio_path
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Load Whisper model
    st.info("🧠 Transcribing with Whisper model (this may take a while)...")
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path)
    transcript = result["text"].strip()

    # Show transcript
    st.subheader("📝 Transcription")
    st.text_area("Transcript", transcript, height=200)

    # GPT-4 API call
    st.info("✍️ Generating news article with GPT-4...")

    url = "https://api.turboline.ai/coreai/deployments/model-router/chat/completions?api-version=2025-01-01-preview"
    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "tl-key": "9950e10d61694f288b5015e16c86112c"
    }

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a professional news article writer. Based on the transcript provided, generate a clear, concise, and engaging news article."
            },
            {
                "role": "user",
                "content": transcript
            }
        ],
        "max_tokens": 4000
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        article = response.json()['choices'][0]['message']['content']
        st.subheader("📰 Generated News Article")
        st.write(article)
    else:
        st.error(f"❌ Error {response.status_code}: {response.text}")

    # Clean up temp files
    os.remove(video_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)
