import streamlit as st
import whisper
import subprocess
import os
import requests
import json
from tempfile import NamedTemporaryFile

# Set ffmpeg path
ffmpeg_bin = r"D:\ffmpeg-7.1.1-essentials_build\bin"
ffmpeg_path = os.path.join(ffmpeg_bin, "ffmpeg.exe")
os.environ["PATH"] += os.pathsep + ffmpeg_bin

# Set Streamlit page settings
st.set_page_config(page_title="🎤 Sports Video to News Article")

st.title("🏆 Sports Video to 📰 News Article Generator")
st.markdown("Upload a sports video file and this will turn it into a news article.")

# Upload video
video_file = st.file_uploader("📤 Upload your sports video (.mp4, .mkv, etc.)", type=["mp4", "mkv", "mov", "avi"])

if video_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    # Extract audio using ffmpeg
    audio_path = "audio.wav"
    st.info("🔊 Extracting audio...")
    ffmpeg_command = [
        ffmpeg_path, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", audio_path
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Transcribe with Whisper
    st.info("🧠 Transcribing audio...")
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path)
    transcript = result["text"].strip()

    # Generate news article using GPT-4
    st.info("📝 Generating news article...")

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
                "content": "You are a world-class sports journalist working for a top international publication. Your job is to craft a compelling, clear, and concise news article based on a raw transcript from a sports event video. Focus on accuracy, tone, and storytelling. Structure the article like a professional report: include a headline, a strong lead, key highlights of the event, and any standout performances, controversies, or turning points. Avoid fluff and filler — prioritize factual reporting with a journalistic tone and a narrative style that engages the reader. Use relevant sports terminology appropriately, and maintain objectivity while making the article exciting and readable. Your writing should feel like it belongs on the front page of a major sports news site."
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
