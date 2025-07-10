import streamlit as st
import os
import cv2
import whisper
import subprocess
import base64
import requests
import json
from tempfile import NamedTemporaryFile

# --- Settings ---
API_KEY = "9950e10d61694f288b5015e16c86112c"
DEPLOYMENT_URL = "https://api.turboline.ai/coreai/deployments/model-router/chat/completions?api-version=2025-01-01-preview"
HEADERS = {
    "Content-Type": "application/json",
    "tl-key": API_KEY
}
FFMPEG_PATH = r"D:\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)

# --- Streamlit UI ---
st.set_page_config(page_title="Sports Video to News Generator")
st.title("🏆 Sports Video ➤ News Article Generator")
st.markdown("Upload a sports video, and this tool will analyze frames and audio to generate a professional news article.")

video_file = st.file_uploader("📤 Upload Sports Video", type=["mp4", "mkv", "mov", "avi"])

# --- Functions ---
def extract_frames(video_path, output_folder, frame_rate=1):
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * frame_rate) if fps > 0 else 30

    count = 0
    saved_count = 0
    success, image = vidcap.read()
    while success:
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, image)
            saved_count += 1
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    return [os.path.join(output_folder, f) for f in sorted(os.listdir(output_folder)) if f.endswith(".jpg")]

def analyze_image(image_path):
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what is happening in this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }
        ],
        "max_tokens": 400
    }
    response = requests.post(DEPLOYMENT_URL, headers=HEADERS, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"[Error analyzing image: {response.status_code}]"

def transcribe_audio(video_path, audio_output):
    ffmpeg_command = [
        FFMPEG_PATH, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", audio_output
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    model = whisper.load_model("medium")
    result = model.transcribe(audio_output)
    return result["text"].strip()

def generate_combined_article(transcript, image_descriptions):
    combined_info = "\n".join(image_descriptions)
    prompt = f"""
Write a news article using only the information provided below. Do not assume or imagine any details. Stick strictly to the facts described in the transcript and image descriptions. Do not add any external context or interpretation.

Transcript:
\"\"\"{transcript}\"\"\"

Image Descriptions:
\"\"\"{combined_info}\"\"\"

Use this content to write a factual, concise, and objective news article. Include a headline and organize the information clearly. Only include points that are explicitly mentioned in the text or image descriptions.
"""
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200
    }
    response = requests.post(DEPLOYMENT_URL, headers=HEADERS, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"[Error generating article: {response.status_code}]"

# --- Main Execution ---
if video_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.info("🎞️ Extracting frames from video...")
    frames_folder = "extracted_frames"
    frame_files = extract_frames(video_path, frames_folder, frame_rate=1)

    st.info("🖼️ Analyzing image frames...")
    image_descriptions = []
    for frame in frame_files[:6]:  # Limit to 6 frames for speed
        image_descriptions.append(analyze_image(frame))

    st.info("🎧 Extracting and transcribing audio...")
    audio_path = "temp_audio.wav"
    transcript = transcribe_audio(video_path, audio_path)

    st.info("📝 Generating final news article...")
    article = generate_combined_article(transcript, image_descriptions)

    st.subheader("📰 Final Generated News Article")
    st.write(article)

    os.remove(video_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)
