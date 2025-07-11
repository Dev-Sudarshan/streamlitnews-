import streamlit as st
import os
import cv2
import whisper
import subprocess
import base64
import requests
import json
from tempfile import NamedTemporaryFile
import shutil

# --- Settings ---
API_KEY = "9950e10d61694f288b5015e16c86112c"
DEPLOYMENT_URL = "https://api.turboline.ai/coreai/deployments/model-router/chat/completions?api-version=2025-01-01-preview"
HEADERS = {
    "Content-Type": "application/json",
    "tl-key": API_KEY
}
FFMPEG_PATH = r"D:\\ffmpeg-7.1.1-essentials_build\\bin\\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)

# --- Streamlit UI ---
st.set_page_config(page_title="Sports Video to News Generator")
st.title("🏆 Sports Video ➤ News Article Generator")
st.markdown("Upload a sports video, and this tool will analyze frames and audio to generate a professional news article.")

video_file = st.file_uploader("📤 Upload Sports Video", type=["mp4", "mkv", "mov", "avi"])

# --- Functions ---
def extract_frame_groups(video_path, output_folder, fps=1, group_size=5):
    """Extract frames from video and group them"""
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    vidcap = cv2.VideoCapture(video_path)
    actual_fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(actual_fps / fps) if actual_fps > 0 else 30
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    count, saved = 0, 0
    group_index = 0
    current_group = []
    all_groups = []
    
    progress_bar = st.progress(0)
    
    success, image = vidcap.read()
    while success:
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"group{group_index}_frame{saved}.jpg")
            cv2.imwrite(frame_filename, image)
            current_group.append(frame_filename)
            saved += 1
            
            if saved == group_size:
                all_groups.append(current_group)
                current_group = []
                saved = 0
                group_index += 1
        
        success, image = vidcap.read()
        count += 1
        
        # Update progress
        progress = min(count / total_frames, 1.0)
        progress_bar.progress(progress)
    
    # Add remaining frames if any
    if current_group:
        all_groups.append(current_group)
    
    vidcap.release()
    progress_bar.empty()
    return all_groups

def describe_image_with_scoring(image_path, timestamp):
    """Describe image and provide importance score"""
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this football match frame and provide:\n"
                            "1. A detailed description of what's happening\n"
                            "2. An importance score (1-10) based on these criteria:\n"
                            "   - 10: Goal being scored, penalty kick, red card\n"
                            "   - 8-9: Shot on goal, yellow card, crucial save\n"
                            "   - 6-7: Corner kick, free kick, player celebration\n"
                            "   - 4-5: Regular play, passing, running\n"
                            "   - 1-3: Players standing, crowd shots, referee walking\n\n"
                            "Format your response as:\n"
                            "DESCRIPTION: [your description]\n"
                            "SCORE: [number 1-10]\n"
                            "REASON: [why this score]"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    }
                ]
            }
        ],
        "max_tokens": 400
    }
    
    try:
        response = requests.post(DEPLOYMENT_URL, headers=HEADERS, data=json.dumps(payload))
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content'].strip()
            
            # Parse the response
            lines = content.split('\n')
            description = ""
            score = 1
            reason = ""
            
            for line in lines:
                if line.startswith("DESCRIPTION:"):
                    description = line.replace("DESCRIPTION:", "").strip()
                elif line.startswith("SCORE:"):
                    try:
                        score = int(line.replace("SCORE:", "").strip())
                    except:
                        score = 1
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()
            
            return {
                "description": description or content,
                "score": max(1, min(10, score)),  # Ensure score is between 1-10
                "reason": reason,
                "timestamp": timestamp,
                "image_path": image_path
            }
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
    
    return {
        "description": "[Error analyzing image]",
        "score": 1,
        "reason": "API Error",
        "timestamp": timestamp,
        "image_path": image_path
    }

def find_best_frames_per_group(frame_groups):
    """Find the best frame from each group"""
    best_frames = []
    all_frame_data = []
    
    progress_bar = st.progress(0)
    total_groups = len(frame_groups)
    
    for group_idx, group in enumerate(frame_groups):
        group_data = []
        
        # Analyze each frame in the group
        for frame_idx, frame_path in enumerate(group):
            timestamp = group_idx * 5 + frame_idx  # Approximate timestamp
            frame_data = describe_image_with_scoring(frame_path, timestamp)
            group_data.append(frame_data)
            all_frame_data.append(frame_data)
        
        # Find best frame in this group
        best_frame_in_group = max(group_data, key=lambda x: x['score'])
        best_frames.append(best_frame_in_group)
        
        # Update progress
        progress = (group_idx + 1) / total_groups
        progress_bar.progress(progress)
    
    progress_bar.empty()
    return best_frames, all_frame_data

def find_global_best_frame(best_frames):
    """Find the overall best frame from all group winners"""
    if not best_frames:
        return None
    
    # Sort by score (descending) and return the best one
    global_best = max(best_frames, key=lambda x: x['score'])
    return global_best

def transcribe_audio(video_path, audio_output):
    """Extract and transcribe audio from video"""
    try:
        ffmpeg_command = [
            FFMPEG_PATH, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", audio_output
        ]
        
        result = subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            return "[Error: Could not extract audio]"
        
        model = whisper.load_model("medium")
        result = model.transcribe(audio_output)
        return result["text"].strip()
    except Exception as e:
        return f"[Error transcribing audio: {str(e)}]"

def generate_article(transcript, all_descriptions, best_frame_data):
    """Generate news article from transcript and descriptions"""
    # Prepare descriptions text
    descriptions_text = "\n".join([
        f"Frame {i+1}: {desc['description']} (Importance: {desc['score']}/10)"
        for i, desc in enumerate(all_descriptions)
    ])
    
    prompt = f"""
Write a factual sports news article based on the transcript and image analysis below. 

KEY INSTRUCTIONS:
- Only use information provided below - do not invent any details
- Focus on the most important events (highest scored frames)
- Structure: Headline, then 2-3 paragraphs
- Be concise but informative
- If information is unclear, say so rather than guessing

AUDIO TRANSCRIPT:
{transcript}

IMAGE ANALYSIS:
{descriptions_text}

MOST IMPORTANT MOMENT:
{best_frame_data['description']} (Score: {best_frame_data['score']}/10)
Reason: {best_frame_data['reason']}

Write the article now:
"""
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }
    
    try:
        response = requests.post(DEPLOYMENT_URL, headers=HEADERS, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error generating article: {str(e)}")
    
    return "[Error generating article]"

# --- Main Execution ---
if video_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    try:
        st.info("🎞️ Extracting frame groups...")
        frame_groups = extract_frame_groups(video_path, "frames", fps=1, group_size=5)
        
        if not frame_groups:
            st.error("No frames could be extracted from the video")
        else:
            st.success(f"Extracted {len(frame_groups)} frame groups")
            
            st.info("🔍 Analyzing frames and finding best moments...")
            best_frames, all_frame_data = find_best_frames_per_group(frame_groups)
            
            if best_frames:
                global_best_frame = find_global_best_frame(best_frames)
                st.success(f"Found {len(best_frames)} key moments")
                
                st.info("🎧 Transcribing audio...")
                audio_path = "temp_audio.wav"
                transcript = transcribe_audio(video_path, audio_path)
                
                st.info("📝 Generating article...")
                article = generate_article(transcript, all_frame_data, global_best_frame)
                
                # Display results
                st.subheader("📰 Generated News Article")
                
                if global_best_frame:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.image(global_best_frame['image_path'], 
                                caption=f"🖼️ Key Moment (Score: {global_best_frame['score']}/10)", 
                                use_container_width=True)
                    
                    with col2:
                        st.write("**Frame Analysis:**")
                        st.write(f"**Description:** {global_best_frame['description']}")
                        st.write(f"**Importance Score:** {global_best_frame['score']}/10")
                        st.write(f"**Reason:** {global_best_frame['reason']}")
                        st.write(f"**Timestamp:** ~{global_best_frame['timestamp']} seconds")
                
                st.write("---")
                st.write(article)
                
                # Optional: Show all analyzed frames
                with st.expander("🔍 View All Analyzed Frames"):
                    for i, frame_data in enumerate(all_frame_data):
                        st.write(f"**Frame {i+1}** (Score: {frame_data['score']}/10)")
                        st.write(f"Description: {frame_data['description']}")
                        st.write(f"Reason: {frame_data['reason']}")
                        st.write("---")
                
                # Optional: Show transcript
                with st.expander("📝 View Audio Transcript"):
                    st.write(transcript)
                
                # Cleanup
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    
            else:
                st.error("Could not analyze any frames")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    finally:
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists("frames"):
            shutil.rmtree("frames")