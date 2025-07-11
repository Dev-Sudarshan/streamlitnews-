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
import time

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
st.markdown("Upload a sports video or provide raw match data to generate a professional news article.")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📹 Video Upload", "📝 Raw Data Input"])

with tab1:
    st.markdown("Upload a sports video, and this tool will analyze frames and audio to generate a professional news article.")
    video_file = st.file_uploader("📤 Upload Sports Video", type=["mp4", "mkv", "mov", "avi"])

with tab2:
    st.markdown("Enter raw match data, commentary, or any text information about the match to generate a news article.")
    raw_match_data = st.text_area(
        "📝 Enter Match Data", 
        placeholder="Enter match commentary, statistics, player information, or any relevant match details...",
        height=200
    )
    generate_from_text = st.button("Generate Article from Text Data")

# --- Functions ---
def image_to_base64(image_path):
    """Convert image to base64 string for storage"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error converting image to base64: {str(e)}")
        return None

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
                            "   CRITICAL: WE WANT THE MOMENT OF ACTION, NOT THE RESULT:\n"
                            "   - 10: Player's foot making contact with ball for a shot, player striking the ball\n"
                            "   - 9: Player in shooting motion, about to kick the ball\n"
                            "   - 8: Player preparing to shoot, ball approaching player's foot\n"
                            "   - 7: Corner kick being taken, free kick setup\n"
                            "   - 6: Regular play, passing, running\n"
                            "   - 2-3: Ball already in net, goal celebration, players celebrating\n"
                            "   - 1: Players standing, crowd shots, referee walking\n\n"
                            "ABSOLUTELY CRITICAL: If you see the ball already in the goal net or players celebrating a goal, give it a LOW score (2-3). We want the SHOOTING ACTION, not the goal result.\n\n"
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

def generate_article_from_text(raw_data):
    """Generate news article from raw text data only"""
    prompt = f"""
Write a factual sports news article based on the raw match data provided below.

KEY INSTRUCTIONS:
- Only use information provided below - do not invent any details
- Structure: Headline, then 2-3 paragraphs
- Be concise but informative
- If information is unclear, say so rather than guessing
- Focus on the most important events mentioned in the data

RAW MATCH DATA:
{raw_data}

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

def generate_short_caption(frame_description):
    """Generate a short 1-2 line caption for the key frame"""
    prompt = f"""
Based on this detailed frame analysis, write a short 1-2 line caption that captures the main action happening in this football match moment.

Frame analysis: {frame_description}

Write a concise caption (maximum 2 lines) that describes the key action:
"""
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50
    }
    
    try:
        response = requests.post(DEPLOYMENT_URL, headers=HEADERS, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return "Key moment from the match"
    
    return "Key moment from the match"

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

def edit_article_with_prompt(original_article, user_prompt):
    """Edit the generated article based on user's custom prompt"""
    prompt = f"""
You are a professional sports news editor. Edit the following article based on the user's specific request.

ORIGINAL ARTICLE:
{original_article}

USER'S EDITING REQUEST:
{user_prompt}

KEY INSTRUCTIONS:
- Follow the user's editing request precisely
- Maintain the factual accuracy of the original content
- Keep it professional and news-appropriate
- If the request asks for information not in the original article, politely mention that the information is not available
- Return only the edited article, no additional commentary

EDITED ARTICLE:
"""
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 600
    }
    
    try:
        response = requests.post(DEPLOYMENT_URL, headers=HEADERS, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error editing article: {str(e)}")
    
    return "[Error editing article]"

# --- Main Execution ---
# Initialize session state for article storage
if 'generated_article' not in st.session_state:
    st.session_state.generated_article = None
if 'article_image_base64' not in st.session_state:
    st.session_state.article_image_base64 = None
if 'article_caption' not in st.session_state:
    st.session_state.article_caption = None

# Handle video upload
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
                
                # Store in session state with base64 image
                st.session_state.generated_article = article
                if global_best_frame:
                    # Convert image to base64 for persistent storage
                    image_base64 = image_to_base64(global_best_frame['image_path'])
                    st.session_state.article_image_base64 = image_base64
                    st.session_state.article_caption = generate_short_caption(global_best_frame['description'])
                
                # Cleanup audio file
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

# Handle raw text input
if generate_from_text and raw_match_data.strip():
    st.info("📝 Generating article from text data...")
    
    try:
        article = generate_article_from_text(raw_match_data)
        
        # Store in session state
        st.session_state.generated_article = article
        st.session_state.article_image_base64 = None
        st.session_state.article_caption = None
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

elif generate_from_text and not raw_match_data.strip():
    st.warning("Please enter some match data in the text area.")

# Display generated article and editor
if st.session_state.generated_article:
    st.subheader("📰 Generated News Article")
    
    # Display image if available
    if st.session_state.article_image_base64:
        try:
            # Decode base64 image and display
            image_data = base64.b64decode(st.session_state.article_image_base64)
            st.image(image_data, use_container_width=True)
            if st.session_state.article_caption:
                st.caption(st.session_state.article_caption)
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
    
    # Display the article
    st.write(st.session_state.generated_article)
    
    # Article Editor Section
    st.markdown("---")
    st.subheader("✏️ Edit Article")
    st.markdown("Want to modify the article? Enter your editing instructions below:")
    
    # Create columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        edit_prompt = st.text_area(
            "Enter your editing instructions:",
            placeholder="Examples:\n• Make it more formal\n• Add more details about the players\n• Make it shorter\n• Change the tone to be more exciting\n• Focus more on the team statistics\n• Rewrite the headline",
            height=100,
            key="edit_prompt"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        edit_button = st.button("🔄 Edit Article", type="primary")
        
        # Reset button to restore original
        if st.button("↩️ Reset to Original"):
            st.rerun()
    
    # Handle article editing
    if edit_button and edit_prompt.strip():
        with st.spinner("✏️ Editing article..."):
            try:
                edited_article = edit_article_with_prompt(st.session_state.generated_article, edit_prompt)
                
                # Update session state with edited article
                st.session_state.generated_article = edited_article
                
                # Show success message and rerun to display updated article
                st.success("Article updated successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error editing article: {str(e)}")
    
    elif edit_button and not edit_prompt.strip():
        st.warning("Please enter your editing instructions.")
    
    # Quick edit buttons for common requests
    st.markdown("**Quick Edit Options:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📏 Make Shorter"):
            with st.spinner("Making article shorter..."):
                edited_article = edit_article_with_prompt(st.session_state.generated_article, "Make this article shorter and more concise while keeping the key information.")
                st.session_state.generated_article = edited_article
                st.rerun()
    
    with col2:
        if st.button("🎯 More Formal"):
            with st.spinner("Making article more formal..."):
                edited_article = edit_article_with_prompt(st.session_state.generated_article, "Make this article more formal and professional in tone.")
                st.session_state.generated_article = edited_article
                st.rerun()
    
    with col3:
        if st.button("⚡ More Exciting"):
            with st.spinner("Making article more exciting..."):
                edited_article = edit_article_with_prompt(st.session_state.generated_article, "Make this article more exciting and engaging while keeping it factual.")
                st.session_state.generated_article = edited_article
                st.rerun()
    
    with col4:
        if st.button("📊 Add Stats Focus"):
            with st.spinner("Adding statistics focus..."):
                edited_article = edit_article_with_prompt(st.session_state.generated_article, "Focus more on statistics and numerical data if available in the original content.")
                st.session_state.generated_article = edited_article
                st.rerun()