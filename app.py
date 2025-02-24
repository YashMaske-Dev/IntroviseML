from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import concurrent.futures
import re
import gdown

app = Flask(__name__)
CORS(app)

# Set the path for your face model file
FER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_model.h5")

# Global variables for lazy-loaded models
emotion_model = None
whisper_model = None

def get_emotion_model():
    """Lazy-load the emotion detection model."""
    global emotion_model
    if emotion_model is None:
        from keras.models import load_model  # Lazy import
        emotion_model = load_model(FER_MODEL_PATH)
    return emotion_model

def get_whisper_model():
    """Lazy-load the Whisper model."""
    global whisper_model
    if whisper_model is None:
        import whisper  # Lazy import
        whisper_model = whisper.load_model("medium")
    return whisper_model

def download_video(drive_link, save_path="video.mp4"):
    """Download a video from a Google Drive link."""
    gdown.download(drive_link, save_path, quiet=True)
    return save_path

def extract_audio(video_path):
    """Extract audio from a video using MoviePy."""
    from moviepy.editor import VideoFileClip  # Lazy import
    audio_path = "extracted_audio.wav"
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)
    return audio_path

def transcribe_audio(audio_path, language):
    """Transcribe audio using the lazy-loaded Whisper model."""
    model = get_whisper_model()
    result = model.transcribe(audio_path, language=language)
    return result.get("text", "")

def filter_english_words(text):
    """Keep only English words from the text."""
    return " ".join(re.findall(r'\b[a-zA-Z]+\b', text))

def check_answer_correctness(transcribed_text, expected_keywords):
    """Compute correctness percentage based on keyword matches."""
    transcribed_text = filter_english_words(transcribed_text)
    matched_keywords = [word for word in expected_keywords if word.lower() in transcribed_text.lower()]
    correctness = (len(matched_keywords) / len(expected_keywords)) * 100 if expected_keywords else 0
    return correctness

def analyze_video(video_path):
    """Analyze video for engagement and emotion detection."""
    # Lazy-load heavy modules
    import cv2  # OpenCV
    import numpy as np
    import mediapipe as mp
    from collections import Counter

    model = get_emotion_model()  # Lazy-load emotion model
    EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    )
    cap = cv2.VideoCapture(video_path)
    engagement_frames, total_frames = 0, 0
    emotions_detected = []
    frame_skip = 5
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame_index % frame_skip != 0:
            continue

        total_frames += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            engagement_frames += 1
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_frame, (48, 48)) / 255.0
            emotion_prediction = model.predict(np.expand_dims(resized_face, axis=0))[0]
            detected_emotion = EMOTION_LABELS[np.argmax(emotion_prediction)]
            emotions_detected.append(detected_emotion)
    cap.release()

    engagement_ratio = engagement_frames / total_frames if total_frames > 0 else 0
    nervousness_score = (
        "Low" if engagement_ratio > 0.7 else ("Moderate" if engagement_ratio > 0.4 else "High")
    )
    most_frequent_emotion = (
        Counter(emotions_detected).most_common(1)[0][0] if emotions_detected else "Neutral"
    )
    return {
        "Overall Confidence Level": "High" if engagement_ratio > 0.7 else ("Moderate" if engagement_ratio > 0.4 else "Low"),
        "Nervousness": nervousness_score,
        "Engagement Level": "High" if engagement_ratio > 0.7 else ("Moderate" if engagement_ratio > 0.4 else "Low"),
        "Most Frequent Emotion": most_frequent_emotion
    }

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    video_link = data.get("video_link")
    language = data.get("language", "english")
    expected_keywords = data.get("keywords", [])

    if not video_link:
        return jsonify({"error": "Missing video link"}), 400

    video_path = download_video(video_link)

    # Process audio and video concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        audio_future = executor.submit(extract_audio, video_path)
        video_future = executor.submit(analyze_video, video_path)
        audio_path = audio_future.result()
        transcribed_text = transcribe_audio(audio_path, language)
        correctness = check_answer_correctness(transcribed_text, expected_keywords)
        video_analysis = video_future.result()

    result = {
        "Audio Analysis": {
            "Transcription": transcribed_text,
            "Correctness": correctness
        },
        "Video Analysis": video_analysis
    }
    return jsonify(result)

# For Vercel, expose the WSGI application
def handler(event, context):
    return app(event, context)

if __name__ == "__main__":
    # Run locally for testing; Vercel will use the exposed handler
    app.run(host="0.0.0.0", port=5000)
