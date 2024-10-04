
### Updated api.py (container2)
from flask import Flask, request, jsonify
import os
import subprocess
import logging
from moviepy.editor import VideoFileClip

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

UPLOAD_DIR = "/app/uploads"

@app.route('/process', methods=['POST'])
def process_video():
    face_video = request.json.get("face")
    audio_file = request.json.get("audio")

    logging.info(f"Received face video path: {face_video}")
    logging.info(f"Received audio file path: {audio_file}")
    logging.info(f"Files in upload directory: {os.listdir(UPLOAD_DIR)}")

    if not face_video or not os.path.exists(face_video):
        logging.error(f"Face video file not provided or doesn't exist at path: {face_video}")
        return jsonify({"error": "Face video file not provided or doesn't exist."}), 400

    # If no audio file is provided, extract the audio from the video
    if not audio_file:
        audio_file = extract_audio_from_video(face_video)
        if not audio_file or not os.path.exists(audio_file):
            logging.error(f"Failed to extract audio from the video: {face_video}")
            return jsonify({"error": "Failed to extract audio from the video."}), 400

    output_file = "/app/uploads/output.mp4"

    cmd = [
        "python3", "inference.py",
        "--face", face_video,
        "--audio", audio_file,
        "--outfile", output_file
    ]

    try:
        logging.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logging.info(f"Processing completed successfully. Output file saved at: {output_file}")
        return jsonify({"message": "Processing completed", "output_file": output_file})
    except subprocess.CalledProcessError as e:
        logging.error(f"Processing failed: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

def extract_audio_from_video(video_path):
    """Extract audio from the video file."""
    try:
        logging.info(f"Extracting audio from video: {video_path}")
        clip = VideoFileClip(video_path)
        audio_path = video_path.replace(".mp4", "_extracted.wav")
        clip.audio.write_audiofile(audio_path)
        logging.info(f"Audio extracted and saved at: {audio_path}")
        return audio_path
    except Exception as e:
        logging.error(f"Error extracting audio from video {video_path}: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)