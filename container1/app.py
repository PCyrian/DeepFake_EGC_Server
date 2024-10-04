import os
import shutil
import gradio as gr
import requests
import logging
from TTS.api import TTS
import torch
from time import time

os.environ["TTS_SKIP_TOS"] = "true"
logging.basicConfig(level=logging.INFO)
UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def generate_tts_audio(text, speaker_wav, tts_output_file, device):
    start_time = time()
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, file_path=tts_output_file, language='fr')
    end_time = time()
    logging.info(f"TTS audio generated in {end_time - start_time:.2f} seconds.")
    return tts_output_file

# Function to process video and send it to the processing container
def process_video(video_file, tts_text, use_video_audio, audio_file, iterations, archive_folder, downscale_percentage,
                  task_name):
    logging.info(f"Task '{task_name}': Starting process for video and audio.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Handle video file upload
    if video_file:
        video_path = os.path.join(UPLOAD_DIR, f"{task_name}_video{os.path.splitext(video_file.name)[1]}")
        shutil.copy(video_file.name, video_path)
        logging.info(f"Task '{task_name}': Video file moved to: {video_path}")
    else:
        logging.error(f"Task '{task_name}': No video file provided.")
        return {"status": "error", "message": "No video file provided."}

    # Handle audio file upload or use video audio
    if not use_video_audio and audio_file:
        audio_path = os.path.join(UPLOAD_DIR, f"{task_name}_audio{os.path.splitext(audio_file.name)[1]}")
        shutil.copy(audio_file.name, audio_path)
        logging.info(f"Task '{task_name}': Audio file moved to: {audio_path}")
    elif use_video_audio:
        # Extract audio from video
        audio_path = os.path.join(UPLOAD_DIR, f"{task_name}_extracted_audio.wav")
        cmd = f"ffmpeg -y -i \"{video_path}\" -q:a 0 -map a \"{audio_path}\""
        os.system(cmd)
        logging.info(f"Task '{task_name}': Audio extracted from video to: {audio_path}")
    else:
        audio_path = None
        logging.info(f"Task '{task_name}': No audio file provided.")

    # Generate TTS audio if tts_text is provided
    if tts_text.strip():
        if audio_path:
            # Use the provided audio as speaker reference
            speaker_wav = audio_path
        else:
            # Use a default speaker or handle the case accordingly
            speaker_wav = None
            logging.warning(f"Task '{task_name}': No speaker audio provided. Using default voice.")
        tts_output_file = os.path.join(UPLOAD_DIR, f"{task_name}_tts_output.wav")
        generate_tts_audio(tts_text, speaker_wav, tts_output_file, device)
        # Use the TTS output as the audio input for processing
        audio_path = tts_output_file
        logging.info(f"Task '{task_name}': TTS audio generated and saved to: {tts_output_file}")
    else:
        logging.info(f"Task '{task_name}': No TTS text provided. Using existing audio.")

    # Prepare the request payload for the processing container
    data = {
        "face": video_path,
        "audio": audio_path,
        "iterations": iterations,
        "downscale_percentage": downscale_percentage,
        "archive_folder": archive_folder
    }

    logging.info(f"Task '{task_name}': Sending request to processing container with payload: {data}")

    # Send request to the processing container
    try:
        response = requests.post("http://processing-container:5000/process", json=data)
        if response.status_code == 200:
            result = response.json()
            logging.info(f"Task '{task_name}': Processing completed successfully. Output: {result['output_file']}")
            return {"status": "success", "output_file": result["output_file"]}
        else:
            logging.error(f"Task '{task_name}': Error from processing container: {response.json()}")
            return {"status": "error", "message": response.json().get('error', 'Unknown error')}
    except Exception as e:
        logging.error(f"Task '{task_name}': Error while contacting processing container: {str(e)}")
        return {"status": "error", "message": str(e)}

# Build UI and launch Gradio app
def build_ui():
    with gr.Blocks(title="DeepFake Processor") as demo:
        gr.Markdown("# DeepFake Processor")

        with gr.Row():
            with gr.Column():
                task_name = gr.Textbox(label="Task Name", placeholder="Enter a name for this task")
                video_file = gr.File(label="Select Face File (MP4 or Image)", file_types=['video', 'image'])
                tts_text = gr.Textbox(label="Enter text for TTS", placeholder="Enter text for TTS", lines=5)
                use_video_audio = gr.Checkbox(label="Use Audio from Video", value=True)
                audio_file = gr.File(label="Select Audio File", file_types=['audio'], visible=False)
                iterations = gr.Number(label="Number of Iterations", value=1, precision=0, minimum=1)
                downscale_percentage = gr.Slider(label="Downscale Percentage", minimum=10, maximum=100, value=100,
                                                 step=1)
                archive_folder = gr.Textbox(label="Archive Folder Path",
                                            placeholder="Enter the path to the archive folder", value=UPLOAD_DIR)
                add_task_button = gr.Button("Add Task")
                start_processing_button = gr.Button("Start Processing Tasks")

            with gr.Column():
                task_list = gr.State([])
                task_list_display = gr.Dataframe(headers=["Task Name", "Status"], datatype=["str", "str"],
                                                 interactive=False, value=[])
                output_message = gr.Textbox(label="Status", interactive=False)
                output_video = gr.Video(label="Output Video")
                output_files = gr.File(label="Download Output Files", file_count="multiple")

        def toggle_audio_input(use_audio: bool):
            return gr.update(visible=not use_audio)

        use_video_audio.change(fn=toggle_audio_input, inputs=use_video_audio, outputs=audio_file)

        def add_task(name, video, text, use_audio, audio, iter_count, archive, downscale, current_tasks):
            updated_tasks = current_tasks.copy()
            if not name:
                name = f"Task_{len(updated_tasks) + 1}"
            task = {
                'task_name': name,
                'video_file': video,
                'tts_text': text,
                'use_video_audio': use_audio,
                'audio_file': audio,
                'iterations': int(iter_count),
                'archive_folder': archive,
                'downscale_percentage': int(downscale),
                'status': 'Pending'
            }
            updated_tasks.append(task)
            return updated_tasks, gr.update(value=[[t['task_name'], t['status']] for t in updated_tasks])

        add_task_button.click(
            fn=add_task,
            inputs=[task_name, video_file, tts_text, use_video_audio, audio_file, iterations, archive_folder,
                    downscale_percentage, task_list],
            outputs=[task_list, task_list_display]
        )

        def start_processing(task_list_input):
            if not task_list_input:
                return task_list_input, "No tasks to process.", None, gr.update(), None

            accumulated_output_files = []
            for task in task_list_input:
                task_name = task['task_name']
                response = process_video(
                    task['video_file'], task['tts_text'], task['use_video_audio'], task['audio_file'],
                    task['iterations'], task['archive_folder'], task['downscale_percentage'], task_name
                )

                if response["status"] == "success":
                    output_file = response["output_file"]
                    accumulated_output_files.append(output_file)
                    task['status'] = "Completed"
                    yield task_list_input, f"Task '{task_name}': Completed successfully.", output_file, gr.update(
                        value=[[t['task_name'], t['status']] for t in task_list_input]), accumulated_output_files
                else:
                    task['status'] = f"Error: {response['message']}"
                    yield task_list_input, f"Task '{task_name}': Failed. {response['message']}", None, gr.update(
                        value=[[t['task_name'], t['status']] for t in task_list_input]), accumulated_output_files

        start_processing_button.click(fn=start_processing, inputs=[task_list],
                                      outputs=[task_list, output_message, output_video, task_list_display,
                                               output_files])

    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=True, server_name="0.0.0.0", server_port=7860)
