"""
DeepFake EGC Project

This application provides a Gradio interface for processing DeepFake tasks.
Users can upload video or image files, input text for Text-to-Speech (TTS) synthesis,
and configure various parameters for processing. The application handles file uploads,
generates TTS audio, and sends requests to a processing container for further processing.
"""

import os
import shutil
import gradio as gr
import requests
import logging
from TTS.api import TTS
import torch
from time import time
from datetime import datetime
from threading import Thread

# Environment and logging configuration
os.environ["TTS_SKIP_TOS"] = "true"
logging.basicConfig(level=logging.INFO)
UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def generate_tts_audio(text, speaker_wav, tts_output_file, device):
    """
    Generate TTS audio from text and save it to a file.

    Args:
        text (str): The text to be converted to speech.
        speaker_wav (str): Path to a WAV file of the speaker's voice.
        tts_output_file (str): Path where the TTS output will be saved.
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        str: The path to the generated TTS audio file.
    """
    start_time = time()
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=True,
    ).to(device)
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        file_path=tts_output_file,
        language="fr",
    )
    end_time = time()
    logging.info(f"TTS audio generated in {end_time - start_time:.2f} seconds.")
    return tts_output_file

def process_video(task):
    """
    Process a single video task: handle file uploads, generate TTS audio if needed, and send data to the processing container.

    Args:
        task (dict): A dictionary containing task information.

    Yields:
        str: Status messages during the processing steps.
    """
    task_name = task["task_name"]
    logging.info(f"Task '{task_name}': Starting process for video and audio.")
    yield f"Task '{task_name}': Starting process."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Handle video file upload
    video_file = task["video_file"]
    if video_file:
        video_path = os.path.join(
            UPLOAD_DIR,
            f"{task_name}_video_{timestamp}{os.path.splitext(video_file.name)[1]}",
        )
        shutil.copy(video_file.name, video_path)
        logging.info(f"Task '{task_name}': Video file moved to: {video_path}")
        yield f"Task '{task_name}': Video file uploaded."
    else:
        logging.error(f"Task '{task_name}': No video file provided.")
        yield f"Task '{task_name}': No video file provided."
        task["status"] = "Error"
        return

    # Handle audio file upload or use video audio
    use_video_audio = task["use_video_audio"]
    audio_file = task["audio_file"]
    if not use_video_audio and audio_file:
        audio_path = os.path.join(
            UPLOAD_DIR,
            f"{task_name}_audio_{timestamp}{os.path.splitext(audio_file.name)[1]}",
        )
        shutil.copy(audio_file.name, audio_path)
        logging.info(f"Task '{task_name}': Audio file moved to: {audio_path}")
        yield f"Task '{task_name}': Audio file uploaded."
    elif use_video_audio:
        # Extract audio from the video using ffmpeg
        audio_path = os.path.join(UPLOAD_DIR, f"{task_name}_extracted_audio_{timestamp}.wav")
        cmd = f'ffmpeg -y -i "{video_path}" -q:a 0 -map a "{audio_path}"'
        os.system(cmd)
        logging.info(f"Task '{task_name}': Audio extracted from video to: {audio_path}")
        yield f"Task '{task_name}': Audio extracted from video."
    else:
        audio_path = None
        logging.info(f"Task '{task_name}': No audio file provided.")
        yield f"Task '{task_name}': No audio provided, skipping audio extraction."

    # Generate TTS audio if tts_text is provided
    tts_text = task["tts_text"]
    if tts_text.strip():
        if audio_path:
            speaker_wav = audio_path
        else:
            speaker_wav = None
            logging.warning(f"Task '{task_name}': No speaker audio provided. Using default voice.")
        tts_output_file = os.path.join(
            UPLOAD_DIR, f"{task_name}_tts_output_{timestamp}.wav"
        )
        generate_tts_audio(tts_text, speaker_wav, tts_output_file, device)
        audio_path = tts_output_file
        logging.info(f"Task '{task_name}': TTS audio generated and saved to: {tts_output_file}")
        yield f"Task '{task_name}': TTS audio generated."
    else:
        logging.info(f"Task '{task_name}': No TTS text provided. Using existing audio.")
        yield f"Task '{task_name}': Using existing audio."

    # Prepare the request payload for the processing container
    data = {
        "face": video_path,
        "audio": audio_path,
        "iterations": task["iterations"],
        "downscale_percentage": task["downscale_percentage"],
        "archive_folder": task["archive_folder"],
    }

    logging.info(f"Task '{task_name}': Sending request to processing container with payload: {data}")
    yield f"Task '{task_name}': Lip-syncing in progress"

    # Send request to the processing container
    try:
        response = requests.post("http://processing-container:5000/process", json=data)
        if response.status_code == 200:
            result = response.json()
            logging.info(
                f"Task '{task_name}': Processing completed successfully. Output: {result['output_file']}"
            )
            task["status"] = "Completed"
            task["output_file"] = result["output_file"]
            yield f"Task '{task_name}': Completed successfully."
        else:
            logging.error(f"Task '{task_name}': Error from processing container: {response.json()}")
            task["status"] = "Error"
            yield f"Task '{task_name}': {response.json().get('error', 'Unknown error')}"
    except Exception as e:
        logging.error(f"Task '{task_name}': Error while contacting processing container: {str(e)}")
        task["status"] = "Error"
        yield f"Task '{task_name}': {str(e)}"

def build_ui():
    """
    Build the Gradio UI for the DeepFake Processor application.

    Returns:
        gradio.Blocks: The Gradio interface.
    """
    with gr.Blocks(title="DeepFake EGC") as demo:
        gr.Markdown("# DeepFake EGC")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Task Configuration")
                gr.Markdown("Fill in the details below to configure your DeepFake task.")

                task_name = gr.Textbox(
                    label="Task Name",
                    placeholder="Enter a name for this task (e.g., 'MyFirstTask')"
                )
                video_file = gr.File(
                    label="Select Face File (MP4 or Image)",
                    file_types=["video", "image"]
                )
                gr.Markdown(
                    "Upload a video or image file containing the face to be used. "
                    "Supported formats: MP4 for videos, common image formats for images."
                )
                tts_text = gr.Textbox(
                    label="Enter Text for TTS",
                    placeholder="Enter text for TTS (optional)",
                    lines=5
                )
                gr.Markdown(
                    "Provide the text that will be converted to speech. "
                    "If left empty, the existing audio will be used."
                )
                gr.Examples(
                    examples=[
                        ["Bienvenue à cette conférence sur le partage de savoir"],
                        ["L'apprentissage automatique ouvre de nouvelles possibilités."]
                    ],
                    inputs=tts_text,
                    label="Example Texts for TTS",
                )
                use_video_audio = gr.Checkbox(
                    label="Use Audio from Video",
                    value=True
                )
                gr.Markdown(
                    "Check this box to use the audio from the uploaded video. "
                    "Uncheck to upload a separate audio file."
                )
                audio_file = gr.File(
                    label="Select Audio File",
                    file_types=["audio"],
                    visible=False
                )
                iterations = gr.Number(
                    label="Number of Iterations",
                    value=1,
                    precision=0,
                    minimum=1
                )
                gr.Markdown(
                    "Set the number of processing iterations to improve the quality. "
                    "Higher numbers may increase processing time."
                )
                downscale_percentage = gr.Slider(
                    label="Downscale Percentage",
                    minimum=10,
                    maximum=100,
                    value=100,
                    step=5
                )
                gr.Markdown(
                    "Adjust the downscale percentage to manage VRAM and RAM usage. "
                    "Lower values reduce resource consumption but may affect quality."
                )
                archive_folder = gr.Textbox(
                    label="Archive Folder Path",
                    placeholder="Enter the path to the archive folder",
                    value=UPLOAD_DIR
                )
                gr.Markdown(
                    "Specify the folder path where the processed files will be archived."
                )
                add_task_button = gr.Button("Add Task")
                start_processing_button = gr.Button("Start Processing Tasks")

            with gr.Column():
                gr.Markdown("## Task List and Output")
                task_list = gr.State([])
                task_list_display = gr.Dataframe(
                    headers=["Task Name", "Status"],
                    datatype=["str", "str"],
                    interactive=False,
                    value=[]
                )
                gr.Markdown(
                    "This table displays the list of tasks and their current status."
                )
                output_message = gr.Textbox(
                    label="Status",
                    value="Idle",
                    interactive=False,
                    lines=10
                )
                gr.Markdown(
                    "Shows the current status of the processing tasks."
                )
                output_files = gr.Files(
                    label="Download Output Files",
                    file_count="multiple"
                )
                gr.Markdown(
                    "Download the processed output files here."
                )
                output_video = gr.Video(
                    label="Output Video",
                    visible=False
                )  # Added video player component
                gr.Markdown(
                    "View the processed output video."
                )

        def toggle_audio_input(use_audio: bool):
            """
            Toggle the visibility of the audio file input based on the 'Use Audio from Video' checkbox.

            Args:
                use_audio (bool): Whether to use audio from the video.

            Returns:
                gradio.Update: Update object to modify the visibility of the audio_file component.
            """
            return gr.update(visible=not use_audio)

        use_video_audio.change(
            fn=toggle_audio_input, inputs=use_video_audio, outputs=audio_file
        )

        def add_task(
            name,
            video,
            text,
            use_audio,
            audio,
            iter_count,
            archive,
            downscale,
            current_tasks,
        ):
            """
            Add a new task to the task list.

            Args:
                name (str): Task name.
                video: Video file.
                text (str): Text for TTS.
                use_audio (bool): Whether to use audio from the video.
                audio: Audio file.
                iter_count (int): Number of iterations.
                archive (str): Archive folder path.
                downscale (int): Downscale percentage.
                current_tasks (list): Current list of tasks.

            Returns:
                tuple: Updated task list and updated task list display.
            """
            updated_tasks = current_tasks.copy()
            if not name:
                name = f"Task_{len(updated_tasks) + 1}"
            task = {
                "task_name": name,
                "video_file": video,
                "tts_text": text,
                "use_video_audio": use_audio,
                "audio_file": audio,
                "iterations": int(iter_count),
                "archive_folder": archive,
                "downscale_percentage": int(downscale),
                "status": "Pending",
                "output_file": None,
            }
            updated_tasks.append(task)
            return updated_tasks, gr.update(
                value=[[t["task_name"], t["status"]] for t in updated_tasks]
            )

        add_task_button.click(
            fn=add_task,
            inputs=[
                task_name,
                video_file,
                tts_text,
                use_video_audio,
                audio_file,
                iterations,
                archive_folder,
                downscale_percentage,
                task_list,
            ],
            outputs=[task_list, task_list_display],
        )

        def start_processing(task_list_input):
            """
            Start processing the tasks in the task list.

            Args:
                task_list_input (list): List of tasks to process.

            Yields:
                str: Status updates during processing.
                list: Updated list of output files.
                Update: Update object for the output_video component.
            """
            if not task_list_input:
                yield "No tasks to process.", [], gr.update()
                return

            output_files_list = []
            first_video_processed = False

            for task in task_list_input:
                if task["status"] == "Pending":
                    task["status"] = "Processing"
                    for update in process_video(task):
                        # During processing, yield status messages, no change to output_video
                        yield update, output_files_list, gr.update()
                    # After processing, collect the output file if available
                    if task["output_file"]:
                        output_files_list.append(task["output_file"])
                        if not first_video_processed:
                            first_video_processed = True
                            # After first task is completed, update output_video
                            yield f"Task '{task['task_name']}' completed.", output_files_list, gr.update(value=task["output_file"], visible=True)
                        else:
                            yield f"Task '{task['task_name']}' completed.", output_files_list, gr.update()
                    else:
                        yield f"Task '{task['task_name']}' completed but no output file.", output_files_list, gr.update()
            yield "All tasks completed.", output_files_list, gr.update()

        start_processing_button.click(
            fn=start_processing,
            inputs=[task_list],
            outputs=[output_message, output_files, output_video],  # Added output_video to outputs
        )

    return demo

# Initialize and launch the Gradio application
if __name__ == "__main__":
    ui = build_ui()
    ui.queue()  # Enable threading
    ui.launch(share=True, server_name="0.0.0.0", server_port=7860)
