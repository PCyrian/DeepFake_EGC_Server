
# DeepFake_EGC_Server

---

## Overview
**DeepFake_EGC_Server** is a deepfake generation server that allows users to manipulate video content with advanced features, including text-to-speech, lipsyncing, and resolution scaling. Built with a containerized architecture for ease of deployment, it leverages GPU acceleration for efficient processing.

## Key Features
1. **Gradio Interface**: User-friendly interface for configuring and managing deepfake tasks, accessible via a web browser.
   
2. **Text-to-Speech Integration**: Converts text input into synthesized speech, which can be applied to the generated deepfake video.

3. **Lipsyncing with video-retalking**: Incorporates advanced lipsync technology using the [video-retalking](https://github.com/OpenTalker/video-retalking) repository to match speech with facial movements.

4. **Downscale Video Resolution**: Adjustable resolution downscaling to optimize output file size and performance.

## Prerequisites
- **Docker Desktop** (Windows) or **Docker Compose** (Linux)
- **NVIDIA Docker Runtime** for GPU acceleration

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/PCyrian/DeepFake_EGC_Server.git
   ```
2. Navigate to the project directory:
   ```bash
   cd DeepFake_EGC_Server
   ```
3. Build and run the container:
   ```bash
   docker-compose up --build
   ```

## Usage
Once the server is running, access it through the browser via the provided Gradio interface. Configure your deepfake task, then queue and process it using the available options.

## Dependencies
- **Docker** & **Docker Compose**: Required for managing the containerized environment.
- **NVIDIA Drivers**: Required on the host machine for GPU access, but CUDA libraries are handled within the container.

## Additional Features
- **File Upload Support**: Upload face video files in MP4 or image formats.
- **Audio Source Options**: Use either text-to-speech or original video audio.
- **Task Management**: Queue multiple tasks with real-time status tracking and download options for output files.

---