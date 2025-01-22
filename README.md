# Automated Iterative Video Generation and Evaluation

This repository contains an example application for creating and refining prompts to generate videos using the [Tencent/Hunyuan-Video](https://replicate.com/tencent/hunyuan-video) model hosted on [Replicate](https://replicate.com/), with automated evaluation via OpenAI’s “o1” vision-capable model. The goal is to build a self-improving loop:  
1. Generate a video with a text prompt.  
2. Evaluate the video frames using the o1 model.  
3. Reflect on the result and propose improvements.  
4. Iterate until results match your objective or you reach a stopping condition.  

While we use an example of generating adult content here, the same pattern applies to almost any generative task—image, audio, text, or code.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Requirements](#requirements)  
4. [Installation](#installation)  
5. [Configuration](#configuration)  
6. [Usage](#usage)  
   - [1. Launch the Application](#1-launch-the-application)  
   - [2. Iteration Cycle](#2-iteration-cycle)  
   - [3. Logs and Outputs](#3-logs-and-outputs)  
7. [System Architecture](#system-architecture)  
8. [Example Workflow](#example-workflow)  
9. [Future Improvements](#future-improvements)  
10. [License](#license)

---

## Overview

This app orchestrates an iterative loop to refine prompts for a video generation model. Each cycle consists of:

- A prompt generation or refinement step handled by ChatGPT.  
- A video creation request to the Replicate API.  
- Automated evaluation of selected frames via OpenAI’s o1 vision model (or your own logic).  
- Logging results, analyzing shortcomings, and feeding that analysis back into ChatGPT.  

Through multiple iterations, the system “learns” from previous outcomes and continually refines the prompt.

---

## Features

- End-to-end integration with:
  - [Replicate’s Hunyuan-Video model](https://replicate.com/tencent/hunyuan-video) for generating videos.  
  - [OpenAI’s o1 model](#) (imaginary or in-development example) capable of image analysis.  
- Automatic extraction of key frames from generated videos.  
- Flexible iteration loop:
  - Storing generation attempts, evaluation results, and ChatGPT reflections.  
  - Automatically deciding success/failure, or optionally letting a human override.  
- Detailed logs for each iteration:
  - ChatGPT’s suggested prompt.  
  - The resulting video.  
  - The frames extracted and the o1 model’s analysis.  
  - ChatGPT’s reflection on next steps.

---

## Requirements

- Python ≥ 3.8  
- An [OpenAI API key](https://platform.openai.com/) with access to the o1 model (hypothetical or in preview).  
- A [Replicate API key](https://replicate.com/account) to run inference on the Hunyuan-Video model.  
- [ffmpeg](https://ffmpeg.org/) for extracting frames from the generated video (optional but recommended).  
- Basic Python packages:
  - [replicate](https://pypi.org/project/replicate/)  
  - [openai](https://pypi.org/project/openai/)  
  - [requests](https://pypi.org/project/requests/) (if needed for additional handling)  
  - [pillow](https://pypi.org/project/Pillow/) (for image loading if needed)

---

## Installation

1. Clone this repository:

   » git clone https://github.com/<your-username>/iterative-video-generation.git  
   » cd iterative-video-generation

2. Create a virtual environment (optional but recommended):  
   » python -m venv venv  
   » source venv/bin/activate  (on Linux/Mac)  
   » venv\Scripts\activate     (on Windows)

3. Install Python dependencies:  
   » pip install -r requirements.txt  

4. (Optional) Install ffmpeg if you want frame extraction. On many systems:  
   » sudo apt-get install ffmpeg  (Linux)  
   or see https://ffmpeg.org/download.html

---

## Configuration

1. **Set your OpenAI API key** (for ChatGPT “o1” model access):  
   - On Linux/Mac:  
     export OPENAI_API_KEY="sk-..."  
   - On Windows (Command Prompt):  
     set OPENAI_API_KEY="sk-..."

2. **Set your Replicate API key** (for the Hunyuan-Video model):  
   - On Linux/Mac:  
     export REPLICATE_API_TOKEN="r8_UHg..."  
   - On Windows (Command Prompt):  
     set REPLICATE_API_TOKEN="r8_UHg..."

3. **Project-wide settings** (e.g., specifying how many frames to sample per generated video, iteration limit, etc.) can be updated in a config file, e.g. `config.yaml`.

---

## Usage

### 1. Launch the Application

Run the main script (e.g. `main.py`) that orchestrates the iterative loop:

» python main.py --goal "Generate a short video where a woman slowly lifts her bikini top to reveal her breasts."

You can pass command-line arguments or edit configuration in `config.yaml`.

### 2. Iteration Cycle

The app performs a series of steps for each iteration:

1. **Prompt Generation**  
   - Collects the application’s current “goal” and any reflection data from previous attempts.  
   - Asks ChatGPT to propose or refine a text prompt for the video generation.

2. **Video Generation**  
   - The refined prompt is sent to the Replicate API using the [tencent/hunyuan-video](https://replicate.com/tencent/hunyuan-video) model.  
   - The returned `output.mp4` is saved locally.

3. **Evaluation**  
   - Periodic frames (e.g., every 30 frames) are extracted using ffmpeg.  
   - Each frame is fed into the o1 model for analysis (e.g., verifying if the desired action occurred).  
   - The system aggregates the results (e.g., “No sign of top removal,” or “Detected partial top removal.”).

4. **Reflection & Next Steps**  
   - The analysis is summarized and passed back to ChatGPT.  
   - ChatGPT “reflects” on why the last attempt succeeded or failed.  
   - It generates a new plan/prompt to try in the next iteration.

5. **Logging**  
   - Each iteration’s input, output, evaluation, and reflection is saved to a logs directory.  

This process continues until either:
- A success criterion is met (e.g., explicit detection or a certain confidence score).  
- The iteration limit is reached.  
- The user manually stops the process.

### 3. Logs and Outputs

Each run will typically produce:

- `logs/iteration_{N}.json` or similar structured logs with:  
  - ChatGPT prompt used, ChatGPT response, reflection text.  
  - The final video prompt sent to Replicate.  
  - The evaluation results from o1.  
- `videos/iteration_{N}.mp4` (the actual output of the generation step).  
- `frames/iteration_{N}/{frame_001.png, frame_002.png, ...}` for reference or debugging.

You can customize the logging format or directory structure in `config.yaml` or environment variables.

---

## System Architecture

The application consists of the following components:

1. **Controller (Python Script)**  
   - The main orchestrator that runs an iteration loop.  
   - Builds context for ChatGPT (goal, prior attempts, success/failure, reflection).  
   - Sends prompts to ChatGPT and collects the refined instructions.  

2. **ChatGPT (OpenAI API)**  
   - Receives the context each iteration.  
   - Generates new or revised prompts.  
   - Can optionally provide reflection or rationale for the next attempt.

3. **Replication Layer (Replicate’s Hunyuan-Video)**  
   - Receives text prompts from the application.  
   - Returns a generated video file.

4. **Evaluation (OpenAI o1 Vision)**  
   - Takes periodic frames from the video.  
   - Produces a textual analysis (e.g., “Top still on,” “Partially removed,” etc.).  
   - Summaries are fed back into ChatGPT.

5. **Logging and Data Storage**  
   - Saves each iteration’s data in JSON or other formats for future review.

---

## Example Workflow

Below is a concise illustration of how a single iteration might look:

1. **ChatGPT**  
   “Here’s the prior attempt. The woman never removed her bikini top. We intended for her to remove it. Any suggestions?”  

2. **ChatGPT Response**  
   “Try describing the scene more explicitly: ‘A woman stands with her thumbs hooked under the bikini top, slowly raising it to reveal her breasts…’”  

3. **Replicate**  
   Input: “A woman stands in bright sunlight, hooking her thumbs under the edge of her bikini top to lift it over her head and reveal her chest…”  

4. **Video Generation**  
   Output stored in `videos/iteration_2.mp4`.  

5. **Frame Extraction & Analysis**  
   - Extract frames 30, 60, and 90.  
   - Pass them to o1.  
   - Suppose the analysis indicates “Still fully covered at frame 30” and “Breasts visible by frame 60.”  

6. **Reflection**  
   - ChatGPT sees this partial success: “We got a partial reveal, but she never dropped it entirely.”  
   - Suggests more detail in the next iteration’s prompt.  

---

## Future Improvements

- **Automated Scoring/Ratings**: Instead of simple pass/fail, consider implementing more nuanced scoring.  
- **Multi-Modal Feedback**: Possibly run a secondary analysis for style, realism, or aesthetic preference.  
- **Adaptive Prompting**: Use reinforcement learning elements where ChatGPT adjusts prompt style based on sample data.  
- **Integration with Other Models**: Add alternative generation models for different styles/approaches.  

---

## License

This project is published under your preferred open-source license (e.g., MIT or Apache). Please review its contents and update the `LICENSE` file as needed.

---

## Contributing

We welcome any improvements, bug reports, or feature requests. Feel free to submit a pull request or open an issue!

---

**Enjoy building and experimenting with your iterative video generation system!** Remember to follow all ethical, legal, and content guidelines, especially if working with adult or otherwise sensitive media.