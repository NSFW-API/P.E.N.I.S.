# P.E.N.I.S. (Prompt-based Evolutionary Nudity Iteration System)
_by NSFW API_

Welcome to the Prompt-based Evolutionary Nudity Iteration System (P.E.N.I.S.). This experimental application leverages automated prompt-refinement and video generation to iteratively achieve a specific adult-oriented goal—such as depicting a woman removing her top in a short video. The system orchestrates calls to both a text-based LLM (e.g., an OpenAI GPT-style model) and the Replicate Hunyuan-Video model, extracting frames to evaluate whether required visual elements are successfully achieved, and improving the prompts in successive iterations.

IMPORTANT: This application is intended for adult or NSFW content generation tasks. It contains code that demonstrates how to automatically refine prompts for explicit scenarios. Please use responsibly, ensure you comply with local laws and platform policies, and obtain proper consent when dealing with adult content.

---

## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [How it Works](#how-it-works)  
4. [Installation and Setup](#installation-and-setup)  
5. [Configuration](#configuration)  
6. [Running the Application](#running-the-application)  
7. [Outputs and Logs](#outputs-and-logs)  
8. [Extending or Customizing](#extending-or-customizing)  
9. [Important Notes and Disclaimers](#important-notes-and-disclaimers)

---

## 1. Overview

P.E.N.I.S. aims to repeatedly generate a custom short video with specific adult (NSFW) attributes, evaluating each generation pass to check whether the visual “required elements” are present. When the target conditions are unmet, it automatically refines the prompt and tries again, until it reaches success or hits a maximum iteration limit.

You can use this to experiment with:
- Iterative prompt refinement.  
- Automated retrieval of frames from the output video.  
- Frame-level image analysis via GPT for presence/absence checks.  

Please be aware that this demo is specifically oriented toward generating adult content. The underlying approach, however, can be applied to many other generative tasks (e.g., non-NSFW scenarios) simply by adjusting the goals and disclaimers.

---

## 2. Features

• Iterative Prompt Refinement:  
  - The system reads your original goal (e.g., “A woman slowly lifts her top”) and extracts each important element.  
  - It then crafts a prompt for Replicate’s Hunyuan-Video model, including user-defined disclaimers or special instructions.  

• Automatic Video Generation (via Replicate):  
  - Uses an open-source or proprietary text-to-video model.  
  - Configurable resolution, FPS, video length, etc.  

• Frame Extraction and Analysis:  
  - After each iteration, it extracts frames at a configurable interval using ffmpeg.  
  - Those frames are sent to a GPT-based vision model to judge whether the key elements are present.  

• Self-Improvement Loop:  
  - If the element presence checks fail, the system refines the prompt and tries again.  
  - Continues until success or a maximum iteration count.  

• Detailed Logging:  
  - Creates iteration-specific logs, storing each refined prompt, notes, and presence checks.  
  - Stores output videos and extracted frames in an organized folder hierarchy.  

---

## 3. How it Works

1. **Goal Extraction**  
   - The system first asks GPT to parse your textual goal into discrete required elements (e.g., “top removal must be visible,” “focus on the torso,” etc.).  

2. **Prompt Creation**  
   - A refined, unified prompt is generated based on the user’s goal and past iteration successes/failures.  
   - GPT decides on a suitable resolution (width × height within 100–512).  

3. **Video Generation**  
   - The refined prompt is submitted to Replicate’s Hunyuan-Video model.  
   - The model outputs a short MP4 video saved in your runs directory.  

4. **Frame Extraction**  
   - The system uses ffmpeg to extract frames (e.g., every 30th frame).  

5. **Evaluation**  
   - A GPT-based vision model inspects each frame to see if the required elements are visibly satisfied (e.g., is the top truly removed?).  

6. **Loop or Terminate**  
   - If all elements are present, it stops. Otherwise, it refines the prompt again, trying to correct what’s missing. This continues until success or max iterations.

---

## 4. Installation and Setup

1. **Clone the Repository**  
   - Clone or fork the code into your local environment.  

2. **Create and Activate a Virtual Environment** (optional but recommended):  
   » python -m venv venv  
   » source venv/bin/activate (on Linux/Mac) or .\venv\Scripts\activate (on Windows)  

3. **Install Dependencies**  
   - Ensure you have Python 3.9+ installed, then:  
     » pip install -r requirements.txt  

4. **Environment Variables**  
   - You will need valid tokens for OpenAI and Replicate.  
   - Create a .env file or set environment variables:  
     OPENAI_API_KEY=<your_openai_key>  
     REPLICATE_API_TOKEN=<your_replicate_token>  

5. **ffmpeg**  
   - Make sure ffmpeg is installed and available on your PATH (necessary for frame extraction).  
   - On many systems, you can install via:  
     - Linux (Debian/Ubuntu): apt-get install ffmpeg  
     - Mac (Homebrew): brew install ffmpeg  
     - Windows: Use a prebuilt ffmpeg from https://ffmpeg.org/download.html  

---

## 5. Configuration

All major settings are found in “config.yaml”. Common fields:

• openai:  
  - model_name: "gpt-4" or "gpt-4-vision" (depending on availability).  
  - max_completion_tokens: 2000 (or your desired limit).  

• replicate:  
  - model_name: "tencentarc/hunyuan-video" (or similar text-to-video checkpoint).  

• local_comfyui:  
  - config.yaml (for ComfyUI settings).

• frames:  
  - extract_interval: 30 (number of frames between each extraction).  

• iterations:  
  - max_iterations: 5 (maximum times the system tries refining).  

• runs_directory: "runs" (where iteration logs, videos, and frames are stored).  

Adjust these values as needed.  

---

## 6. Running the Application

From the project’s root directory, simply run:  
default replcate engine
» python main.py --goal "A woman removing her top" --run_name "demo_run"   
local comfyui engine
» python main.py --goal "A woman removing her top" --run_name "demo_run" --gen_engine local_comfyui

- “--goal” is your high-level scenario or user request.  
- “--run_name” is an optional label for this run; if omitted, the system uses a timestamp. 
- "--gen_engine" can be local_comfyui or replicate, default is the default engine for replicate. 

When run, the system will:  
1. Create a new subfolder in “runs/” (or your configured runs_directory).  
2. Parse your goal with GPT to identify required elements.  
3. Generate an initial prompt.  
4. Call Replicate to create a short MP4 video.  
5. Extract frames and evaluate them with GPT’s vision model.  
6. Continue refining until success or until the maximum iteration limit.  

---

## 7. Outputs and Logs

Inside the newly created “runs/<run_name>” folder, you’ll find:

• logs/:  
  - JSON files named “iteration_1.json”, “iteration_2.json”, etc., each containing the prompt, notes, presence checks, etc.  
  - final_summary.txt containing an overview of the final outcome.  

• frames/:  
  - Subfolders like “iteration_unified_1”, containing extracted frames from each iteration.  

• videos/:  
  - MP4 files named “iteration_unified_1.mp4”, “iteration_unified_2.mp4”, etc.  

Review the logs to see how the system refined the prompt each time, what the GPT-based vision model concluded about the frames, and whether or not it satisfied all required elements.

---

## 8. Extending or Customizing

• Model Swaps:  
  - In “config.yaml,” swap out “replicate/model_name” if you want a different text-to-video model on Replicate.  

• Non-NSFW Scenarios:  
  - Despite the name, you can adapt the system to handle safer goals (e.g., generating short animations for non-adult tasks).  

• Additional Elements or Custom Evaluations:  
  - For more advanced checks, you can refine the logic in “src/evaluation.py” to parse bounding boxes, detected objects, or more nuanced element requirements.  

• Prompt Refinement Logic:  
  - Adjust how “src/chatgpt_utils.refine_unified_prompt” constructs the JSON used by your text-to-video model.  
  - You can incorporate style prompts, disclaimers, or relevant textual instructions to better shape the final video generation.  

---

## 9. Important Notes and Disclaimers

1. **Adult Content**  
   - This application is explicitly intended for adult (NSFW) video generation. If you do not wish to produce such content, please modify the goals accordingly.  

2. **Usage and Consent**  
   - Always comply with local laws and ethical guidelines when generating explicit content or handling user requests.  

3. **OpenAI and Replicate Policies**  
   - Your usage of OpenAI’s GPT models and Replicate’s video-model endpoints is subject to their respective terms of service. The code as provided here does not guarantee compliance—please ensure your usage follows all relevant terms.  

4. **Experimental Quality**  
   - Generated content (video) can be inconsistent or low fidelity, especially for nuanced instructions. Expect to iterate and experiment frequently.  

5. **No Minors, No Non-consensual Depictions**  
   - The system is not intended to create or depict minors, non-consensual acts, or other disallowed content. Please use responsibly and ethically.  

---

## Contributing

Contributions are welcome for improvements, bug fixes, or extending the system’s capabilities to additional generative tasks. If you find issues or would like to suggest a feature, feel free to open a pull request.

---

## License

This project is made available under a permissible open-source license. See the included LICENSE file for more details.

---

Thank you for exploring the Prompt-based Evolutionary Nudity Iteration System (P.E.N.I.S.). Use it responsibly, and have fun experimenting with iterative prompt engineering in the NSFW domain!