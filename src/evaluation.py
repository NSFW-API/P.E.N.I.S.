import base64
import json
import os
import subprocess
from json import JSONDecodeError

from dotenv import load_dotenv
from openai import OpenAI


###############################################################################
# Utility: Frame Extraction
###############################################################################
def extract_frames(video_path, interval, output_dir):
    """
    Extract frames from the video every 'interval' frames using ffmpeg.
    Example command:
      ffmpeg -i video.mp4 -vf "select='not(mod(n,30))'" -vsync 0 out%03d.png
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"select='not(mod(n,{interval}))'",
        "-vsync",
        "0",
        os.path.join(output_dir, f"{basename}_%03d.png")
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


###############################################################################
# Utility: Encode image as Base64 Data URL
###############################################################################
def encode_image_as_data_url(image_path):
    """
    Convert an image file to a base64 data URI string.
    """
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


###############################################################################
# Main: Evaluate if All Required Elements Are Present
###############################################################################
def evaluate_all_elements(video_path, iteration_name, config, user_goal, required_elements, prev_presence=None):
    """
    1) Extract frames from the given video at a configurable interval.
    2) Pass them + the required_elements list into GPT for a presence/absence check.
    3) Return:
        - details: A short textual summary of which elements are present/missing
        - all_satisfied: Bool indicating if *all* required elements are present
        - presence_dict: Dict mapping element_id => True/False
        - notes: GPT's textual commentary or explanation

    Example GPT response structure:
        {
          "results": [
            {"id": "foot_contact", "present": true},
            {"id": "close_up_view", "present": false}
          ],
          "notes": "The foot isn't clearly visible, just a blur at the edge..."
        }
    """

    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = config["openai"]["model_name"]
    max_tokens = config["openai"].get("max_completion_tokens", 2000)

    # 1) Extract frames
    frames_parent_dir = config["frames"]["output_directory"]
    iteration_frames_dir = os.path.join(frames_parent_dir, f"iteration_{iteration_name}")
    os.makedirs(iteration_frames_dir, exist_ok=True)

    frame_interval = config["frames"]["extract_interval"]
    extract_frames(video_path, frame_interval, iteration_frames_dir)

    # 2) Construct the GPT prompt
    # We'll pass the user_goal and each required element as text, plus the extracted frames as data URLs.

    # Format the required elements text
    elements_text = "\n".join([f"- ID: {elem['id']} | {elem['description']}" for elem in required_elements])
    previous_results_json = json.dumps(prev_presence, indent=2) if prev_presence else "{}"

    system_instructions = (
        "You are a video-evaluation system. You will receive:\n"
        "1) A set of video frames.\n"
        "2) A list of required elements for the scene.\n"
        "3) (Optional) Data about previous iteration's presence booleans.\n\n"
        "You must return valid JSON with exactly these keys:\n"
        "{\n"
        "  \"results\": [\n"
        "     {\"id\": \"some_element_id\", \"present\": true/false},\n"
        "     ... one entry for each required element ...\n"
        "  ],\n"
        "  \"notes\": \"Short explanation or commentary\"\n"
        "}\n"
        "No extra keys, no next prompt. Focus on correctness of presence booleans."
    )

    # Build the final user message (list of frames + text describing required elements)
    user_msg_content = []
    user_msg_content.append({
        "type": "text",
        "text": (
            f"User Goal: {user_goal}\n"
            f"Required Elements:\n{elements_text}\n\n"
            "Previous iteration presence booleans:\n"
            f"{previous_results_json}\n\n"
            "Now evaluate these frames for each required element.\n"
        )
    })

    # Attach frames as data URLs
    frame_files = sorted(f for f in os.listdir(iteration_frames_dir) if f.endswith(".png"))
    for filename in frame_files:
        filepath = os.path.join(iteration_frames_dir, filename)
        data_url = encode_image_as_data_url(filepath)
        user_msg_content.append({
            "type": "image_url",
            "image_url": {"url": data_url, "detail": "auto"}
        })

    # Send to GPT
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_msg_content},
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=max_tokens,
        )
        raw_content = response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = f"OpenAI API call encountered an error: {e}"
        return (error_msg, False, {}, "No notes")

    # 3) Parse GPT's JSON
    try:
        parsed = json.loads(raw_content)
        results = parsed.get("results", [])
        notes = parsed.get("notes", "")
    except JSONDecodeError:
        return ("Model returned invalid JSON", False, {}, "No notes")

    # Build presence dict
    presence_dict = {}
    for elem in required_elements:
        eid = elem["id"]
        # Find matching result from GPT
        found_item = next((r for r in results if r["id"] == eid), None)
        is_present = found_item.get("present") if found_item else False
        presence_dict[eid] = bool(is_present)

    # 4) Check if all required elements are present
    all_satisfied = all(presence_dict.values())

    # For logging, build a short summary
    lines = []
    lines.append("Checklist Evaluation:")
    for elem in required_elements:
        eid = elem["id"]
        found = presence_dict[eid]
        lines.append(f" - {eid} => {found}")
    lines.append(f"All Satisfied? {all_satisfied}")
    details = "\n".join(lines)

    return (details, all_satisfied, presence_dict, notes)
