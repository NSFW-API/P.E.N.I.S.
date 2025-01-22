import os
import json
import base64
import subprocess
from json import JSONDecodeError

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
from PIL import Image


##################################################
# 1) Frame Extraction
##################################################
def extract_frames(video_path, interval, output_dir):
    """
    Use ffmpeg to extract frames every 'interval' frames.
    Example command:
      ffmpeg -i video.mp4 -vf "select='not(mod(n,30))'" -vsync 0 out%03d.png
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(video_path))[0]

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select='not(mod(n,{interval}))'",
        "-vsync", "0",
        os.path.join(output_dir, f"{basename}_%03d.png")
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


##################################################
# 2) Base64 Encoder
##################################################
def encode_image_as_data_url(image_path):
    """
    Convert an image file to a base64 data URI string.
    """
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


##################################################
# 3) Evaluate Video by Sending All Frames at Once
##################################################
def evaluate_with_checklist(video_path, iteration, config, goal, rubric_items):
    """
    1) Extract frames using ffmpeg.
    2) Pass them along with the `rubric_items` (list of {id, description, points})
       to GPT. GPT should respond with a JSON object marking each `id: True/False`.
    3) Sum the points for any item that is True.
    4) Return (details, total_score, checkbox_dict), where `checkbox_dict` is
       e.g. {"nudity": true, "tentacles_touching": false, ...}.
    """
    frames_dir = os.path.join(config["frames"]["output_directory"], f"iteration_{iteration}")
    os.makedirs(frames_dir, exist_ok=True)
    interval = config["frames"]["extract_interval"]
    max_tokens = config["openai"].get("max_completion_tokens", 25000)

    # Extract frames
    extract_frames(video_path, interval, frames_dir)

    # Build the checklist instructions for GPT
    # We'll ask it to examine each item from rubric_items and return True or False.
    checklist_instructions = (
        "You have a set of checklist items. For each item, you must return a boolean indicating "
        "whether or not the item is satisfied in the video frames. Output EXACTLY this JSON structure:\n\n"
        "{\n"
        "  \"results\": [\n"
        "    {\n"
        "      \"id\": \"<string>\",\n"
        "      \"present\": <true/false>\n"
        "    },\n"
        "    ... (for each item)\n"
        "  ]\n"
        "}\n"
        "No extra fields. No additional wrapping text.\n"
    )

    # Prepare user_content array with text + images
    user_content = [
        {
            "type": "text",
            "text": (
                f"Goal: {goal}\n"
                "Below is the rubric's checklist. Please say true/false if each item is visible/achieved.\n"
                f"{json.dumps(rubric_items, indent=2)}\n"
                "Now, evaluate the following frames:\n"
            ),
        }
    ]

    # Attach frames as data URLs
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    for filename in frame_files:
        filepath = os.path.join(frames_dir, filename)
        data_url = encode_image_as_data_url(filepath)
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "auto"}
            }
        )

    # Build messages for GPT
    messages = [
        {
            "role": "system",
            "content": (
                "You are a video-evaluation system. "
                "Return valid JSON indicating for each rubric item whether it's present."
            ),
        },
        {"role": "user", "content": user_content},
        {"role": "system", "content": checklist_instructions},
    ]

    try:
        response = client.chat.completions.create(
            model=config["openai"]["model_name"],
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=max_tokens,
        )
    except Exception as e:
        print("OpenAI API call encountered an error:", e)
        return "Failed to process frames.", 0, {}

    raw_content = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw_content)
        results = parsed.get("results", [])
    except JSONDecodeError:
        print("Model returned invalid JSON:\n", raw_content)
        return "Model returned invalid JSON", 0, {}

    # Tally the score
    # results should look like: [ { "id": "nudity", "present": true }, ... ]
    checkbox_dict = {}  # to store each item’s boolean
    total_score = 0
    for item in rubric_items:
        # Find result for item['id'] in results
        found = next((r for r in results if r["id"] == item["id"]), None)
        if found and found.get("present") is True:
            total_score += item.get("points", 0)
            checkbox_dict[item["id"]] = True
        else:
            checkbox_dict[item["id"]] = False

    details = (
        "Checklist Evaluation:\n" +
        "\n".join(
            f"- {item['id']} => {checkbox_dict[item['id']]} (worth {item['points']} points)"
            for item in rubric_items
        )
        + f"\nTotal Score = {total_score}\n"
    )

    return details, total_score, checkbox_dict
