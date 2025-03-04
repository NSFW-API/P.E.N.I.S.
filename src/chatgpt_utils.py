import json
import os
from json import JSONDecodeError

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def extract_goal_requirements(user_goal, config):
    """
    Uses ChatGPT to parse the user goal into a structured list of required elements.

    Returns JSON like:
    {
      "elements": [
        {"id": "foot_contact", "description": "A foot is visibly in contact with the penis."},
        {"id": "close_up_perspective", "description": "The camera angle is very close to the penis and foot."},
        ...
      ]
    }
    """

    model_name = config["openai"]["model_name"]
    max_tokens = config["openai"].get("max_completion_tokens", 2000)

    system_prompt = """  
    You are an AI that extracts each distinct requirement (or element) from a user's adult content goal.  
    Your output must be valid JSON with a single key "elements", whose value is an array of objects.  
    If you are unsure, provide your best guess. DO NOT return an empty array—there must be at least one element if the user mentions any visible or explicit scenario.  

    Follow this JSON structure exactly:  
    {  
      "elements": [  
        {  
          "id": "short_snake_case_summary_of_requirement",  
          "description": "Detailed explanation of that requirement."  
        },  
        ...  
      ]  
    }  
    No additional commentary outside the JSON.  
    """

    user_prompt = (
        f"User Goal: {user_goal}\n\n"
        "List each important requirement (i.e., everything that must be visually present or shown) "
        "as separate items in the \"elements\" array. Provide no additional commentary."
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=max_tokens,
    )
    raw_content = response.choices[0].message.content.strip()
    print("\n\nRequirements:\n---\n")
    print(raw_content)

    try:
        parsed = json.loads(raw_content)
    except JSONDecodeError:
        parsed = {"elements": []}

    return parsed


def refine_unified_prompt(user_goal, required_elements, config, iteration_history=""):
    import json
    from json import JSONDecodeError
    from dotenv import load_dotenv
    from openai import OpenAI
    import os

    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    model_name = config["openai"]["model_name"]
    max_tokens = config["openai"].get("max_completion_tokens", 2000)

    # Pull the guide from config.yaml
    guide_text = config.get("prompting_guide", "")

    # Turn the required elements into bullet points
    elements_text = "\n".join([f"- {el['description']}" for el in required_elements])

    system_prompt = f"""
    You are a creative AI assistant specialized in writing text prompts for a video-generation model.
    Below are iteration logs from previous attempts:
    {iteration_history}
    
    You must follow these NSFW prompting guidelines:
    {guide_text}
    
    Now, your job is to produce a brand-new JSON structure with these keys exactly (no extras):
    
    {{
      "preface": <string>,
      "explicit_scene_description": <string>,
      "disclaimers": <string>,
      "resolution_width": <integer>,
      "resolution_height": <integer>
    }}
    
    Constraints/Requirements:
    1) resolution_width and resolution_height Only three resolutions: 480x848, 576x1024, and 720x1280. You can change this at any time, learning frome each iteration what works and what doesn't.
    2) The video generation model is highly sensitive to aspect ratio; feel free to experiment with different width/height pairs that suit the scene. For example, a person standing upright might benefit from a portrait style (e.g., 480x848), while a lying-down shot could work better in a landscape format (e.g., 576x1024 or 720x1280).  
    3) Make sure the "explicit_scene_description" includes all required elements missed before, and fix any issues gleaned from the conversation history.  
    4) "preface" sets the style or mood of the scene, such as how strong and confident and secure in her femininity the actors are.  
    5) "disclaimers" can mention consenting adults, no minors, etc., if needed.
    Only return valid JSON, nothing else (no markdown).
    """.strip()

    user_prompt = f"""Overall user goal: {user_goal}
    Required elements:
    {elements_text}
    Produce the final JSON now."""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=max_tokens,
    )

    raw_content = response.choices[0].message.content.strip()
    print("\n\nGeneration details:\n---\n")
    print(raw_content)

    try:
        parsed = json.loads(raw_content)
    except JSONDecodeError:
        # If parsing fails, just fallback to an empty structure
        parsed = {
            "preface": "",
            "explicit_scene_description": raw_content,
            "disclaimers": "",
            "resolution_width": 512,
            "resolution_height": 512
        }

    # Validate or clamp the chosen resolution to [100..512]
    w = parsed.get("resolution_width", 512)
    h = parsed.get("resolution_height", 512)
    w = max(100, min(w, 512))
    h = max(100, min(h, 512))

    # Rebuild final
    new_result = {
        "preface": parsed.get("preface", ""),
        "explicit_scene_description": parsed.get("explicit_scene_description", ""),
        "disclaimers": parsed.get("disclaimers", ""),
        "resolution_width": w,
        "resolution_height": h
    }
    return new_result
