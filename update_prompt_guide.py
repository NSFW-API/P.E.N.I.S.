import json
import os
import shutil

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def main():
    # 1) Load config
    config_path = "config.yaml"
    with open(config_path, "r") as cf:
        config = yaml.safe_load(cf)

    runs_directory = config.get("runs_directory", "runs")
    # We'll assume subfolders: runs_untrained/unlearned, runs_untrained/learned
    unlearned_dir = os.path.join(runs_directory, "unlearned")
    learned_dir = os.path.join(runs_directory, "learned")

    # Make sure "learned" directory exists
    os.makedirs(learned_dir, exist_ok=True)

    # 2) Read the current prompting guide from config
    current_guide = config.get("prompting_guide", "")

    # 3) Collect logs from each sub-run folder in "unlearned"
    #    For each run subfolder, we might read iteration_1.json, iteration_2.json, etc.
    if not os.path.exists(unlearned_dir):
        print(f"No 'unlearned' directory found at {unlearned_dir}. Exiting.")
        return

    subfolders = [
        f for f in os.listdir(unlearned_dir)
        if os.path.isdir(os.path.join(unlearned_dir, f))
    ]

    if not subfolders:
        print("No unlearned run folders found. Exiting.")
        return

    logs_text = []
    for run_name in subfolders:
        run_path = os.path.join(unlearned_dir, run_name)
        logs_dir = os.path.join(run_path, "logs")
        if not os.path.exists(logs_dir):
            # No logs folder? Skip
            continue

        for file_name in os.listdir(logs_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(logs_dir, file_name)
                with open(file_path, "r") as f:
                    try:
                        data = json.load(f)
                        # We can store just the relevant fields
                        iteration = data.get("iteration")
                        prompt_used = data.get("prompt_used")
                        presence = data.get("presence_dict")
                        notes = data.get("notes")
                        user_review = data.get("user review")
                        # Build a short summary
                        logs_text.append(
                            f"Run: {run_name}, Iteration: {iteration}, Prompt Used: {prompt_used}, Presence: {presence}, Notes: {notes}, User Review (Optional): {user_review}")
                    except:
                        pass

    if not logs_text:
        print("No logs found in unlearned runs. Exiting.")
        return

    # 4) Make an LLM call to produce an updated guide
    #    Summarize the logs, then instruct the model to refine your guide.

    joined_logs_snippet = "\n".join(logs_text)

    # We’ll do a simple chat completion call:
    model_name = config["openai"]["update_guide_model_name"]  # e.g., "gpt-4"

    # System instructions for how to revise the guide
    system_prompt = (
        "You are an AI specialized in revising an NSFW prompting guide. "
        "You will receive:\n"
        "1) The current prompting guide.\n"
        "2) A summary of logs from previous runs.\n\n"
        "Use these inputs to produce an updated guide that incorporates best practices, "
        "fixes repeated errors, and clarifies any instructions to avoid model confusion."
    )

    user_prompt = f"""Current NSFW Prompting Guide:
{current_guide}

Logs from unlearned runs:
{joined_logs_snippet}

Please reply with an updated prompting guide. Keep the same overall structure, but revise or add points derived from the logs' lessons. 
Output only the updated guide text—no extra commentary.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        updated_guide = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI call failed: {e}")
        return

    # 5) Save updated guide to a file. You can either overwrite config.yaml or store in a new file.
    # Here, we'll store in a new file: updated_prompting_guide.txt
    updated_guide_path = os.path.join(runs_directory, "updated_prompting_guide.txt")
    with open(updated_guide_path, "w") as f:
        f.write(updated_guide)
    print(f"Successfully wrote updated guide to {updated_guide_path}.")

    # (Optional) If you want to also overwrite config.yaml, do something like:
    # config["prompting_guide"] = updated_guide
    # with open(config_path, "w") as cf:
    #     yaml.safe_dump(config, cf)
    # print("config.yaml has been updated with the new guide.")

    # 6) Move the processed runs from "unlearned" to "learned"
    for run_name in subfolders:
        src = os.path.join(unlearned_dir, run_name)
        dst = os.path.join(learned_dir, run_name)
        shutil.move(src, dst)
        print(f"Moved {src} => {dst}")

    print("All unlearned runs have been moved to the learned directory.")


if __name__ == "__main__":
    main()
