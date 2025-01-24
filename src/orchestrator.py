import os
import json

from src.chatgpt_utils import extract_goal_requirements, refine_unified_prompt
from src.generate_video import generate_video
from src.evaluation import evaluate_all_elements

import os
import json
from src.chatgpt_utils import extract_goal_requirements, refine_unified_prompt
from src.generate_video import generate_video
from src.evaluation import evaluate_all_elements


def run_iterations(config, user_goal, run_directory):
    # Prepare directories
    logs_dir = os.path.join(run_directory, "logs")
    frames_dir = os.path.join(run_directory, "frames")
    videos_dir = os.path.join(run_directory, "videos")
    for d in [logs_dir, frames_dir, videos_dir]:
        os.makedirs(d, exist_ok=True)

    # 1) Extract all required elements
    requirements_data = extract_goal_requirements(user_goal, config)
    required_elements = requirements_data.get("elements", [])
    if not required_elements:
        print("No elements found in user goal. Exiting.")
        return

    max_iterations = config["iterations"]["max_iterations"]
    iteration_count = 0
    all_satisfied = False

    # This will track iteration logs in-memory, so we can build a final summary at the end
    iteration_history = []

    # (A) Write run metadata right now, so our final_summary has it.
    run_metadata = {
        "user_goal": user_goal,
        "max_iterations": max_iterations
    }
    with open(os.path.join(run_directory, "run_metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=2)

    # Begin iteration loop
    while iteration_count < max_iterations and not all_satisfied:
        iteration_count += 1
        print(f"--- Iteration {iteration_count} ---")

        history_text = summarize_history(iteration_history)

        # A) Refine prompt
        refined_result = refine_unified_prompt(
            user_goal=user_goal,
            required_elements=required_elements,
            config=config,
            iteration_history=history_text
        )
        refined_prompt = refined_result["explicit_scene_description"]
        chosen_width = refined_result["resolution_width"]
        chosen_height = refined_result["resolution_height"]
        iteration_tag = f"unified_{iteration_count}"

        # B) Generate video
        video_path = generate_video(
            prompt=refined_prompt,
            config=config,
            iteration=iteration_tag,
            custom_width=chosen_width,
            custom_height=chosen_height
        )

        # C) Evaluate
        eval_details, all_satisfied, presence_dict, notes = evaluate_all_elements(
            video_path=video_path,
            iteration_name=iteration_tag,
            config=config,
            user_goal=user_goal,
            required_elements=required_elements,
            prev_presence=None
        )

        # D) Log iteration
        iteration_log = {
            "iteration": iteration_count,
            "prompt_used": refined_prompt,
            "chosen_width": chosen_width,
            "chosen_height": chosen_height,
            "video_path": video_path,
            "eval_details": eval_details,
            "notes": notes,
            "all_satisfied": all_satisfied,
            "presence_dict": presence_dict,
        }
        iteration_history.append(iteration_log)

        # Also write to disk
        with open(os.path.join(logs_dir, f"iteration_{iteration_count}.json"), "w") as f:
            json.dump(iteration_log, f, indent=2)

        if all_satisfied:
            print(f"All required elements satisfied in {iteration_count} iterations!")
            break

    if not all_satisfied:
        print(f"Reached max iterations ({max_iterations}) without satisfying all elements.")

    # Finally, create a summary that actually includes the iteration history
    generate_final_summary(run_directory, iteration_history)


def summarize_history(iteration_history):
    if not iteration_history:
        return "No prior attempts."
    lines = []
    for it in iteration_history:
        lines.append(f"Iteration {it['iteration']}:")
        lines.append(f" Prompt used: {it['prompt_used']}")
        lines.append(f" Presence results: {it['presence_dict']}")
        lines.append(f" Notes: {it['notes']}")
        lines.append("")
    return "\n".join(lines)


def summarize_history(iteration_history):
    """
    Optional helper function to produce a textual summary of prior iterations:
    - The prompt used
    - Which elements were found / missing
    - GPT's notes
    """
    if not iteration_history:
        return "No prior attempts."
    lines = []
    for it in iteration_history:
        lines.append(f"Iteration {it['iteration']}:")
        lines.append(f"  Prompt used: {it['prompt_used']}")
        lines.append(f"  Chosen resolution: {it['chosen_width']} x {it['chosen_height']}")
        lines.append(f"  Presence results: {it['presence_dict']}")
        lines.append(f"  Notes: {it['notes']}")
        lines.append("")
    return "\n".join(lines)


def generate_final_summary(run_directory, iteration_history):
    """
    Create a text summary of:
      1) The user’s overall goal (from run_metadata.json).
      2) Each iteration’s prompt, presence results, and final success status.
    """
    summary_file = os.path.join(run_directory, "final_summary.txt")
    run_metadata_file = os.path.join(run_directory, "run_metadata.json")

    # Load the user goal from metadata if present
    if os.path.exists(run_metadata_file):
        with open(run_metadata_file, "r") as f:
            run_metadata = json.load(f)
        user_goal = run_metadata.get("user_goal", "Unknown")
    else:
        user_goal = "Unknown"

    lines = []
    lines.append("FINAL SUMMARY")
    lines.append("=============")
    lines.append(f"User's Overall Goal: {user_goal}")
    lines.append("")

    if not iteration_history:
        lines.append("No iterations were run.")
    else:
        for it in iteration_history:
            lines.append(f"Iteration {it['iteration']}:")
            lines.append(f" • Prompt: {it['prompt_used']}")
            lines.append(f" • Resolution: {it['chosen_width']}x{it['chosen_height']}")
            lines.append(f" • Presence Check: {it['presence_dict']}")
            lines.append(f" • Satisfied? {it['all_satisfied']}")
            lines.append(f" • Notes: {it['notes']}")
            lines.append("")

    # Write the final summary
    with open(summary_file, "w") as f:
        f.write("\n".join(lines))

    print(f"\nA final summary has been written to:\n  {summary_file}\n")
