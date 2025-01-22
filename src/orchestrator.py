from src.chatgpt_utils import (
    refine_prompt,
    reflect_and_improve,
    generate_rubric_with_checklist
)
from src.generate_video import generate_video
from src.evaluation import evaluate_with_checklist
import os
import json
from datetime import datetime


def run_iterations(config, goal, run_directory):
    # 1) Generate the ordered checklist-based rubric
    rubric_response = generate_rubric_with_checklist(goal, config)
    rubric_items = rubric_response.get("rubric_items", [])
    if not rubric_items:
        print("No rubric items returned, nothing to do.")
        return

    # Save the rubric to run_metadata
    run_metadata = {
        "user_goal": goal,
        "rubric_items": rubric_items,
    }
    with open(os.path.join(run_directory, "run_metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=2)

    # We’ll keep a boolean array to mark which items have been “locked in” (True = satisfied).
    locked_in = [False] * len(rubric_items)

    # Basic housekeeping
    logs_dir = config["logs"]["directory"]
    frames_dir = config["frames"]["output_directory"]
    videos_dir = config["video_output_dir"]
    for d in [logs_dir, frames_dir, videos_dir]:
        os.makedirs(d, exist_ok=True)

    max_iterations = config["iterations"]["max_iterations"]

    # This is our “current_step,” which points to the item in rubric_items we’re trying to satisfy.
    current_step = 0

    # Start with some initial text for the first prompt
    current_prompt = (
        f"You are solving the following user goal:\n{goal}\n"
        "Please propose an initial text prompt to achieve it."
    )

    iteration_count = 0
    while iteration_count < max_iterations and current_step < len(rubric_items):
        iteration_count += 1
        print(f"\n=== Iteration {iteration_count} (Working on Step {current_step + 1}) ===")

        # A) Refine the prompt
        refined_prompt_text = refine_prompt(current_prompt, config)

        # B) Generate the video
        video_path = generate_video(refined_prompt_text, config, iteration_count)

        # C) Evaluate with the checklist
        eval_details, total_score, checkbox_dict = evaluate_with_checklist(
            video_path, iteration_count, config, goal, rubric_items
        )

        # Let’s see which items are currently found (True) in the new output
        # We’ll store them in a “current_found” array of booleans
        current_found = []
        for i, item in enumerate(rubric_items):
            present = checkbox_dict.get(item["id"], False)
            current_found.append(present)

        # D) Check if we lost any locked-in items
        lost_items = []
        for i in range(len(rubric_items)):
            if locked_in[i] and not current_found[i]:
                # Oops, we lost an item that was previously satisfied!
                lost_items.append(rubric_items[i]["id"])

        # E) Did we satisfy the item for “current_step” this round?
        step_item_id = rubric_items[current_step]["id"]
        step_satisfied_now = current_found[current_step]

        # Build reflection prompt
        reflection_prompt = (
            f"Goal: {goal}\n"
            f"Currently focusing on item #{current_step + 1}: '{rubric_items[current_step]['description']}'\n\n"
            f"This iteration’s refined prompt:\n{refined_prompt_text}\n"
            f"Evaluation:\n{eval_details}\n"
            f"Locked-in items lost: {lost_items}\n"
            f"Did we succeed in item #{current_step + 1}? {step_satisfied_now}\n\n"
            "Please reflect on how to keep all previously satisfied items AND satisfy the current step.\n"
            "Suggest a next prompt accordingly."
        )

        reflection_text, next_prompt_text = reflect_and_improve(reflection_prompt, config)

        # Create iteration log
        iteration_log = {
            "iteration": iteration_count,
            "current_step": current_step + 1,
            "prompt_attempted": refined_prompt_text,
            "video_path": video_path,
            "evaluation_details": eval_details,
            "lost_items": lost_items,
            "step_satisfied_now": step_satisfied_now,
            "reflection": reflection_text,
            "suggested_next_prompt": next_prompt_text,
        }

        # Write iteration logs
        log_filename = os.path.join(logs_dir, f"iteration_{iteration_count}.json")
        with open(log_filename, "w") as f:
            json.dump(iteration_log, f, indent=2)

        # Print summary
        print(f"Prompt Attempted: {refined_prompt_text}")
        print(f"Lost Items: {lost_items}")
        print(f"Step {current_step + 1} Satisfied?: {step_satisfied_now}")
        print(f"Reflection: {reflection_text}\nNext Prompt: {next_prompt_text}")

        # F) Decide how to proceed:
        if lost_items:
            # We regressed. We do NOT advance the step. We remain on the same step.
            # We’ll just pick up next iteration with next_prompt_text as our new prompt,
            # hoping reflection can fix the regression + achieve the step item.
            print("Regression detected! Will retry the same step. Not advancing.")
        else:
            # No regression. Check if we satisfied the step item.
            if step_satisfied_now:
                # Great, we can lock in the step item
                locked_in[current_step] = True
                print(f"Success on step {current_step + 1}. Locking it in and moving to the next step.")
                current_step += 1
            else:
                # No regression, but we haven’t satisfied the step item yet
                print(f"Still missing step {current_step + 1}, will keep trying this step.")
                # We remain on the same step

        # G) Update the prompt for the next iteration
        current_prompt = next_prompt_text

    # If we exit the loop, either we reached max_iterations or we satisfied all steps
    if current_step >= len(rubric_items):
        print(f"All {len(rubric_items)} steps satisfied by iteration {iteration_count}!")
    else:
        print(f"Reached iteration limit {max_iterations} but only completed steps 1..{current_step}.")

    generate_final_summary(run_directory)


def generate_final_summary(run_directory):
    # same function as before or adapted. Summarizes logs, etc.
    logs_dir = os.path.join(run_directory, "logs")
    summary_file = os.path.join(run_directory, "final_summary.txt")

    run_metadata_path = os.path.join(run_directory, "run_metadata.json")
    if os.path.exists(run_metadata_path):
        with open(run_metadata_path, "r") as f:
            run_metadata = json.load(f)
    else:
        run_metadata = {"user_goal": "Unknown", "rubric_items": []}

    user_goal = run_metadata.get("user_goal", "Not found")
    rubric_items = run_metadata.get("rubric_items", [])
    iteration_summaries = []

    # Gather iteration logs
    for filename in sorted(os.listdir(logs_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(logs_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
            iteration_summaries.append(data)

    # Build a readable “one‐pager”
    lines = []
    lines.append("FINAL ONE-PAGER SUMMARY OF THIS RUN")
    lines.append("====================================")
    lines.append("")
    lines.append(f"User Goal: {user_goal}")
    lines.append("")
    lines.append("Rubric Items:")
    for idx, item in enumerate(rubric_items):
        lines.append(f"{idx + 1}) {item['id']} - {item['description']} ({item['points']} points)")
    lines.append("")
    lines.append("Iteration Breakdown\n-------------------")
    for item in iteration_summaries:
        lines.append(f"Iteration {item['iteration']} (Step {item['current_step']}):")
        lines.append(f"  Prompt: {item['prompt_attempted']}")
        lines.append(f"  Lost Items: {item['lost_items']}")
        lines.append(f"  Step {item['current_step']} Satisfied?: {item['step_satisfied_now']}")
        lines.append(f"  Reflection: {item['reflection']}")
        lines.append("")

    with open(summary_file, "w") as f:
        f.write("\n".join(lines))

    print(f"\nA final one-page summary has been written to: {summary_file}\n")
