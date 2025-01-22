import os
import replicate


def generate_video(prompt, config, iteration):
    """Call Replicate's Hunyuan-Video model with the given prompt."""

    # Ensure your Replicate API token is set
    replicate_token = os.environ.get("REPLICATE_API_TOKEN")
    if not replicate_token:
        raise ValueError("REPLICATE_API_TOKEN not set. Please set your token as an environment variable.")

    # According to the schema, we can pass parameters like width, height, video_length, etc.
    input_params = {
        "prompt": prompt,
        "width": 256,  # required: 256 px wide
        "height": 256,  # required: 256 px tall
        "video_length": 69,  # required: generate 69 frames
        "infer_steps": 50,  # number of denoising steps
        "fps": 24,  # frames per second (can be adjusted if needed)
        "embedded_guidance_scale": 6,  # default guidance
        # "seed":   # optionally specify a seed here if desired, e.g., "seed": 42
    }

    # Make sure the replicate library is configured
    output = replicate.run(
        config["replicate"]["model_name"],
        input=input_params
    )

    # The returned object is typically a file-like (StreamingResponse) that we can write to disk
    video_output_dir = config["video_output_dir"]
    os.makedirs(video_output_dir, exist_ok=True)
    video_path = os.path.join(video_output_dir, f"iteration_{iteration}.mp4")

    with open(video_path, "wb") as f:
        f.write(output.read())

    return video_path
