def generate_video(prompt, config, iteration, custom_width=512, custom_height=512):
    """
    Call Replicate's Hunyuan-Video model with the given prompt and
    user-chosen numeric width/height.
    """
    import os
    import replicate

    replicate_token = os.environ.get("REPLICATE_API_TOKEN")
    if not replicate_token:
        raise ValueError("REPLICATE_API_TOKEN not set. Please set your token as an environment variable.")

    # Just use custom_width, custom_height directly
    width = int(custom_width)
    height = int(custom_height)

    input_params = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "video_length": 69,
        "infer_steps": 50,
        "fps": 24,
        "embedded_guidance_scale": 6,
        # "seed": 42, # etc.
    }

    output = replicate.run(
        config["replicate"]["model_name"],
        input=input_params
    )
    video_output_dir = config["video_output_dir"]
    os.makedirs(video_output_dir, exist_ok=True)
    video_path = os.path.join(video_output_dir, f"iteration_{iteration}.mp4")

    with open(video_path, "wb") as f:
        f.write(output.read())
    return video_path
