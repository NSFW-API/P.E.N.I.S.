def generate_video(prompt, config, iteration, custom_width=512, custom_height=512):
    """
    通过本地 ComfyUI 服务生成视频（同步等待任务完成，并拷贝生成的文件）。
    
    参数:
      prompt: 视频描述文本
      config: 包含以下可选配置的字典：
          - "api_url": ComfyUI 服务的 API 地址，默认 "http://127.0.0.1:6006/api/prompt"
          - "client_id": 客户端标识，默认 "default_client"
          - "video_output_dir": 目标视频输出目录（相对于 /root/ComfyUI/output），默认 "videos"
          - "video_length": 视频长度，默认 69
          - "steps": 采样步数，默认 50
          - "lora": 用于生成视频的 lora 前缀，可用于构造文本描述
          - "lora_1": lora1 模型文件路径
          - "lora_1_strength": lora1 的权重（strength）
          - "lora_2": lora2 模型文件路径
          - "lora_2_strength": lora2 的权重（strength）
      iteration: 当前迭代编号，将用作输出文件名的一部分
      custom_width: 视频宽度
      custom_height: 视频高度
    
    返回:
      生成视频的文件路径（拷贝到目标目录后的路径）。
    """
   
    # FIX IT HERE
    custom_width=config.get("comfyui", {}).get("custom_width", 320)
    custom_height=config.get("comfyui", {}).get("custom_height", 320)
    # print("custom_width:",custom_width)
    # print("custom_height:",custom_height)

    # print(iteration,custom_width,custom_height)
    # print(config)
    import json
    import os
    import time
    import shutil
    import glob
    from urllib import request
    from datetime import datetime

    # 定义本地 ComfyUI 工作流 JSON（直接来自示例）
    workflow_text = '''
{
  "10": {
    "inputs": {
      "vae_name": "hunyuan_video_vae_bf16.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "11": {
    "inputs": {
      "clip_name1": "clip_l.safetensors",
      "clip_name2": "llava_llama3_fp8_scaled.safetensors",
      "type": "hunyuan_video",
      "device": "default"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "12": {
    "inputs": {
      "unet_name": "hunyuan_video_FastVideo_720_fp8_e4m3fn.safetensors",
      "weight_dtype": "fp8_e4m3fn"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "16": {
    "inputs": {
      "sampler_name": "euler"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "17": {
    "inputs": {
      "scheduler": "beta",
      "steps": 12,
      "denoise": 1,
      "model": [
        "252",
        0
      ]
    },
    "class_type": "BasicScheduler",
    "_meta": {
      "title": "BasicScheduler"
    }
  },
  "22": {
    "inputs": {
      "model": [
        "67",
        0
      ],
      "conditioning": [
        "26",
        0
      ]
    },
    "class_type": "BasicGuider",
    "_meta": {
      "title": "BasicGuider"
    }
  },
  "25": {
    "inputs": {
      "noise_seed": [
        "308",
        0
      ]
    },
    "class_type": "RandomNoise",
    "_meta": {
      "title": "Noise"
    }
  },
  "26": {
    "inputs": {
      "guidance": 4,
      "conditioning": [
        "44",
        0
      ]
    },
    "class_type": "FluxGuidance",
    "_meta": {
      "title": "Guidance"
    }
  },
  "44": {
    "inputs": {
      "text": [
        "395",
        0
      ],
      "clip": [
        "11",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Positive Prompt)"
    }
  },
  "67": {
    "inputs": {
      "shift": 7.5,
      "model": [
        "252",
        0
      ]
    },
    "class_type": "ModelSamplingSD3",
    "_meta": {
      "title": "Shift"
    }
  },
  "73": {
    "inputs": {
      "tile_size": 128,
      "overlap": 64,
      "temporal_size": 64,
      "temporal_overlap": 8,
      "samples": [
        "243",
        0
      ],
      "vae": [
        "10",
        0
      ]
    },
    "class_type": "VAEDecodeTiled",
    "_meta": {
      "title": "VAE Decode (Tiled)"
    }
  },
  "106": {
    "inputs": {
      "sharpen_radius": 1,
      "sigma": 0.43,
      "alpha": 0.5,
      "image": [
        "73",
        0
      ]
    },
    "class_type": "ImageSharpen",
    "_meta": {
      "title": "Image Sharpen"
    }
  },
  "180": {
    "inputs": {
      "CLIP": [
        "11",
        0
      ],
      "VAE": [
        "10",
        0
      ]
    },
    "class_type": "Anything Everywhere3",
    "_meta": {
      "title": "Anything Everywhere3"
    }
  },
  "232": {
    "inputs": {
      "width": 320,
      "height": 560,
      "length": 9,
      "batch_size": 1
    },
    "class_type": "EmptyHunyuanLatentVideo",
    "_meta": {
      "title": "EmptyHunyuanLatentVideo"
    }
  },
  "243": {
    "inputs": {
      "select": 0,
      "on_missing": "next",
      "input_2": [
        "332",
        1
      ],
      "input_3": [
        "232",
        0
      ]
    },
    "class_type": "Big Latent Switch [Dream]",
    "_meta": {
      "title": "⭆ Big Latent Switch"
    }
  },
  "252": {
    "inputs": {
      "lora_name": "",
      "strength": 0.8,
      "blocks_type": "double_blocks",
      "model": [
        "256",
        0
      ]
    },
    "class_type": "HunyuanVideoLoraLoader",
    "_meta": {
      "title": "Hunyuan Video LoRA Loader"
    }
  },
  "253": {
    "inputs": {
      "lora_name": "Hunyuan/hyvideo_FastVideo_LoRA-fp8.safetensors",
      "strength": -0.25,
      "blocks_type": "all",
      "model": [
        "12",
        0
      ]
    },
    "class_type": "HunyuanVideoLoraLoader",
    "_meta": {
      "title": "Hunyuan Video LoRA Loader"
    }
  },
  "256": {
    "inputs": {
      "lora_name": "",
      "strength": 0.8,
      "blocks_type": "double_blocks",
      "model": [
        "253",
        0
      ]
    },
    "class_type": "HunyuanVideoLoraLoader",
    "_meta": {
      "title": "Hunyuan Video LoRA Loader"
    }
  },
  "296": {
    "inputs": {
      "message": "ONE TASK COMPLETED!",
      "mode": "always",
      "any": [
        "106",
        0
      ]
    },
    "class_type": "SystemNotification|pysssss",
    "_meta": {
      "title": "Notification1"
    }
  },
  "306": {
    "inputs": {
      "text": "",
      "seed": [
        "308",
        0
      ],
      "autorefresh": "Yes"
    },
    "class_type": "DPRandomGenerator",
    "_meta": {
      "title": "Random Prompts"
    }
  },
  "307": {
    "inputs": {
      "output": "",
      "source": [
        "395",
        0
      ]
    },
    "class_type": "Display Any (rgthree)",
    "_meta": {
      "title": "Prompt Check"
    }
  },
  "308": {
    "inputs": {
      "seed": [
        "392",
        0
      ]
    },
    "class_type": "Seed Generator",
    "_meta": {
      "title": "Seed Generator"
    }
  },
  "332": {
    "inputs": {
      "speedup": "Original (1x)",
      "enable_custom_speed": false,
      "custom_speed": 1,
      "noise": [
        "25",
        0
      ],
      "guider": [
        "22",
        0
      ],
      "sampler": [
        "16",
        0
      ],
      "sigmas": [
        "17",
        0
      ],
      "latent_image": [
        "232",
        0
      ]
    },
    "class_type": "TeaCacheHunyuanVideoSampler",
    "_meta": {
      "title": "TeaCache HunyuanVideo Sampler"
    }
  },
  "354": {
    "inputs": {
      "frame_rate": 24,
      "loop_count": 0,
      "filename_prefix": "Hunyuan_raw",
      "format": "video/h264-mp4",
      "pix_fmt": "yuv420p",
      "crf": 15,
      "save_metadata": true,
      "trim_to_audio": false,
      "pingpong": false,
      "save_output": [
        "397",
        0
      ],
      "images": [
        "106",
        0
      ],
      "vae": [
        "10",
        0
      ]
    },
    "class_type": "VHS_VideoCombine",
    "_meta": {
      "title": "Video Combine"
    }
  },
  "366": {
    "inputs": {},
    "class_type": "Anything Everywhere",
    "_meta": {
      "title": "Anything Everywhere"
    }
  },
  "392": {
    "inputs": {
      "seed": -1
    },
    "class_type": "Seed (rgthree)",
    "_meta": {
      "title": "Seed (rgthree)"
    }
  },
  "395": {
    "inputs": {
      "Text": [
        "306",
        0
      ],
      "Pattern": "\\n",
      "Replace_With": "",
      "Mode": "Strict"
    },
    "class_type": "DF_String_Replace",
    "_meta": {
      "title": "String Replace"
    }
  },
  "397": {
    "inputs": {
      "value": true
    },
    "class_type": "Bool-🔬",
    "_meta": {
      "title": "Save RAW Output"
    }
  }
}
'''
    # 载入工作流 JSON
    workflow = json.loads(workflow_text)

    # 更新节点 "44" 中的文本（如果 config 中有 "lora" 则作为前缀）
    lora_1_trigger = config.get("comfyui", {}).get("lora_1_trigger", "")
    lora_2_trigger = config.get("comfyui", {}).get("lora_2_trigger", "")
    prompt =  f"{lora_1_trigger},{lora_2_trigger}, High quality video of " + prompt
    print("prompt:",  prompt)
    #FIX IT
    if lora_1_trigger:
        workflow["306"]["inputs"]["text"] = prompt
    else:
        workflow["306"]["inputs"]["text"] = prompt

   

    # 更新视频尺寸和长度（节点 "45"）
    workflow["232"]["inputs"]["width"]   = custom_width
    workflow["232"]["inputs"]["height"]  = custom_height
    workflow["232"]["inputs"]["length"]  = config.get("comfyui", {}).get("video_length", 69)

    # 更新采样步数（节点 "17"）
    workflow["17"]["inputs"]["steps"] = config.get("comfyui", {}).get("steps", 6)
    # print("steps:",config.get("comfyui", {}).get("steps", 6))

    # 更新 lora1 和 lora2 模型路径及其权重（strength）均从 config 中读取
    if "lora_1" in config["comfyui"]:
        workflow["256"]["inputs"]["lora_name"] = config["comfyui"]["lora_1"]
        if "lora_1_strength" in config["comfyui"]:
            workflow["256"]["inputs"]["strength"] = config["comfyui"]["lora_1_strength"]
    if "lora_2" in config["comfyui"]:
        workflow["252"]["inputs"]["lora_name"] = config["comfyui"]["lora_2"]
        if "lora_2_strength" in config["comfyui"]:
            workflow["252"]["inputs"]["strength"] = config["comfyui"]["lora_2_strength"]
            
    # print("lora2 st:", config["comfyui"]["lora_2_strength"])
    # print("lora1:",config["comfyui"]["lora_1"])
    # print("lora1 s:",config["comfyui"]["lora_1_strength"])


    workflow["67"]["inputs"]["shift"] = config.get("comfyui", {}).get("shift", 7.5)
    # print("shift :",config["comfyui"]["shift"])


    # 构造存储路径（ComfyUI 生成文件时使用的前缀）
    # current_date = datetime.now().strftime("%Y_%m_%d")
    # if lora:
    #     storage_path = os.path.join("Hunyuan", lora, current_date, lora)
    # else:
    #     storage_path = os.path.join("Hunyuan", current_date)
    storage_path =  os.path.join("Hunyuan",f"iteration_{iteration}")
    workflow["354"]["inputs"]["filename_prefix"] = storage_path

    # 获取 ComfyUI API 地址和 client_id（若未指定则使用默认值）
    api_url = config.get("api_url", "http://127.0.0.1:6006/api/prompt")
    client_id = config.get("client_id", "default_client")

    # 提交任务：构造 POST 请求，将工作流 JSON 发送给本地 ComfyUI 服务
    payload = {
        "prompt": workflow,
        "client_id": client_id
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with request.urlopen(req) as res:
        response_data = res.read().decode("utf-8")
        print("[INFO] 提交任务响应:", response_data)

    # 从返回的 JSON 中解析任务 ID（假设返回数据格式为：{"prompt_id": "xxx"}）
    try:
        task_info = json.loads(response_data)
        task_id = task_info.get("prompt_id")
        if not task_id:
            raise ValueError("返回数据中未包含任务ID")
    except Exception as e:
        raise ValueError("解析任务ID失败: " + str(e))

    print("[INFO] 提交 ComfyUI 任务，任务ID:", task_id)

    # 构造 history 接口地址（假设与 api_url 同一级，即替换 "prompt" 为 "history"）
    base_url = api_url.rsplit("/", 1)[0]
    history_url = f"{base_url}/history/{task_id}"

    # 轮询等待任务完成（每5秒一次）
    while True:
        try:
            with request.urlopen(history_url) as hist_res:
                hist_response = hist_res.read().decode("utf-8")
                #print(hist_response)
            history_data = json.loads(hist_response)
        except Exception as e:
            print("[WARN] 获取任务状态失败:", e)
            time.sleep(5)
            continue

        # 假设返回 JSON 格式为 { task_id: { "status": { "status_str": "xxx", "completed": true/false } } }
        task_status = history_data.get(task_id, {}).get("status", {})
        status_str = task_status.get("status_str", "")
        completed = task_status.get("completed", False)

        if completed is True or str(completed).lower() == "true":
            print("[INFO] ComfyUI 任务完成！")
            break
        elif status_str.lower() == "failed":
            raise RuntimeError("ComfyUI 任务失败！")
        else:
            # print("[INFO] 等待任务完成，当前状态:", status_str)
            print('.', end='', flush=True)
            time.sleep(5)

    # 任务完成后，将 ComfyUI 生成的视频文件从 base_output_dir 拷贝到目标目录
    # 这里 base_output_dir 为 ComfyUI 生成文件所在目录，其路径由 storage_path 构成，位于 /root/ComfyUI/output 下
    base_output_dir = os.path.join("/root/ComfyUI/output", "Hunyuan")
    print("base output dir:",base_output_dir)
    # 最终目标目录，配置文件 video_output_dir 为相对于 /root/ComfyUI/output 的路径
    target_output_dir = os.path.join("/root/ComfyUI/output", config.get("video_output_dir", "videos"))
    os.makedirs(target_output_dir, exist_ok=True)

    # 在 base_output_dir 下查找 .mp4 文件（假设只生成一个视频文件，若有多个则取最新的）
    mp4_files = glob.glob(os.path.join(base_output_dir, "*.mp4"))
    if not mp4_files:
        raise FileNotFoundError("在 {} 中未找到生成的视频文件".format(base_output_dir))
    generated_video = max(mp4_files, key=os.path.getmtime)

    # 拷贝到目标目录，文件名设为 iteration_{iteration}.mp4
    final_video_path = os.path.join(target_output_dir, f"iteration_{iteration}.mp4")
    shutil.copy(generated_video, final_video_path)
    print("[INFO] 生成视频已拷贝到:", final_video_path)

    return final_video_path
