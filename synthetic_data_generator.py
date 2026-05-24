import os
import argparse
import config

# Define output path
SYNTHETIC_DIR = os.path.join(config.DATASET_DIR, "synthetic")

def generate_image_stability(prompt, output_path, api_key):
    """
    Generate image using Stability AI REST API.
    """
    import base64
    import requests

    print(f"Generating image for prompt: '{prompt}'")
    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    headers = {
        "Accept": "image/png",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "steps": 40,
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "cfg_scale": 5,
        "samples": 1,
        "text_prompts": [
            {"text": prompt, "weight": 1},
            {"text": "blurry, bad anatomy, deformed hands, missing fingers", "weight": -1}
        ],
    }

    response = requests.post(url, headers=headers, json=body)
    
    if response.status_code != 200:
        raise Exception(f"Non-200 response: {str(response.text)}")
    
    data = response.json()
    for image in data["artifacts"]:
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(image["base64"]))
    print(f"Saved: {output_path}")

def generate_image_diffusers(prompt, output_path):
    """
    Generate image using local HuggingFace diffusers pipeline.
    """
    print(f"Generating image for prompt: '{prompt}'")
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError:
        print("Please install diffusers, transformers, accelerate: pip install diffusers transformers accelerate torch")
        return

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    image = pipe(prompt, negative_prompt="blurry, bad anatomy, deformed hands, missing fingers").images[0]
    image.save(output_path)
    print(f"Saved: {output_path}")

def generate_synthetic_sign(sign_label, num_samples, backend="diffusers", api_key=None):
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    sign_label = str(sign_label).strip().upper()
    sign_dir = os.path.join(SYNTHETIC_DIR, sign_label)
    os.makedirs(sign_dir, exist_ok=True)

    for i in range(num_samples):
        prompt = f"A person showing the American Sign Language gesture for the letter '{sign_label}', highly detailed hands, clear fingers, well-lit, realistic photograph"
        output_path = os.path.join(sign_dir, f"synth_{sign_label}_{i}.png")
        
        if backend == "stability":
            if not api_key:
                print("Stability API Key required.")
                return
            generate_image_stability(prompt, output_path, api_key)
        else:
            generate_image_diffusers(prompt, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ASL dataset images")
    parser.add_argument("--sign", type=str, required=True, help="ASL sign/letter to generate")
    parser.add_argument("--samples", type=int, default=5, help="Number of images to generate")
    parser.add_argument("--backend", type=str, choices=["diffusers", "stability"], default="diffusers")
    parser.add_argument("--api_key", type=str, default=None, help="Stability API Key")
    
    args = parser.parse_args()
    generate_synthetic_sign(args.sign, args.samples, args.backend, args.api_key)
