import base64
import os

import requests
from PIL import Image


def validate_image(image_path: str) -> bool:
    """
    Validate if an image file can be opened and is not corrupted.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if the image is valid and can be opened, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that it's a valid image
        # Re-open to check if it can be fully loaded (verify() may not catch all issues)
        with Image.open(image_path) as img:
            img.load()  # Force load the image data
        return True
    except Exception as e:
        print(f"Warning: Image '{image_path}' is invalid or corrupted: {e}")
        return False


def generate_image(
    prompt_file: str,
    reference_images: list[str],
    output_file: str,
    aspect_ratio: str = "16:9",
) -> str:
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()
    
    # Filter out invalid reference images
    valid_reference_images = []
    for ref_img in reference_images:
        if validate_image(ref_img):
            valid_reference_images.append(ref_img)
        else:
            print(f"Skipping invalid reference image: {ref_img}")
    
    if len(valid_reference_images) < len(reference_images):
        print(f"Note: {len(reference_images) - len(valid_reference_images)} reference image(s) were skipped due to validation failure.")

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return "LLM_API_KEY is not set"
    
    # Convert aspect ratio to size parameter
    size_map = {
        "1:1": "1024x1024",
        "4:3": "1024x768",
        "3:4": "768x1024",
        "16:9": "1024x576",
        "9:16": "576x1024",
    }
    size = size_map.get(aspect_ratio, "1024x1024")
    
    # 火山引擎 doubao-seedream 文生图 API 请求
    response = requests.post(
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/text2image/doubao-seedream-4-5-251128",
        headers={
            "Content-Type": "application/json",
        },
        params={
            "access_token": api_key
        },
        json={
            "prompt": prompt,
            "size": size,
            "n": 1,
            "steps": 20,
            "sampler_name": " Euler a",
            "cfg_scale": 7.0,
            "seed": -1
        },
    )
    response.raise_for_status()
    result = response.json()
    
    if "data" in result and len(result["data"]) > 0:
        base64_image = result["data"][0]["b64_image"]
        # Save the image to a file
        with open(output_file, "wb") as f:
            f.write(base64.b64decode(base64_image))
        return f"Successfully generated image to {output_file}"
    else:
        error_msg = result.get("error_msg", "Unknown error")
        raise Exception(f"Failed to generate image: {error_msg}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using Doubao Seedream API")
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="Absolute path to JSON prompt file",
    )
    parser.add_argument(
        "--reference-images",
        nargs="*",
        default=[],
        help="Absolute paths to reference images (space-separated)",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output path for generated image",
    )
    parser.add_argument(
        "--aspect-ratio",
        required=False,
        default="16:9",
        help="Aspect ratio of the generated image",
    )

    args = parser.parse_args()

    try:
        print(
            generate_image(
                args.prompt_file,
                args.reference_images,
                args.output_file,
                args.aspect_ratio,
            )
        )
    except Exception as e:
        print(f"Error while generating image: {e}")
