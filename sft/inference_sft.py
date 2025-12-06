#!/usr/bin/env python3
"""
Simple inference script for fine-tuned Stable Diffusion LoRA.

Usage:
    # Text-to-image generation
    python inference_lora.py --lora_path gradient_logs/20251201-155339/epoch20 --prompt "A clear photo of a bird"
    
    # Image-to-image generation
    python inference_lora.py --lora_path gradient_logs/20251201-155339/epoch20 --prompt "A clear photo of a bird" --input_image path/to/image.jpg
"""

import argparse
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler,
)
from peft import PeftModel
from PIL import Image
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images using fine-tuned Stable Diffusion LoRA"
    )
    
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory (e.g., gradient_logs/20251201-155339/epoch20)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Optional input image path for img2img generation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="generated_image.png",
        help="Output path for generated image",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.4,
        help="Strength for img2img (0.0 = no change, 1.0 = full generation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--sd_model",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="Base Stable Diffusion model path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    
    return parser.parse_args()


def load_lora_pipeline(lora_path: str, sd_model: str, device: str, use_img2img: bool = False):
    """
    Load Stable Diffusion pipeline with fine-tuned LoRA.
    
    Args:
        lora_path: Path to LoRA checkpoint directory
        sd_model: Base Stable Diffusion model identifier
        device: Device to load model on
        use_img2img: Whether to use img2img pipeline
    
    Returns:
        Pipeline with LoRA loaded
    """
    print(f"Loading base Stable Diffusion model: {sd_model}")
    
    if use_img2img:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            sd_model,
            torch_dtype=torch.float16,
            safety_checker=None,  # Disable safety checker for faster inference
            requires_safety_checker=False,
        ).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_model,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
    
    # Use DDIM scheduler (same as training)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    print(f"Loading LoRA from: {lora_path}")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.unet.eval()
    
    print("LoRA loaded successfully!")
    return pipe


def generate_image(
    pipe,
    prompt: str,
    output_path: str,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int = None,
    input_image: Image.Image = None,
    strength: float = 0.4,
):
    """
    Generate an image using the pipeline.
    
    Args:
        pipe: Loaded diffusion pipeline
        prompt: Text prompt
        output_path: Where to save the image
        num_inference_steps: Number of diffusion steps
        guidance_scale: CFG scale
        seed: Random seed
        input_image: Optional input image for img2img
        strength: Strength for img2img
    """
    generator = None
    if seed is not None:
        generator = torch.manual_seed(seed)
    
    print(f"Generating image with prompt: '{prompt}'")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    
    if input_image is not None:
        print(f"Using img2img mode with strength: {strength}")
        image = pipe(
            prompt=prompt,
            image=input_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
    else:
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
    
    # Save image
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    
    return image


def main():
    args = parse_args()
    
    print("=" * 60)
    print("LoRA Inference Script")
    print("=" * 60)
    print(f"LoRA path: {args.lora_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Device: {args.device}")
    print()
    
    # Load input image if provided
    input_image = None
    if args.input_image:
        print(f"Loading input image: {args.input_image}")
        input_image = Image.open(args.input_image).convert("RGB")
        use_img2img = True
    else:
        use_img2img = False
    
    # Load pipeline with LoRA
    pipe = load_lora_pipeline(
        lora_path=args.lora_path,
        sd_model=args.sd_model,
        device=args.device,
        use_img2img=use_img2img,
    )
    
    # Generate image
    generate_image(
        pipe=pipe,
        prompt=args.prompt,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        input_image=input_image,
        strength=args.strength,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

