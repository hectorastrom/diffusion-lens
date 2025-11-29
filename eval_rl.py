# @Time    : 2025-11-29 13:12
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : eval_rl.py
"""
Goal: Evaluate the performance of the diffusion + CLIP combo on the COD dataset.
"""
from COD_dataset import build_COD_torch_dataset
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from torch.utils.data import DataLoader

test_set = build_COD_torch_dataset(split_name="test")

device = "cuda"
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

###############################
# Training run hyperparams
###############################
strength = 0.4
guidance = 7.0
lora_path = "./weights/robust-totem-89/epoch189/lora.safetensors"
prompt = ""

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to(device)

# loads onto UNet (most likely) - whatever sd_pipeline.get_trainable_layers()
# returns & trained LoRA on top of in DDPO
pipe.load_lora_weights(lora_path)

test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# pull much of this from img2img.py and clip_classifier.py
for item in test_set:
    img_tensors = item["pixel_values"].to(pipe.device)
    label = item["label"]  # int
    print(f"Original image: {item['image_path']}")
    generations = pipe(
        prompt,
        img_tensors,
        strength=strength,
        guidance_scale=guidance,
        # FIXME: uh oh - not sure what the right value for this is. In ddpo.py, we had to
        # patch num_inference_steps to depend on strength, otherwise our latent and
        # reward tensors were of the wrong dim. But Img2ImgPipeline PROBABLY already
        # does this for us. so we might be double-applying strength here?
        num_inference_steps=50 * strength,
        output_type='pil'
    ).images
    break

gen_image = generations[0]
gen_image.save('outputs/generated_img.png')
print("Saved generated_img.png")

