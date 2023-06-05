import json
import torch
import transformers
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from itertools import batched

def load_captions(cap_json):
    with open(cap_json, 'r') as captions_file:
        captions_data = json.load(captions_file)

    imgIDToCap = {int(i['image_id']): i['caption'] for i in captions_data['annotations']}
    captions = []
    for id in [int(i['id']) for i in captions_data['images']]:
        captions.append(imgIDToCap[id])
    imgIDs = [i['id'] for i in captions_data['images']]
    return captions, imgIDs

def main():
    captions, imgIDs = load_captions('/nlp/data/vision_datasets/coco/annotations/captions_train2014.json')

    # Load Stable Diffusion Model and generate images
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    
    batch_size = 256
    i = 0

    for batch in batched(captions, batch_size):
        images = pipe(batch, num_inference_steps=20).images
        for image in images:
            image.save(f'/nlp/data/rhuang99/fakenews/sd_coco/{imgIDs[i]}.jpg')
            i += 1

if __name__ == "__main__":
    main()