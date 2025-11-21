import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


def load_caption_model():
    """
    Load the image captioning model (BLIP).
    """
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


def generate_caption(image_path, processor, model, device):
    """
    Generate a caption for a single image.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def caption_png_in_folder(folder_path, output_jsonl_path):
    """
    Generate captions for all PNG images in the specified folder,
    and save the results as JSONL: { "file_name": ..., "text": ... }
    """
    # Load model once
    print("Loading caption model...")
    processor, model, device = load_caption_model()
    print("Model loaded.")

    # Collect PNG files
    png_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".png")
    ]

    if not png_files:
        print("No PNG files found.")
        return

    with open(output_jsonl_path, "w", encoding="utf-8") as f_out:
        for filename in png_files:
            image_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")

            try:
                caption = generate_caption(image_path, processor, model, device)
                record = {
                    "file_name": filename,
                    "text": caption
                }
                # Write one JSON object per line
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"  → caption: {caption}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Done. Output saved to: {output_jsonl_path}")


if __name__ == "__main__":
    target_folder = "dataset/ghost_in_the_shell" 
    output_jsonl = f"{target_folder}/metadata.jsonl"

    caption_png_in_folder(target_folder, output_jsonl)
