import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from deep_translator import GoogleTranslator
import json

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

def generate_caption(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image path {image_path} does not exist.")
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
    
    out = model.generate(**inputs, max_length=100, num_beams=4, do_sample=True)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

# Load ground truth captions from a CSV file
cap_df = pd.read_csv("Image_Captioning/captions.csv")
cap_eng_dict = dict(zip(cap_df['image_name'] + ".jpg", cap_df['english_caption']))
cap_vie_dict = dict(zip(cap_df['image_name'] + ".jpg", cap_df['vietnamese_caption']))

results = []
predictions_eng = {}
gts_eng = {}
predictions_vie = {}
gts_vie = {}

input_dir = "Image_Captioning/test_images"
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        try:
            caption_eng = generate_caption(image_path)
            caption_vie = GoogleTranslator(source='en', target='vi').translate(caption_eng)

            print(f"Caption EN for {filename}: {caption_eng}")
            print(f"Caption VI for {filename}: {caption_vie}")

            # Ground truth captions
            gt_eng = cap_eng_dict.get(filename, "")
            gt_vie = cap_vie_dict.get(filename, "")

            # Calculate BLEU score (English)
            bleu_eng = sentence_bleu([gt_eng.split()], caption_eng.split(), smoothing_function=SmoothingFunction().method1)
            print(f"BLEU EN for {filename}: {bleu_eng:.4f}")

            # Calculate BLEU score (Vietnamese)
            bleu_vie = sentence_bleu([gt_vie.split()], caption_vie.split(), smoothing_function=SmoothingFunction().method1)
            print(f"BLEU VI for {filename}: {bleu_vie:.4f}")

            # Prepare data for CIDEr (English)
            predictions_eng[filename] = [caption_eng]
            gts_eng[filename] = [gt_eng]

            # Prepare data for CIDEr (Vietnamese)
            predictions_vie[filename] = [caption_vie]
            gts_vie[filename] = [gt_vie]

            # Save result
            results.append({
                'image_name': filename,
                'caption_eng': caption_eng,
                'caption_vie': caption_vie,
                'ground_truth_eng': gt_eng,
                'ground_truth_vie': gt_vie,
                'bleu_score_eng': bleu_eng,
                'bleu_score_vie': bleu_vie
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Calculate CIDEr score (English)
cider_scorer = Cider()
cider_score_eng, _ = cider_scorer.compute_score(gts_eng, predictions_eng)
print(f"\nCIDEr EN score: {cider_score_eng:.4f}")

# Calculate CIDEr score (Vietnamese)
cider_score_vie, _ = cider_scorer.compute_score(gts_vie, predictions_vie)
print(f"CIDEr VI score: {cider_score_vie:.4f}")

# Save results to a JSON file
with open("Image_Captioning/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
