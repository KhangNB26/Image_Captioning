import torch 
from PIL import Image
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

# Function to generate caption for an image
def generate_caption_gr(image):
    if image is None:
        return "No image provided."
    
    image = image.convert("RGB")  # Ensure the image is in RGB format
    inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs, max_length=100, num_beams=4, do_sample=True)
    eng_caption = processor.decode(out[0], skip_special_tokens=True)
    # Translate the caption to Vietnamese
    vie_caption = GoogleTranslator(source='en', target='vi').translate(eng_caption)

    return eng_caption, vie_caption

# Gradio interface
iface = gr.Interface(
    fn = generate_caption_gr,
    inputs = gr.Image(type="pil", label="Upload an image"),
    outputs = [
        gr.Textbox(label="English Caption"),
        gr.Textbox(label="Vietnamese Caption")
    ],
    title = "Image Captioning with BLIP",
    description = "Upload an image to generate a caption using the BLIP model.",
)

# Launch the Gradio app
iface.launch()