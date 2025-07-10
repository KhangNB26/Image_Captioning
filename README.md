# Image Captioning

This project provides an end-to-end pipeline for generating image captions in English and Vietnamese using the BLIP model, with a web interface for interactive use and scripts for evaluation.

## Why BLIP?

I chose the **BLIP (Bootstrapped Language Image Pretraining)** model because it is a great quality and small model, suitable for most of devices. BLIP is pre-trained on large-scale image-text datasets, enabling it to generate accurate and contextually rich captions for a wide variety of images. It's an open-source model, contains efficient architecture, easy to integrate with Hugging Face Transformers. Lastly, it works well for multilingual applications (including English, Vietnamese, ...).

## Features

- **Image Captioning**: Generate captions for images using the BLIP model (`Salesforce/blip-image-captioning-base`).
- **Translation**: Automatically translates English captions to Vietnamese using Google Translate.
- **Web Interface**: User-friendly Gradio app for uploading images and viewing captions.
- **Evaluation**: Scripts to evaluate generated captions against ground truth using BLEU and CIDEr metrics.

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/KhangNB26/Image_Captioning.git
    cd Image_Captioning
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Web Demo

Run the Gradio app to generate captions interactively:

```bash
python Image_Captioning/app.py
```

Web will run at http://127.0.0.1:7860, you can upload image from your device to use the app.

### 2. Evaluation

Run the provided script to evaluate the "Salesforce/blip-image-captioning-base" model:

```bash
python Image_Captioning/model/blip.py
```

## Production Improvement Proposal

- **Use better GPU**
- **Convert inference to ONNX**
- **Apply quantization 8-bit/4-bit**
- **For an application that server more people, consider deploying using API**