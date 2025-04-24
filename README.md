# Smart Face Age Editing with GANs

![Age Progression Example](https://github.com/user/smart-face-age-editing/raw/main/examples/age_progression_example.jpg)

## Overview

Smart Face Age Editing is a facial age progression and regression system that transforms a person's appearance across different age groups using Generative Adversarial Networks (GANs). The system can generate realistic age-transformed versions of the same person across eight distinct age categories: 0-5, 6-12, 13-19, 20-29, 30-39, 40-49, 50-69, and 70+ years.

This project uses a modified StarGAN architecture to perform multi-domain image-to-image translation while preserving the identity of the subject.

## Features

- Transform facial images across 8 age categories
- Preserve identity-specific features during transformation
- User-friendly web interface for uploading images and selecting target ages
- Realistic simulation of age-related changes (skin texture, facial structure, hair patterns)
- REST API for integration with other applications

## Architecture

The system consists of:

1. **GAN Model**: Modified StarGAN architecture with:
   - Generator network that takes an input image and target age label
   - Discriminator network that classifies images as real/fake and determines age category
   - Custom loss functions for realism, identity preservation, and age accuracy

2. **Web Application**:
   - Frontend: React.js
   - Backend: Flask API with Python

3. **Face Processing Pipeline**:
   - Face detection and alignment using MTCNN
   - Image preprocessing for optimal model input

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA-compatible GPU (recommended for model inference)
- Node.js and npm for the web interface

### Setup

1. Clone the repository:
```bash
git clone https://github.com/user/smart-face-age-editing.git
cd smart-face-age-editing
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

4. Download the pre-trained model from Google Drive
```bash
mkdir -p stargan/models
```
# Download G model from: [https://drive.google.com/file/d/YOUR_G_MODEL_ID/view?usp=sharing](https://drive.google.com/file/d/1GYoW5Ge7Up9d6cItF-PuFI3jyz5Dodp8/view?usp=sharing)
# Download D model from: [https://drive.google.com/file/d/YOUR_D_MODEL_ID/view?usp=sharing](https://drive.google.com/file/d/10JSPYbhiTz4ZvKTuO_U7e0dj5YgGDWmT/view?usp=sharing)

## Usage

### Running the Web Application

1. Start the backend server:
```bash
python app.py
```

2. In a separate terminal, start the frontend:
```bash
cd frontend
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

### API Usage

The system provides a REST API for integration with other applications:

```
POST /api/transform
Content-Type: multipart/form-data

Parameters:
- image: The image file to transform
- targetAge: Target age category (one of: "0-5", "6-12", "13-19", "20-29", "30-39", "40-49", "50-69", "70+")

Response:
{
  "processId": "unique-process-id",
  "originalImage": "/api/images/process-id/original",
  "transformedImages": {
    "0-5": "/api/images/process-id/age/0-5",
    "6-12": "/api/images/process-id/age/6-12",
    ...
  },
  "message": "Image processing completed"
}
```

## Training Your Own Model

If you want to train the model on your own data:

1. Prepare your dataset:
   - Images should be aligned facial images
   - Create a CSV file with columns: image_number, age_group_original, age_group_binned, age_group_confidence, gender, gender_confidence, aligned_path

2. Run the training script:
```bash
python main.py --mode train --dataset FFHQ --batch_size 16 --num_iters 200000 --ffhq_image_dir your_aligned_faces --ffhq_attr_path your_dataset.csv
```

## Dataset

The model was trained on the Flickr-Faces-HQ (FFHQ) dataset with age labels. The dataset consists of 70,000 high-quality images with considerable variation in terms of age, ethnicity, and image background. Images were aligned using MTCNN facial detection and alignment, and resized to 128×128 pixels.

## Evaluation

The model was evaluated using:
- Qualitative assessment of visual results
- Loss convergence analysis during training
- Identity preservation metrics

## Limitations

- Works best with frontal facial images with neutral expressions
- May struggle with significant occlusions (glasses, heavy makeup)
- Limited ability to remove facial hair when converting to younger age groups
- Performance varies based on the quality of input images

## Acknowledgments

- The FFHQ dataset for providing high-quality facial images
- The original StarGAN paper and implementation which formed the basis of our approach
