# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import shutil
import tempfile
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
TEMP_DIR = 'temp'
MODEL_DIR = 'stargan/models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/api/transform', methods=['POST'])
def transform_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Generate unique ID for this process
        process_id = str(uuid.uuid4())
        
        # Create directories for this process
        process_dir = os.path.join(TEMP_DIR, process_id)
        single_test_dir = os.path.join(process_dir, 'single_test_image')
        os.makedirs(single_test_dir, exist_ok=True)
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1]
        saved_path = os.path.join(single_test_dir, f"00001{ext}")
        file.save(saved_path)
        
        # Get target age from request
        target_age = request.form.get('targetAge', '20-29')
        
        # Create CSV file for testing
        csv_path = os.path.join(process_dir, 'single_test.csv')
        create_test_csv(saved_path, csv_path, target_age)
        
        # Run the model
        result_images = run_model(single_test_dir, csv_path, process_id)
        
        # Return URLs to the resulting images
        return jsonify({
            'processId': process_id,
            'originalImage': f'/api/images/{process_id}/original',
            'transformedImages': {age: f'/api/images/{process_id}/age/{age}' for age in result_images},
            'message': 'Image processing completed'
        })

@app.route('/api/images/<process_id>/original', methods=['GET'])
def get_original_image(process_id):
    result_dir = 'stargan/results'
    try:
        # Look for the original image file (e.g., 1-original.jpg)
        image_path = os.path.join(result_dir, '1-original.jpg')
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Original image not found'}), 404
    except Exception as e:
        print(f"Error serving original image: {e}")
        return jsonify({'error': 'Internal server error'}), 500


from urllib.parse import unquote

@app.route('/api/images/<process_id>/age/<age_group>', methods=['GET'])
def get_age_image(process_id, age_group):
    result_dir = 'stargan/results'
    
    try:
        # Decode age group in case it was URL-encoded (e.g., 70%2B to 70+)
        age_group = unquote(age_group)

        # Construct the filename directly (e.g., 1-age-20-29.jpg)
        filename = f'1-age-{age_group}.jpg'
        image_path = os.path.join(result_dir, filename)

        if not os.path.exists(image_path):
            return jsonify({'error': f'Transformed image not found: {filename}'}), 404

        return send_file(image_path, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error serving transformed image: {e}")
        return jsonify({'error': 'Internal server error'}), 500


def create_test_csv(image_path, csv_path, age_group):
    """Create a CSV file for the test image."""
    # The model uses this format:
    # image_number,age_group_original,age_group_binned,age_group_confidence,gender,gender_confidence,aligned_path
    data = {
        'image_number': [1, 1],  # Duplicate entry to ensure at least one goes to test set
        'age_group_original': [age_group, age_group],
        'age_group_binned': [age_group, age_group],
        'age_group_confidence': [0.9, 0.9],
        'gender': ['Unknown', 'Unknown'],
        'gender_confidence': [0.5, 0.5],
        'aligned_path': [image_path, image_path]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path

def run_model(image_dir, csv_path, process_id):
    """Run the StarGAN model on the input image."""
    # Call the modified main.py with our parameters
    cmd = [
        'python', 'stargan/main.py',
        '--mode', 'test',
        '--ffhq_image_dir', image_dir,
        '--ffhq_attr_path', csv_path,
        '--result_dir', 'stargan/results',
        '--test_iters', '300000'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        
        # Get the list of generated images
        result_dir = 'stargan/results'
        age_groups = ['0-5', '6-12', '13-19', '20-29', '30-39', '40-49', '50-69', '70+']
        result_images = {}
        
        for age in age_groups:
            img_path = os.path.join(result_dir, f'1-age-{age}.jpg')
            if os.path.exists(img_path):
                result_images[age] = age
                
        return result_images
        
    except subprocess.CalledProcessError as e:
        print(f"Error running model: {e}")
        return {}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)