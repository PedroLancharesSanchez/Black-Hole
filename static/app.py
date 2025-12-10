from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel, AutoProcessor
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os
import io
import base64
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import threading

app = Flask(__name__, static_folder='static')

# Global variables to store processed data
processed_data = {
    'images': [],
    'embeddings': [],
    'pca_coords': [],
    'neighbors': [],
    'image_paths': []
}

# Variable to ensure content window reference is kept
root = None

def get_folder_path():
    """Open folder dialog in the main thread"""
    # Create a hidden root window
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Make sure dialog appears on top
        folder_selected = filedialog.askdirectory()
        root.destroy()
        return folder_selected
    except Exception as e:
        print(f"Error opening dialog: {e}")
        return None

# Load ResNet-50 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global model cache to avoid reloading
model_cache = {}

def get_model_and_processor(model_name):
    """Load model and processor based on name"""
    if model_name in model_cache:
        return model_cache[model_name]
        
    print(f"Loading model: {model_name}...")
    
    if model_name == 'clip':
        # OpenAI CLIP
        model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device)
        
    elif model_name == 'siglip':
        # Google SigLIP
        model_id = "google/siglip-base-patch16-224"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        

    elif model_name == 'resnet':
        # ResNet-50 from torchvision
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
        model = model.to(device)
        model.eval()
        
        processor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else: 
        # Fallback to ResNet if unknown
        if 'resnet' in model_cache:
            return model_cache['resnet']
        return get_model_and_processor('resnet')

    model_cache[model_name] = (model, processor)
    return model, processor

def extract_features(model_name, image_path):
    """Extract features using specific model"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Load model and processor using valid function
        model, processor = get_model_and_processor(model_name)
        
        if model_name == 'resnet' or model_name not in ['clip', 'siglip']:
             # Force get resnet if unknown
             if model_name not in ['resnet', 'clip', 'siglip']:
                 model_name = 'resnet'
                 
             # ResNet uses torchvision transforms
             model, processor = get_model_and_processor('resnet')
             
             # Support both PIL Image and file path (though function takes path, let's be safe)
             if isinstance(image, str):
                 image = Image.open(image).convert("RGB")
                 
             img_tensor = processor(image).unsqueeze(0).to(device)
             with torch.no_grad():
                 embedding = model(img_tensor)
             return embedding.cpu().numpy().flatten()
            
        else:
            # Load from cache
            model, processor = get_model_and_processor(model_name)
            
            # Process image
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                if model_name == 'clip':
                    outputs = model.get_image_features(**inputs)
                elif model_name == 'siglip':
                    outputs = model.get_image_features(**inputs) 

                else:
                    outputs = model(**inputs).last_hidden_state.mean(dim=1)
            
            return outputs.cpu().numpy().flatten()
            
    except Exception as e:
        print(f"Error extracting features with {model_name}: {e}")
        return None


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/select-folder', methods=['POST'])
def select_folder():
    """Open system dialog to select folder"""
    try:
        # We need to run this carefully. Tkinter must run in main thread usually. 
        # But Flask runs in threads. 
        # However, for a local tool, calling it directly often works if main thread isn't blocked 
        # or if we are lucky. If it fails, we might need a different approach.
        # Let's try direct call first.
        
        folder_path = get_folder_path()
        
        if not folder_path:
            return jsonify({'canceled': True}), 200
            
        return jsonify({'folder_path': folder_path, 'canceled': False})
    except Exception as e:
        print(f"Error selecting folder: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-images', methods=['POST'])
def load_images():
    """Load images from folder, compute embeddings, apply PCA, and find nearest neighbors"""
    
    # Needs to be extracted before the stream starts
    data = request.json
    folder_path = data.get('folder_path')
    model_name = data.get('model', 'resnet')
    # Default method to pca since UI sends model, but logic assumed method. We just rely on standard from now
    
    def generate():
        try:
            if not folder_path or not os.path.exists(folder_path):
                yield json.dumps({'type': 'error', 'message': 'Invalid folder path'}) + '\n'
                return
            
            # Supported image extensions
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
            
            # Find all images in folder
            image_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        image_files.append(os.path.join(root, file))
            
            if len(image_files) == 0:
                yield json.dumps({'type': 'error', 'message': 'No images found'}) + '\n'
                return
            
            yield json.dumps({'type': 'progress', 'value': 0, 'message': f"Found {len(image_files)} images"}) + '\n'
            
            # Extract embeddings
            embeddings = []
            valid_image_paths = []
            
            yield json.dumps({'type': 'progress', 'value': 5, 'message': f"Loading model {model_name}..."}) + '\n'
            
            for i, img_path in enumerate(image_files):
                # Calculate progress
                progress = 10 + int((i / len(image_files)) * 80)
                
                # Yield progress occasionally
                if i % 2 == 0 or i == len(image_files) - 1:
                    filename = os.path.basename(img_path)
                    yield json.dumps({
                        'type': 'progress', 
                        'value': progress, 
                        'message': f"Processing {filename} ({i+1}/{len(image_files)})..."
                    }) + '\n'

                embedding = extract_features(model_name, img_path)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_image_paths.append(img_path)
            
            if len(embeddings) < 2:
                yield json.dumps({'type': 'error', 'message': 'Not enough valid images'}) + '\n'
                return
            
            embeddings = np.array(embeddings)
            
            # Apply dimensionality reduction
            yield json.dumps({'type': 'progress', 'value': 90, 'message': "Applying dimensionality reduction..."}) + '\n'
            
            n_components = min(2, len(embeddings))
            # Default to PCA for stability
            pca = PCA(n_components=n_components)
            pca_coords = pca.fit_transform(embeddings)

            # Find k nearest neighbors
            yield json.dumps({'type': 'progress', 'value': 95, 'message': "Finding neighbors..."}) + '\n'
            k = min(11, len(pca_coords)) 
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(pca_coords)
            distances, indices = nbrs.kneighbors(pca_coords)
            
            # Build neighbor connections
            neighbor_connections = []
            for i, neighbor_list in enumerate(indices):
                for j, neighbor_idx in enumerate(neighbor_list[1:]):  # Skip first (self)
                    neighbor_connections.append({
                        'source': int(i),
                        'target': int(neighbor_idx)
                    })
            
            # Store processed data globally
            processed_data['embeddings'] = embeddings
            processed_data['pca_coords'] = pca_coords
            processed_data['neighbors'] = neighbor_connections
            processed_data['image_paths'] = valid_image_paths
            
            # Prepare final result
            points = []
            for i, (coord, img_path) in enumerate(zip(pca_coords, valid_image_paths)):
                points.append({
                    'id': i,
                    'x': float(coord[0]),
                    'y': float(coord[1]),
                    'filename': os.path.basename(img_path),
                    'path': img_path
                })
            
            response_data = {
                'points': points,
                'connections': neighbor_connections,
                'total_images': len(valid_image_paths),
                'folder_path': folder_path
            }
            
            yield json.dumps({'type': 'result', 'data': response_data}) + '\n'
            
        except Exception as e:
            print(f"Error in stream: {e}")
            import traceback
            traceback.print_exc()
            yield json.dumps({'type': 'error', 'message': str(e)}) + '\n'

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

@app.route('/api/get-image/<int:image_id>', methods=['GET'])
def get_image(image_id):
    """Get image data for display"""
    try:
        if image_id >= len(processed_data['image_paths']):
            return jsonify({'error': 'Invalid image ID'}), 400
        
        image_path = processed_data['image_paths'][image_id]
        
        # Read image and convert to base64
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Determine image type
        ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp'
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        return jsonify({
            'image': f"data:{mime_type};base64,{image_base64}",
            'filename': os.path.basename(image_path)
        })
    
    except Exception as e:
        print(f"Error getting image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-labels', methods=['POST'])
def export_labels():
    """Export labels to CSV"""
    try:
        data = request.json
        polygons = data.get('polygons', [])
        
        # Create a mapping of point index to label
        point_labels = {}
        for polygon in polygons:
            label = polygon.get('name', 'unlabeled')
            points = polygon.get('points', [])
            for point_id in points:
                point_labels[point_id] = label
        
        # Build CSV data
        csv_data = []
        for i, img_path in enumerate(processed_data['image_paths']):
            filename = os.path.basename(img_path)
            label = point_labels.get(i, 'unlabeled')
            csv_data.append({'filename': filename, 'label': label})
        
        # Create DataFrame and convert to CSV
        df = pd.DataFrame(csv_data)
        csv_string = df.to_csv(index=False)
        
        return jsonify({
            'csv': csv_string,
            'filename': 'image_labels.csv'
        })
    
    except Exception as e:
        print(f"Error exporting labels: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Threaded=False might be needed for Tkinter in some environments but usually it's blocking anyway
    app.run(debug=True, host='0.0.0.0', port=5000)
