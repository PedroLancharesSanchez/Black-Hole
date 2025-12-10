from flask import Flask, request, jsonify, send_from_directory
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
except ImportError:
    umap = None
    print("UMAP not installed")
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

# Load pre-trained ResNet-50 and remove the final classification layer
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
resnet = resnet.to(device)
resnet.eval()

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(image_path):
    """Extract embedding from an image using ResNet-50"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = resnet(img_tensor)
        
        # Flatten the embedding
        embedding = embedding.cpu().numpy().flatten()
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
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
    try:
        data = request.json
        folder_path = data.get('folder_path')
        
        if not folder_path or not os.path.exists(folder_path):
            return jsonify({'error': 'Invalid folder path'}), 400
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
        # Find all images in folder
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        if len(image_files) == 0:
            return jsonify({'error': 'No images found in folder'}), 400
        
        print(f"Found {len(image_files)} images")
        
        # Extract embeddings
        embeddings = []
        valid_image_paths = []
        
        for i, img_path in enumerate(image_files):
            embedding = get_image_embedding(img_path)
            if embedding is not None:
                embeddings.append(embedding)
                valid_image_paths.append(img_path)
            
            # Send progress update (using print for now, could use SSE)
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
        
        if len(embeddings) < 2:
            return jsonify({'error': 'Not enough valid images for PCA'}), 400
        
        embeddings = np.array(embeddings)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Apply dimensionality reduction
        method = data.get('method', 'pca')
        n_components = min(2, len(embeddings))
        
        print(f"Applying dimensionality reduction using: {method}")
        
        if method == 'tsne':
            # t-SNE
            # Perplexidad debe ser menor que n_samples
            perplexity = min(30, len(embeddings) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
            pca_coords = tsne.fit_transform(embeddings)

        elif method == 'umap':
            # UMAP
            if umap is None:
                # Fallback to PCA if UMAP missing
                print("UMAP requested but not installed. Falling back to PCA.")
                pca = PCA(n_components=n_components)
                pca_coords = pca.fit_transform(embeddings)
            else:
                reducer = umap.UMAP(n_components=2, random_state=42)
                pca_coords = reducer.fit_transform(embeddings)
                
        else:
            # Default to PCA
            pca = PCA(n_components=n_components)
            pca_coords = pca.fit_transform(embeddings)
            
        print(f"Coords shape: {pca_coords.shape}")
        
        # Find k nearest neighbors for each point
        # We want 10 neighbors + self = 11
        k = min(11, len(pca_coords)) 
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(pca_coords)
        distances, indices = nbrs.kneighbors(pca_coords)
        
        # Build neighbor connections (exclude self)
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
        
        # Prepare response
        points = []
        for i, (coord, img_path) in enumerate(zip(pca_coords, valid_image_paths)):
            points.append({
                'id': i,
                'x': float(coord[0]),
                'y': float(coord[1]),
                'filename': os.path.basename(img_path),
                'path': img_path
            })
        
        response = {
            'points': points,
            'connections': neighbor_connections,
            'total_images': len(valid_image_paths),
            'folder_path': folder_path
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in load_images: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

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
