from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import cv2
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["ALLOWED_EXTENSIONS"] = {"mp4", "avi", "mov", "mkv"}

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )

def clear_uploads_folder():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename == '.gitkeep': # Keep .gitkeep for git
            continue
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

def extract_key_frames(video_path, interval=30):
    cap = cv2.VideoCapture(video_path)
    prev_hist = None
    key_frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Uniform sampling
        if frame_count % interval == 0:
            key_frames.append(frame)

        # Shot bounday detection 
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if hist_diff < 0.5:  # Threshold
                key_frames.append(frame)

        prev_hist = hist
        frame_count += 1

    cap.release()

    return key_frames

def get_image_embeddings(key_frames, model, preprocess, device):
    image_embeddings = []

    for frame in tqdm(key_frames):
        # Convert OpenCV image (BGR) to PIL image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Preprocess the image
        preprocessed_image = preprocess(pil_image).unsqueeze(0).to(device)

        # Get the image embedding
        with torch.no_grad():
            image_embedding = model.encode_image(preprocessed_image)
        
        image_embeddings.append(image_embedding.cpu().numpy())

    image_embeddings_tensor = torch.tensor(np.vstack(image_embeddings))
    return image_embeddings_tensor

def get_text_embedding(text, model, device):
    # Preprocess the input text
    text_inputs = clip.tokenize([text]).to(device)
    
    # Get the text embedding
    with torch.no_grad():
        text_embedding = model.encode_text(text_inputs)
    
    return text_embedding.cpu()

def cosine_sim(x, y):
    x = x / x.norm(dim=1, keepdim=True)
    y = y / y.norm(dim=1, keepdim=True)
    return x @ y.T # dot prod

def top_k_similarity(image_embeddings, text_embedding, k=5):
    similarities = cosine_sim(image_embeddings, text_embedding)
    if k == 1:
        top_k_indices = torch.topk(similarities, k, dim=0).indices.item()
        return [top_k_indices]
    else:
        top_k_indices = torch.topk(similarities, k, dim=0).indices.squeeze().cpu().numpy()
        return top_k_indices.tolist()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_and_search', methods=['POST'])
def upload_and_search():
    file = request.files['file']
    if not file or not file.filename:
        return redirect(request.url)
    elif file and allowed_file(file.filename):
        clear_uploads_folder()  # clear everytime
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        key_frames = extract_key_frames(file_path)
        image_embeddings = get_image_embeddings(key_frames, model, preprocess, device)
        torch.save(image_embeddings, os.path.join(app.config['UPLOAD_FOLDER'], 'image_embeddings.pt'))

        # Save the key frames as images
        for i, frame in enumerate(key_frames):
            frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f'key_frame_{i}.jpg')
            cv2.imwrite(frame_path, frame)
        video_url = url_for('uploaded_file', filename=filename)
        return jsonify({'message': 'Processing completed', 'video_url': video_url}), 200
    else:
        return jsonify({'error': 'Allowed file types are mp4, avi, mov, mkv'}), 400

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    k = int(request.form.get('k', 1))  # get k from form
    image_embeddings = torch.load(os.path.join(app.config['UPLOAD_FOLDER'], 'image_embeddings.pt'))
    text_embedding = get_text_embedding(query, model, device)

    top_k_indices = top_k_similarity(image_embeddings, text_embedding, k=k)

    top_k_frame_paths = [url_for('uploaded_file', filename=f'key_frame_{i}.jpg') for i in top_k_indices]

    return jsonify(top_k_frame_paths)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    app.run(debug=True)