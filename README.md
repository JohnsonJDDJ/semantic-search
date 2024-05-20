# Semantic Search - Zhihao Du

Time taken - 103 min

## About this App

This is a simple web application for performing minimal semantic search on videos. Users can upload a video, perform semantic searches using text prompts, and view the results. The app consists of two main modules:

1. Key Frame Extraction
   
Performing semantic analysis on an entire video is expensive. Downsampling is preferred, as most of the information in a video can be captured by extracting key frames optimally. I implemented a **shot boundary detection** technique combined with **uniform sampling** at an interval of 30 frames.

Uniform sampling is effective for videos taken in a single shot, such as smartphone recordings. An interval of 30 frames is chosen because most videos are now recorded at 60fps, making half a second interval sufficient. Information that appears for less than half a second is rare, as it is difficult for humans to capture such brief details.

For videos with lower fps or high information density, shot boundary detection helps capture important information. This technique captures frames that differ significantly from previous frames, indicating a change in semantics. It is particularly effective in cases like video game speedruns with high information density.

2. Text and Image Embeddings using CLIP

Modern models can perform accurate semantic searches, especially with LLMs. However, LLMs can be expensive and involve restrictive APIs, making them less ideal for distribution. Therefore, I used CLIP to generate embeddings for texts and images (key frames).

The results are ranked by the highest cosine similarity between embeddings. Users can specify the number of results to query by adjusting the `k` value in the app.

## Instructions

### 1. Clone Repo

```bash
git clone https://github.com/JohnsonJDDJ/semantic-search.git
cd semantic-search
```

### 2. Dependencies

```bash
python -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python run.py
```

Open http://127.0.0.1:8787 in a web browser.