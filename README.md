# SotaVideoRAG
A complete, state-of-the-art VideoRAG implementation!
##
# 🎥 SotaVideoRAG: Multi-Modal Video Retrieval with FAISS

Advanced video understanding and retrieval system using Qwen3-VL models, FAISS indexing, and multi-modal search.

## ✨ Features

- 🎬 **Hierarchical Video Processing**: Scene detection, keyframe extraction, multi-modal encoding
- 💾 **FAISS Persistent Indexing**: Process once, query forever with lightning-fast vector search
- 🔍 **Text Search**: Natural language queries to find specific moments
- 🖼️ **Image Search**: Upload images to find visually similar frames
- 🎯 **Multi-Modal Reranking**: Advanced relevance scoring with vision-language models
- 🤖 **AI-Powered Answers**: Context-aware response generation using Qwen3-VL

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              Video Input (MP4, AVI, etc.)        │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│     Scene Detection & Keyframe Extraction        │
│  • Histogram-based scene boundaries              │
│  • Uniform keyframe sampling                     │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│        Multi-Modal Encoding (Qwen3-VL)          │
│  • Caption generation (Ollama)                   │
│  • Visual embeddings (Qwen3-VL-Embedding)        │
│  • Text embeddings (Qwen3-VL-Embedding)          │
│  • Audio transcription (optional)                │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│           FAISS Indexing & Storage               │
│  • 2048-dim normalized vectors                   │
│  • IndexFlatIP (exact cosine similarity)         │
│  • Persistent disk storage                       │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│              Query Processing                    │
│  • Text queries → embedding                      │
│  • Image queries → embedding                     │
│  • FAISS similarity search (<10ms)               │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│      Multi-Modal Reranking (Optional)           │
│  • Qwen3-VL-Reranker scores image+text pairs    │
│  • Combines retrieval + reranking scores         │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│           Answer Generation                      │
│  • Context assembly from top results            │
│  • Qwen3-VL generates final answer              │
└─────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 20GB+ free disk space

### Step 1: Clone and Setup

```bash
git clone <your-repo>
cd videorag
mkdir scripts
touch scripts/__init__.py
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt

# For GPU-accelerated FAISS (optional but recommended):
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Step 3: Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Qwen3-VL model
ollama pull qwen3-vl

# Start Ollama server (keep this running)
ollama serve
```

### Step 4: Download Model Scripts

Place these files in the `scripts/` directory:
- `qwen3_vl_embedding.py`
- `qwen3_vl_reranker.py`
- `__init__.py` (empty file)

## 🚀 Quick Start

### Start the Application

```bash
python videorag_app.py
```

Open your browser to `http://localhost:7860`

### Basic Usage

1. **Settings Tab**:
   - Click "Initialize Models" (first time only)
   - Wait for models to load

2. **Process Video Tab**:
   - Upload your video
   - Adjust FPS (1.0 recommended)
   - Click "Process Video"
   - Index is saved automatically

3. **Text Search Tab**:
   - Ask questions: "What activities are shown?"
   - Get AI-generated answers with relevant frames

4. **Image Search Tab**:
   - Upload a reference image
   - Find visually similar frames in the video

## 📖 Usage Examples

### Example 1: Text Search

```python
# In the Text Search tab:
Query: "Show me scenes with people talking"
Top K: 5
Use reranking: ✓

# Results: Relevant frames + AI answer
```

### Example 2: Image Search

```python
# In the Image Search tab:
1. Upload image of a dog
2. Click "Find Similar Frames"
3. Get all frames with similar dogs
```

### Example 3: Programmatic Usage

```python
from videorag_app import VideoRAG
from PIL import Image

# Initialize
rag = VideoRAG(
    embedding_model_path="Qwen/Qwen3-VL-Embedding-2B",
    reranker_model_path="Qwen/Qwen3-VL-Reranker-2B"
)

# Process video (once)
rag.process_video("my_video.mp4", fps=1.0)

# Text search
results = rag.search_with_text("Find outdoor scenes", top_k=5)

# Image search
query_image = Image.open("reference.jpg")
results = rag.search_with_image(query_image, top_k=5)

# Rerank
reranked = rag.rerank_results("outdoor scenes", results)

# Generate answer
answer = rag.generate_answer("What's happening?", reranked)
```

## 🎯 Models Used

| Model | Purpose | Size | Link |
|-------|---------|------|------|
| **Qwen3-VL** | Caption generation, Answer generation | Via Ollama | [Ollama](https://ollama.com) |
| **Qwen3-VL-Embedding-2B** | Multi-modal embeddings | ~4GB | [HF](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) |
| **Qwen3-VL-Embedding-8B** | Higher quality embeddings | ~16GB | [HF](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) |
| **Qwen3-VL-Reranker-2B** | Multi-modal reranking | ~4GB | [HF](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) |
| **Qwen3-VL-Reranker-8B** | Higher quality reranking | ~16GB | [HF](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B) |

## 💾 Storage & Caching

### FAISS Index Files

Processed videos are cached in `./video_indexes/`:

```
./video_indexes/
├── abc123def456.faiss    # FAISS vector index
├── abc123def456.json     # Metadata (segments, captions, etc.)
└── xyz789ghi012.faiss    # Another video's index
```

### Storage Requirements

- **Per video**: ~8-15 MB per 1000 frames
- **1-minute video @ 1 FPS**: ~500 KB - 1 MB
- **10-minute video @ 1 FPS**: ~5-10 MB

### Cache Management

In the Settings tab:
- **View Cache Info**: See all cached videos and sizes
- **Clear All Cache**: Delete all indexes (forces reprocessing)

## ⚙️ Configuration

### Model Selection

**For Development/Testing** (Lower VRAM):
```python
VideoRAG(
    embedding_model_path="Qwen/Qwen3-VL-Embedding-2B",
    reranker_model_path="Qwen/Qwen3-VL-Reranker-2B"
)
```

**For Production/Best Quality** (Higher VRAM):
```python
VideoRAG(
    embedding_model_path="Qwen/Qwen3-VL-Embedding-8B",
    reranker_model_path="Qwen/Qwen3-VL-Reranker-8B"
)
```

### Processing Parameters

- **FPS**: 0.5-2.0 (1.0 recommended)
  - Lower = faster processing, less detail
  - Higher = slower processing, more detail

- **Use Cache**: Always on unless you want to reprocess

- **Top K**: 3-10 results (5 recommended)

- **Reranking**: Enable for best quality, disable for speed

## 🔧 Troubleshooting

### Models Not Loading

```bash
# Ensure transformers is up to date
pip install transformers>=4.57.0 --upgrade

# Check for qwen-vl-utils
pip install qwen-vl-utils>=0.0.14
```

### Ollama Connection Error

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve

# Verify Qwen3-VL is installed
ollama list | grep qwen3-vl
```

### Out of Memory

**Solutions:**
1. Use smaller models (2B instead of 8B)
2. Reduce FPS (try 0.5)
3. Process shorter videos
4. Close other applications

### FAISS Installation Issues

```bash
# CPU version (easier to install)
pip install faiss-cpu

# GPU version (better performance)
conda install -c conda-forge faiss-gpu
```

### Import Errors

```bash
# Make sure scripts/__init__.py exists
touch scripts/__init__.py

# Check Python path
python -c "import sys; print(sys.path)"
```

## 📊 Performance

### Processing Time (First Time)

| Video Length | FPS | Time (2B models) | Time (8B models) |
|--------------|-----|------------------|------------------|
| 1 minute     | 1.0 | ~2-3 min         | ~4-6 min         |
| 5 minutes    | 1.0 | ~10-15 min       | ~20-30 min       |
| 10 minutes   | 1.0 | ~20-30 min       | ~40-60 min       |

### Search Time (After Indexing)

| Operation | Time |
|-----------|------|
| Load index from cache | <1 second |
| Query embedding | <1 second |
| FAISS search (10K frames) | <10ms |
| Reranking (5 results) | ~1-2 seconds |
| Answer generation | ~2-5 seconds |
| **Total search time** | **~3-8 seconds** |

## 🎓 Advanced Features

### Custom Embedding Dimensions

Qwen3-VL-Embedding supports Matryoshka Representation Learning:

```python
# Use smaller dimensions for faster search
embeddings = embedder.process(inputs, dim=1024)  # Instead of 2048
```

### Batch Processing Multiple Videos

```python
import glob

video_files = glob.glob("videos/*.mp4")
for video in video_files:
    rag.process_video(video, fps=1.0)
    print(f"Processed: {video}")
```

### Export Search Results

```python
import json

results = rag.search_with_text("query", top_k=10)
export_data = [{
    'timestamp': r.timestamp,
    'caption': r.caption,
    'score': r.relevance_score
} for r in results]

with open('results.json', 'w') as f:
    json.dump(export_data, f, indent=2)
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- GPU-accelerated FAISS (IVF indices)
- Video streaming support
- Multi-video search
- Custom reranking strategies
- UI improvements

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- [Qwen Team](https://github.com/QwenLM) for the amazing VL models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [VideoRAG Paper](https://arxiv.org/abs/2410.10713) for the methodology
- [Gradio](https://gradio.app) for the UI framework

## 📚 References

- [Qwen3-Embedding Repository](https://github.com/QwenLM/Qwen3-Embedding)
- [VideoRAG Paper](https://arxiv.org/abs/2410.10713)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Ollama Documentation](https://ollama.com/docs)

---

**Built with ❤️ using Qwen3-VL, FAISS, and Gradio**
