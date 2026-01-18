![Beispielbild](pp_my_qrcode_1768774121925.jpg)


# 🎥 SotaVideoRAG: State-of-the-Art Video Retrieval System

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

Advanced video understanding and retrieval system using Qwen3-VL models, FAISS indexing, and multi-modal search.

## ✨ Features

- 🎬 **Hierarchical Video Processing**: Scene detection, keyframe extraction, multi-modal encoding
- 💾 **FAISS Persistent Indexing**: Process once, query forever with lightning-fast vector search
- 🔍 **Text Search**: Natural language queries to find specific moments
- 🖼️ **Image Search**: Upload images to find visually similar frames
- 🎯 **Multi-Modal Reranking**: Advanced relevance scoring with vision-language models
- 🤖 **AI-Powered Answers**: Context-aware response generation using Qwen3-VL
- 📦 **Docker Support**: Easy deployment with Docker Compose
- 🔧 **Configurable**: Environment variables for easy customization

## 🏗️ Architecture

```
Video Input → Scene Detection → Keyframe Extraction
     ↓
Multi-Modal Encoding (Qwen3-VL)
     ↓
FAISS Indexing & Storage (Persistent)
     ↓
Query Processing (Text/Image)
     ↓
FAISS Similarity Search (<10ms)
     ↓
Multi-Modal Reranking (Optional)
     ↓
Answer Generation (Qwen3-VL via Ollama)
```

## 📦 Installation

### Quick Start (Recommended)

```bash
# Clone repository
git clone <your-repo-url>
cd SotaVideoRAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install FAISS (GPU version recommended)
pip install faiss-gpu
# Or CPU version: pip install faiss-cpu

# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-vl
ollama serve  # Keep running in separate terminal

# Run application
python videorag_app.py
```

Visit `http://localhost:7860` in your browser.

### Docker Deployment

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f videorag

# Stop services
docker-compose down
```

## 🚀 Usage

### 1. Initialize Models

Go to the **Settings** tab and click "Initialize Models". This loads:
- Qwen3-VL-Embedding-2B (or 8B for better quality)
- Qwen3-VL-Reranker-2B (or 8B for better quality)

### 2. Process Video

**Process Video** tab:
- Upload your video file
- Set FPS (1.0 recommended for balance)
- Enable "Use Cache" to load cached indexes
- Click "Process Video"

### 3. Search

**Text Search** tab:
- Enter natural language query
- Select number of results (Top K)
- Enable/disable reranking
- Get AI-generated answer + relevant frames

**Image Search** tab:
- Upload a reference image
- Find visually similar frames in the video

## 💡 Examples

### Text Search
```
Query: "Show me scenes with people talking"
Results: Relevant frames + AI answer explaining the conversation context
```

### Image Search
```
Upload: Image of a dog
Results: All frames in the video containing similar dogs
```

### Programmatic Usage

```python
from videorag_app import VideoRAG
from PIL import Image

# Initialize
rag = VideoRAG()

# Process video (once)
rag.process_video("my_video.mp4", fps=1.0)

# Text search
results = rag.search_with_text("Find outdoor scenes", top_k=5)

# Image search
query_image = Image.open("reference.jpg")
results = rag.search_with_image(query_image, top_k=5)

# Rerank and generate answer
reranked = rag.rerank_results("outdoor scenes", results)
answer = rag.generate_answer("What's happening?", reranked)
```

## 🎯 Models

| Model | Purpose | Size | Link |
|-------|---------|------|------|
| **Qwen3-VL** | Caption & Answer generation | Via Ollama | [Ollama](https://ollama.com) |
| **Qwen3-VL-Embedding-2B** | Multi-modal embeddings | ~4GB | [HF](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) |
| **Qwen3-VL-Embedding-8B** | Higher quality embeddings | ~16GB | [HF](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) |
| **Qwen3-VL-Reranker-2B** | Multi-modal reranking | ~4GB | [HF](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) |
| **Qwen3-VL-Reranker-8B** | Higher quality reranking | ~16GB | [HF](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B) |

## ⚙️ Configuration

Create a `.env` file for custom configuration:

```bash
# Models
EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-8B
RERANKER_MODEL=Qwen/Qwen3-VL-Reranker-8B

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-vl

# Storage
INDEX_DIR=./video_indexes
CACHE_SIZE_MB=2000

# Processing
DEFAULT_FPS=1.0
DEFAULT_TOP_K=5
```

## 💾 Storage & Caching

### Index Files
Processed videos are cached in `./video_indexes/`:
```
./video_indexes/
├── abc123def456.faiss    # FAISS vector index
├── abc123def456.json     # Metadata (segments, captions)
└── xyz789ghi012.faiss    # Another video's index
```

### Storage Requirements
- **Per video**: ~8-15 MB per 1000 frames
- **1-minute video @ 1 FPS**: ~500 KB - 1 MB
- **10-minute video @ 1 FPS**: ~5-10 MB

### Cache Management
In the Settings tab:
- **View Cache Info**: See all cached videos and sizes
- **Clear All Cache**: Delete all indexes

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

## 🔧 Troubleshooting

### Models Not Loading
```bash
pip install transformers>=4.57.0 --upgrade
pip install qwen-vl-utils>=0.0.14
```

### Ollama Connection Error
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Verify model
ollama list | grep qwen3-vl
```

### Out of Memory
1. Use smaller models (2B instead of 8B)
2. Reduce FPS (try 0.5)
3. Process shorter videos
4. Close other applications

### FAISS Installation Issues
```bash
# CPU version (easier)
pip install faiss-cpu

# GPU version (better performance)
conda install -c conda-forge faiss-gpu
```

## 🐳 Docker Usage

### Build Image
```bash
docker build -t videorag:latest .
```

### Run with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

### Run Standalone Container
```bash
docker run -d \
  -p 7860:7860 \
  -v $(pwd)/video_indexes:/app/video_indexes \
  --gpus all \
  --name videorag \
  videorag:latest
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] GPU-accelerated FAISS (IVF indices)
- [ ] Video streaming support
- [ ] Multi-video search
- [ ] Custom reranking strategies
- [ ] UI improvements
- [ ] Batch processing API
- [ ] Support for more video formats

## 📝 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

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

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: your.email@example.com

---

**Built with ❤️ using Qwen3-VL, FAISS, and Gradio**

⭐ Star this repo if you find it useful!
