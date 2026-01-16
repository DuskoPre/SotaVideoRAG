# 📘 Complete Setup Guide for SotaVideoRAG

This guide will walk you through setting up SotaVideoRAG from scratch.

## 📋 Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10/11
- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU with 8GB+ VRAM (recommended)
  - Works on CPU but significantly slower
- **RAM**: 16GB+ recommended
- **Disk Space**: 20GB+ free space

### Software Requirements

- Git
- Python 3.8+
- pip
- (Optional) Docker and Docker Compose
- (Optional) Conda

## 🚀 Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd SotaVideoRAG
```

#### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n videorag python=3.10
conda activate videorag
```

#### Step 3: Create Project Structure

```bash
# Create necessary directories
mkdir -p scripts video_indexes logs

# Create __init__.py for scripts
touch scripts/__init__.py
```

#### Step 4: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install FAISS
# For GPU (recommended):
pip install faiss-gpu

# For CPU only:
pip install faiss-cpu
```

#### Step 5: Install Ollama

```bash
# On Linux/Mac:
curl -fsSL https://ollama.com/install.sh | sh

# On Windows: Download installer from https://ollama.com/download

# Pull Qwen3-VL model
ollama pull qwen3-vl

# Verify installation
ollama list
```

#### Step 6: Download Model Scripts

Ensure these files are in the `scripts/` directory:
- `qwen3_vl_embedding.py`
- `qwen3_vl_reranker.py`
- `__init__.py`

#### Step 7: Create Configuration

```bash
# Create config file
cp config.py.example config.py

# Or create from scratch
cat > config.py << 'EOF'
import os
from pathlib import Path

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-VL-Reranker-2B")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl")
INDEX_DIR = Path(os.getenv("INDEX_DIR", "./video_indexes"))
CACHE_SIZE_MB = int(os.getenv("CACHE_SIZE_MB", "1000"))
DEFAULT_FPS = float(os.getenv("DEFAULT_FPS", "1.0"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
EOF
```

#### Step 8: Start Ollama Server

```bash
# In a separate terminal, start Ollama
ollama serve
```

#### Step 9: Run Application

```bash
# Start VideoRAG
python videorag_app.py
```

Visit `http://localhost:7860` in your browser!

### Method 2: Docker Installation

#### Step 1: Install Docker

```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
```

#### Step 2: Clone and Build

```bash
git clone <your-repo-url>
cd SotaVideoRAG

# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f videorag
```

Access at `http://localhost:7860`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```bash
# Models
EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
RERANKER_MODEL=Qwen/Qwen3-VL-Reranker-2B

# For better quality (requires more VRAM):
# EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-8B
# RERANKER_MODEL=Qwen/Qwen3-VL-Reranker-8B

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3-vl

# Storage
INDEX_DIR=./video_indexes
CACHE_SIZE_MB=1000

# Processing defaults
DEFAULT_FPS=1.0
DEFAULT_TOP_K=5
```

### Model Selection Guide

| Use Case | Embedding Model | Reranker Model | VRAM Required |
|----------|----------------|----------------|---------------|
| Development/Testing | Qwen3-VL-Embedding-2B | Qwen3-VL-Reranker-2B | 8GB |
| Production/Best Quality | Qwen3-VL-Embedding-8B | Qwen3-VL-Reranker-8B | 24GB+ |
| CPU Only | Qwen3-VL-Embedding-2B | Qwen3-VL-Reranker-2B | 16GB RAM |

## 🧪 Verification

### Test Installation

```bash
# Test Python imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import faiss; print(f'FAISS: OK')"
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"

# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test Ollama
curl http://localhost:11434/api/tags
```

### Run Quick Test

```python
# test_setup.py
from videorag_app import VideoRAG
from PIL import Image
import numpy as np

print("Initializing VideoRAG...")
rag = VideoRAG()

print("Testing text embedding...")
emb = rag.get_text_embedding("test query")
print(f"✓ Text embedding shape: {emb.shape}")

print("Testing image embedding...")
test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
emb = rag.get_image_embedding(test_img)
print(f"✓ Image embedding shape: {emb.shape}")

print("\n✓ All tests passed!")
```

Run with: `python test_setup.py`

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```bash
# Use smaller models
export EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
export RERANKER_MODEL=Qwen/Qwen3-VL-Reranker-2B

# Or process videos at lower FPS
# In UI: Set FPS slider to 0.5
```

### Issue: Ollama Connection Refused

**Solution:**
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama
ollama serve

# Check port
lsof -i :11434
```

### Issue: Import Errors

**Solution:**
```bash
# Ensure scripts/__init__.py exists
touch scripts/__init__.py

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Issue: FAISS Not Found

**Solution:**
```bash
# Uninstall both versions
pip uninstall faiss-cpu faiss-gpu -y

# Install correct version
# For GPU:
pip install faiss-gpu

# For CPU:
pip install faiss-cpu
```

### Issue: Gradio Interface Not Loading

**Solution:**
```bash
# Check firewall
sudo ufw allow 7860

# Try different port
python videorag_app.py --server-port 8080

# Check if port is in use
lsof -i :7860
```

## 📊 Performance Tuning

### For Low VRAM (8GB)

```python
# config.py
EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
DEFAULT_FPS = 0.5  # Process fewer frames
```

### For High VRAM (24GB+)

```python
# config.py
EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-8B"
DEFAULT_FPS = 2.0  # Process more frames
```

### For CPU Only

```bash
# Reduce batch size and use lower FPS
export DEFAULT_FPS=0.5
export CACHE_SIZE_MB=500

# Use CPU FAISS
pip install faiss-cpu
```

## 🎯 Next Steps

1. **Process Your First Video**:
   - Go to "Process Video" tab
   - Upload a short video (1-2 minutes)
   - Set FPS to 1.0
   - Click "Process Video"

2. **Try Text Search**:
   - Go to "Text Search" tab
   - Enter: "What is shown in the video?"
   - Click "Search"

3. **Try Image Search**:
   - Go to "Image Search" tab
   - Upload a reference image
   - Click "Find Similar Frames"

4. **Explore Cache Management**:
   - Go to "Settings" tab
   - Click "View Cache Info"
   - See processed videos and sizes

## 📚 Additional Resources

- [README.md](README.md) - Full documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [Qwen3 Documentation](https://github.com/QwenLM/Qwen3-Embedding)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Gradio Documentation](https://gradio.app/docs)

## 💬 Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share ideas
- **Discord**: Join our community (coming soon)

---

Happy video searching! 🎬🔍
