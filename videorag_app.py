import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import json
import tempfile
import os
import pickle
import hashlib
from typing import List, Dict, Tuple, Optional, Union
import requests
from PIL import Image
import base64
from dataclasses import dataclass, asdict
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import faiss

# Import Qwen3 models (text-only versions)
import sys
sys.path.append('./scripts')

# Try to import vision models first, fall back to text-only
try:
    from qwen3_vl_embedding import Qwen3VLEmbedder
    from qwen3_vl_reranker import Qwen3VLReranker
    USE_VISION_MODELS = True
except ImportError:
    # Fall back to text-only models
    from qwen3_embedding import Qwen3Embedding
    from qwen3_reranker import Qwen3Reranker
    USE_VISION_MODELS = False
    print("⚠️ Using text-only models. For better results, use Qwen3-VL-Embedding and Qwen3-VL-Reranker")

@dataclass
class VideoSegment:
    """Represents a video segment with hierarchical structure"""
    segment_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    frames: List[np.ndarray]
    frame_embeddings: List[np.ndarray]
    segment_embedding: np.ndarray
    captions: List[str]
    segment_summary: str
    audio_transcript: str
    scene_type: str
    keyframes: List[int]

@dataclass
class RetrievalResult:
    """Stores retrieval results with scores"""
    segment: VideoSegment
    frame_idx: int
    relevance_score: float
    frame: np.ndarray
    caption: str
    timestamp: float

class VideoRAG:
    def __init__(self, 
                 embedding_model_path: str = "Qwen/Qwen3-VL-Embedding-2B",
                 reranker_model_path: str = "Qwen/Qwen3-VL-Reranker-2B",
                 use_vision_models: bool = None,
                 index_dir: str = "./video_indexes"):
        self.segments: List[VideoSegment] = []
        self.video_embedding: Optional[np.ndarray] = None
        self.segment_embeddings: List[np.ndarray] = []
        self.frame_embeddings: List[np.ndarray] = []
        self.video_metadata: Dict = {}
        self.ollama_url = "http://localhost:11434"
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # FAISS index
        self.faiss_index = None
        self.frame_to_segment_map = []  # Maps FAISS index to (segment_id, frame_idx)
        self.current_video_hash = None
        
        # Auto-detect or use specified model type
        if use_vision_models is None:
            use_vision_models = USE_VISION_MODELS
        
        self.use_vision_models = use_vision_models
        
        # Initialize embedding model
        print(f"Loading {'vision-language' if use_vision_models else 'text-only'} embedding model...")
        if use_vision_models:
            self.embedder = Qwen3VLEmbedder(
                model_name_or_path=embedding_model_path,
                torch_dtype=torch.bfloat16
            )
        else:
            # Text-only embedding model
            self.embedder = Qwen3Embedding(
                model_name_or_path=embedding_model_path,
                use_fp16=True
            )
        print("✓ Embedding model loaded")
        
        # Initialize reranker model
        print(f"Loading {'vision-language' if use_vision_models else 'text-only'} reranker model...")
        if use_vision_models:
            self.reranker = Qwen3VLReranker(
                model_name_or_path=reranker_model_path,
                torch_dtype=torch.bfloat16
            )
        else:
            # Text-only reranker
            self.reranker = Qwen3Reranker(
                model_name_or_path=reranker_model_path,
                max_length=4096
            )
        print("✓ Reranker model loaded")
        
    def compute_video_hash(self, video_path: str) -> str:
        """Compute hash of video file for caching"""
        hasher = hashlib.md5()
        with open(video_path, 'rb') as f:
            # Read first 10MB for speed
            chunk = f.read(10 * 1024 * 1024)
            hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_index_path(self, video_hash: str) -> Path:
        """Get path for storing/loading index"""
        return self.index_dir / f"{video_hash}.index"
    
    def save_index(self, video_hash: str):
        """Save video index to disk"""
        index_path = self.get_index_path(video_hash)
        
        # Prepare data for serialization
        segments_data = []
        for seg in self.segments:
            seg_dict = asdict(seg)
            # Convert numpy arrays to lists for JSON serialization
            seg_dict['frames'] = [frame.tobytes() for frame in seg.frames]
            seg_dict['frame_embeddings'] = [emb.tolist() for emb in seg.frame_embeddings]
            seg_dict['segment_embedding'] = seg.segment_embedding.tolist()
            segments_data.append(seg_dict)
        
        data = {
            'segments': segments_data,
            'video_metadata': self.video_metadata,
            'frame_to_segment_map': self.frame_to_segment_map,
            'video_embedding': self.video_embedding.tolist() if self.video_embedding is not None else None
        }
        
        # Save metadata as JSON
        with open(index_path.with_suffix('.json'), 'w') as f:
            json.dump(data, f)
        
        # Save FAISS index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(index_path.with_suffix('.faiss')))
        
        print(f"✓ Index saved to {index_path}")
    
    def load_index(self, video_hash: str) -> bool:
        """Load video index from disk"""
        index_path = self.get_index_path(video_hash)
        json_path = index_path.with_suffix('.json')
        faiss_path = index_path.with_suffix('.faiss')
        
        if not json_path.exists() or not faiss_path.exists():
            return False
        
        try:
            # Load metadata
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct segments
            self.segments = []
            for seg_dict in data['segments']:
                # Reconstruct numpy arrays
                frames = []
                for frame_bytes in seg_dict['frames']:
                    # Need to know frame shape - store it in metadata
                    pass  # Will reconstruct from stored shape
                
                seg_dict['frame_embeddings'] = [
                    np.array(emb) for emb in seg_dict['frame_embeddings']
                ]
                seg_dict['segment_embedding'] = np.array(seg_dict['segment_embedding'])
                
                # For now, store frames as None (can be regenerated if needed)
                seg_dict['frames'] = []
                
                # Create segment (simplified)
                segment = VideoSegment(**seg_dict)
                self.segments.append(segment)
            
            self.video_metadata = data['video_metadata']
            self.frame_to_segment_map = data['frame_to_segment_map']
            self.video_embedding = np.array(data['video_embedding']) if data['video_embedding'] else None
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(str(faiss_path))
            
            print(f"✓ Index loaded from {index_path}")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def build_faiss_index(self):
        """Build FAISS index from frame embeddings"""
        if not self.segments:
            return
        
        # Collect all frame embeddings
        all_embeddings = []
        self.frame_to_segment_map = []
        
        for segment in self.segments:
            for frame_idx, frame_emb in enumerate(segment.frame_embeddings):
                all_embeddings.append(frame_emb)
                self.frame_to_segment_map.append({
                    'segment_id': segment.segment_id,
                    'frame_idx': frame_idx,
                    'timestamp': segment.start_time + (frame_idx / len(segment.frames)) * (segment.end_time - segment.start_time)
                })
        
        # Convert to numpy array
        embeddings_matrix = np.array(all_embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Build FAISS index (use IndexFlatIP for exact cosine similarity)
        dimension = embeddings_matrix.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings_matrix)
        
        print(f"✓ Built FAISS index with {len(all_embeddings)} frame embeddings")
    
    def extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from video"""
        try:
            import moviepy.editor as mp
            video = mp.VideoFileClip(video_path)
            if video.audio is None:
                return None
            
            audio_path = tempfile.mktemp(suffix=".wav")
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            return audio_path
        except Exception as e:
            print(f"Audio extraction error: {e}")
            return None
    
    def transcribe_audio(self, audio_path: str, start_time: float, end_time: float) -> str:
        """Transcribe audio segment"""
        try:
            import librosa
            import soundfile as sf
            import speech_recognition as sr
            
            y, sr_rate = librosa.load(audio_path, sr=16000, 
                                     offset=start_time, 
                                     duration=end_time-start_time)
            
            temp_audio = tempfile.mktemp(suffix=".wav")
            sf.write(temp_audio, y, sr_rate)
            
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio) as source:
                audio_data = recognizer.record(source)
                try:
                    transcript = recognizer.recognize_google(audio_data)
                    os.remove(temp_audio)
                    return transcript
                except:
                    os.remove(temp_audio)
                    return ""
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def detect_scene_changes(self, frames: List[np.ndarray], threshold: float = 30.0) -> List[int]:
        """Detect scene changes using histogram differences"""
        scene_boundaries = [0]
        
        for i in range(1, len(frames)):
            hist1 = cv2.calcHist([cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2HSV)], 
                                [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)], 
                                [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            
            distance = np.sqrt(np.sum((hist1 - hist2) ** 2))
            
            if distance > threshold:
                scene_boundaries.append(i)
        
        scene_boundaries.append(len(frames))
        return scene_boundaries
    
    def extract_keyframes(self, frames: List[np.ndarray], num_keyframes: int = 3) -> List[int]:
        """Extract keyframes using uniform sampling"""
        if len(frames) <= num_keyframes:
            return list(range(len(frames)))
        
        indices = np.linspace(0, len(frames)-1, num_keyframes, dtype=int)
        return indices.tolist()
    
    def classify_scene_type(self, frames: List[np.ndarray], captions: List[str]) -> str:
        """Classify scene type based on caption keywords"""
        all_text = " ".join(captions).lower()
        
        if any(word in all_text for word in ["indoor", "room", "building", "office"]):
            return "indoor"
        elif any(word in all_text for word in ["outdoor", "street", "park", "sky"]):
            return "outdoor"
        elif any(word in all_text for word in ["person", "people", "man", "woman"]):
            return "person-focused"
        elif any(word in all_text for word in ["action", "moving", "running", "driving"]):
            return "action"
        else:
            return "general"
    
    def get_frame_embedding(self, frame: np.ndarray, caption: str = None) -> np.ndarray:
        """Get multi-modal embedding for frame"""
        try:
            if self.use_vision_models:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                input_data = {"image": pil_image}
                if caption:
                    input_data["text"] = caption
                
                embeddings = self.embedder.process([input_data])
                return embeddings[0].cpu().numpy()
            else:
                # Text-only: use caption
                if caption:
                    embeddings = self.embedder.encode([caption], is_query=False)
                    return embeddings[0].cpu().numpy()
                return np.zeros(1024)
            
        except Exception as e:
            print(f"Error getting frame embedding: {e}")
            return np.zeros(2048 if self.use_vision_models else 1024)
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding"""
        try:
            if self.use_vision_models:
                embeddings = self.embedder.process([{"text": text}])
                return embeddings[0].cpu().numpy()
            else:
                embeddings = self.embedder.encode([text], is_query=True)
                return embeddings[0].cpu().numpy()
        except Exception as e:
            print(f"Error getting text embedding: {e}")
            return np.zeros(2048 if self.use_vision_models else 1024)
    
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Get embedding for uploaded image"""
        try:
            if self.use_vision_models:
                embeddings = self.embedder.process([{"image": image}])
                return embeddings[0].cpu().numpy()
            else:
                # Text-only models can't process images
                print("⚠️ Text-only models cannot process images directly")
                return np.zeros(1024)
        except Exception as e:
            print(f"Error getting image embedding: {e}")
            return np.zeros(2048 if self.use_vision_models else 1024)
    
    def generate_caption(self, frame: np.ndarray) -> str:
        """Generate caption for frame using Qwen3-VL via Ollama"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen3-vl",
                    "prompt": "Describe this image in detail, focusing on key objects, actions, people, and context.",
                    "images": [img_base64],
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return "Caption generation failed"
                
        except Exception as e:
            print(f"Caption generation error: {e}")
            return "Error generating caption"
    
    def generate_segment_summary(self, captions: List[str], audio_transcript: str) -> str:
        """Generate summary for video segment"""
        try:
            context = f"Frame descriptions:\n" + "\n".join([f"- {cap}" for cap in captions])
            if audio_transcript:
                context += f"\n\nAudio transcript: {audio_transcript}"
            
            prompt = f"""Summarize this video segment in 2-3 sentences based on the following information:

{context}

Summary:"""
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen3-vl",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return "Summary generation failed"
                
        except Exception as e:
            return f"Error: {e}"
    
    def compute_segment_embedding(self, frame_embeddings: List[np.ndarray], 
                                  frames: List[np.ndarray],
                                  captions: List[str], 
                                  audio_transcript: str) -> np.ndarray:
        """Compute segment-level embedding"""
        try:
            visual_embedding = np.mean(frame_embeddings, axis=0)
            
            text_content = " ".join(captions)
            if audio_transcript:
                text_content += " " + audio_transcript
            text_embedding = self.get_text_embedding(text_content)
            
            segment_embedding = 0.6 * visual_embedding + 0.4 * text_embedding
            return segment_embedding
        except Exception as e:
            print(f"Error computing segment embedding: {e}")
            return np.zeros(2048 if self.use_vision_models else 1024)
    
    def process_video(self, video_path: str, fps: float = 1.0, 
                     segment_duration: float = 10.0, use_cache: bool = True) -> str:
        """Process video with hierarchical structure and FAISS indexing"""
        
        # Check cache
        video_hash = self.compute_video_hash(video_path)
        self.current_video_hash = video_hash
        
        if use_cache and self.load_index(video_hash):
            return f"✓ Loaded cached index for this video ({len(self.segments)} segments)"
        
        self.segments = []
        self.video_metadata = {}
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        all_frames = []
        frame_interval = int(video_fps / fps)
        frame_count = 0
        
        print(f"Extracting frames at {fps} FPS...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                all_frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if len(all_frames) == 0:
            return "No frames extracted from video"
        
        print(f"Extracted {len(all_frames)} frames")
        
        # Extract audio
        audio_path = self.extract_audio(video_path)
        
        # Detect scene changes
        print("Detecting scene changes...")
        scene_boundaries = self.detect_scene_changes(all_frames)
        print(f"Detected {len(scene_boundaries)-1} scenes")
        
        # Process each segment
        segment_id = 0
        for i in range(len(scene_boundaries) - 1):
            print(f"\nProcessing segment {segment_id + 1}/{len(scene_boundaries)-1}...")
            
            start_idx = scene_boundaries[i]
            end_idx = scene_boundaries[i + 1]
            segment_frames = all_frames[start_idx:end_idx]
            
            if len(segment_frames) == 0:
                continue
            
            start_time = start_idx / fps
            end_time = end_idx / fps
            
            keyframe_indices = self.extract_keyframes(segment_frames)
            keyframes = [segment_frames[idx] for idx in keyframe_indices]
            
            print(f"  Generating captions...")
            captions = [self.generate_caption(kf) for kf in keyframes]
            
            print(f"  Computing embeddings...")
            frame_embeddings = [
                self.get_frame_embedding(kf, cap) 
                for kf, cap in zip(keyframes, captions)
            ]
            
            audio_transcript = ""
            if audio_path:
                print(f"  Transcribing audio...")
                audio_transcript = self.transcribe_audio(audio_path, start_time, end_time)
            
            print(f"  Generating summary...")
            segment_summary = self.generate_segment_summary(captions, audio_transcript)
            
            segment_embedding = self.compute_segment_embedding(
                frame_embeddings, keyframes, captions, audio_transcript
            )
            
            scene_type = self.classify_scene_type(keyframes, captions)
            
            segment = VideoSegment(
                segment_id=segment_id,
                start_frame=start_idx,
                end_frame=end_idx,
                start_time=start_time,
                end_time=end_time,
                frames=keyframes,
                frame_embeddings=frame_embeddings,
                segment_embedding=segment_embedding,
                captions=captions,
                segment_summary=segment_summary,
                audio_transcript=audio_transcript,
                scene_type=scene_type,
                keyframes=keyframe_indices
            )
            
            self.segments.append(segment)
            segment_id += 1
        
        # Compute video-level embedding
        if self.segments:
            segment_embeddings = [seg.segment_embedding for seg in self.segments]
            self.video_embedding = np.mean(segment_embeddings, axis=0)
        
        # Build FAISS index
        print("\nBuilding FAISS index...")
        self.build_faiss_index()
        
        # Save index
        print("Saving index to disk...")
        self.save_index(video_hash)
        
        # Clean up
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        
        return f"✓ Processed video into {len(self.segments)} segments with {self.faiss_index.ntotal} indexed frames"
    
    def search_with_faiss(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Search using FAISS index"""
        if self.faiss_index is None:
            return []
        
        # Normalize query
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.frame_to_segment_map):
                frame_info = self.frame_to_segment_map[idx]
                segment = self.segments[frame_info['segment_id']]
                frame_idx = frame_info['frame_idx']
                
                results.append({
                    'segment': segment,
                    'frame_idx': frame_idx,
                    'frame': segment.frames[frame_idx],
                    'caption': segment.captions[frame_idx],
                    'timestamp': frame_info['timestamp'],
                    'similarity': float(dist)
                })
        
        return results
    
    def search_with_image(self, image: Image.Image, top_k: int = 5) -> List[RetrievalResult]:
        """Search using uploaded image"""
        if self.faiss_index is None:
            return []
        
        print("Getting image embedding...")
        image_embedding = self.get_image_embedding(image)
        
        print("Searching with FAISS...")
        faiss_results = self.search_with_faiss(image_embedding, top_k * 2)
        
        # Convert to RetrievalResult
        results = []
        for res in faiss_results[:top_k]:
            result = RetrievalResult(
                segment=res['segment'],
                frame_idx=res['frame_idx'],
                relevance_score=res['similarity'],
                frame=res['frame'],
                caption=res['caption'],
                timestamp=res['timestamp']
            )
            results.append(result)
        
        return results
    
    def search_with_text(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search using text query"""
        if self.faiss_index is None:
            return []
        
        print("Getting query embedding...")
        query_embedding = self.get_text_embedding(query)
        
        print("Searching with FAISS...")
        faiss_results = self.search_with_faiss(query_embedding, top_k * 2)
        
        # Convert to RetrievalResult
        results = []
        for res in faiss_results[:top_k]:
            result = RetrievalResult(
                segment=res['segment'],
                frame_idx=res['frame_idx'],
                relevance_score=res['similarity'],
                frame=res['frame'],
                caption=res['caption'],
                timestamp=res['timestamp']
            )
            results.append(result)
        
        return results
    
    def rerank_results(self, query: str, results: List[RetrievalResult], query_image: Image.Image = None) -> List[RetrievalResult]:
        """Rerank results using Qwen3-VL-Reranker"""
        try:
            print(f"Reranking with {'vision-language' if self.use_vision_models else 'text-only'} reranker...")
            
            if self.use_vision_models:
                documents = []
                for result in results:
                    frame_rgb = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    doc_parts = [
                        f"Frame: {result.caption}",
                        f"Context: {result.segment.segment_summary}",
                    ]
                    doc_text = " ".join(doc_parts)
                    
                    documents.append({
                        "image": pil_image,
                        "text": doc_text
                    })
                
                # Build query
                query_dict = {"text": query}
                if query_image:
                    query_dict["image"] = query_image
                
                rerank_input = {
                    "instruction": "Retrieve the most relevant video frame",
                    "query": query_dict,
                    "documents": documents,
                    "fps": 1.0
                }
                
                scores = self.reranker.process(rerank_input)
                
                for i, score in enumerate(scores):
                    if i < len(results):
                        original_score = results[i].relevance_score
                        results[i].relevance_score = 0.3 * original_score + 0.7 * score
            else:
                # Text-only reranking
                pairs = []
                for result in results:
                    doc_text = f"{result.caption}. {result.segment.segment_summary}"
                    pairs.append((query, doc_text))
                
                scores = self.reranker.compute_scores(pairs)
                
                for i, score in enumerate(scores):
                    if i < len(results):
                        results[i].relevance_score = score
            
            return sorted(results, key=lambda x: x.relevance_score, reverse=True)
            
        except Exception as e:
            print(f"Reranking error: {e}")
            return results
    
    def generate_answer(self, query: str, results: List[RetrievalResult]) -> str:
        """Generate answer using retrieved context"""
        try:
            context_parts = []
            for result in results:
                context_parts.append(
                    f"[Time {result.timestamp:.1f}s]\n"
                    f"Scene: {result.segment.scene_type}\n"
                    f"Description: {result.caption}\n"
                )
            
            context = "\n".join(context_parts)
            
            prompt = f"""Based on these video frames, answer the question:

{context}

Question: {query}

Answer:"""
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen3-vl",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=90
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return "Error generating answer"
                
        except Exception as e:
            return f"Error: {e}"

# Initialize VideoRAG
video_rag = None

def initialize_models(embedding_model, reranker_model):
    """Initialize VideoRAG with specified models"""
    global video_rag
    try:
        video_rag = VideoRAG(
            embedding_model_path=embedding_model,
            reranker_model_path=reranker_model
        )
        return "✓ Models loaded successfully! FAISS indexing enabled."
