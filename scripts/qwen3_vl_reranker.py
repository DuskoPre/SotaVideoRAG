"""
Qwen3-VL-Reranker: Multi-modal reranking model for images and text
Based on: https://github.com/QwenLM/Qwen3-Embedding
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from PIL import Image
from typing import Optional, List, Union, Dict, Any
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

# Constants
MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS


def sample_frames(
    frames: List[Union[str, Image.Image]], 
    num_segments: int, 
    max_segments: int
) -> List[Union[str, Image.Image]]:
    """Sample frames uniformly from a video"""
    duration = len(frames)
    frame_id_array = np.linspace(0, duration - 1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    last_frame_id = frame_id_list[-1]

    sampled_frames = []
    for frame_idx in frame_id_list:
        try:
            sampled_frames.append(frames[frame_idx])
        except:
            break
    
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[last_frame_id])
    
    return sampled_frames[:max_segments]


class Qwen3VLReranker:
    """
    Qwen3-VL Reranker for multi-modal relevance scoring.
    
    Scores query-document pairs where documents can contain:
    - Text only
    - Image only
    - Text + Image (multi-modal)
    - Video
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        num_frames: int = MAX_FRAMES,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Retrieve relevant documents for the query.",
        **kwargs
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.default_instruction = default_instruction

        # Load model
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **kwargs
        ).to(device)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            padding_side='left',
            trust_remote_code=True
        )
        
        self.model.eval()
        self.device = device
        
        # Get token IDs for yes/no
        self.token_yes_id = self.processor.tokenizer.convert_tokens_to_ids("yes")
        self.token_no_id = self.processor.tokenizer.convert_tokens_to_ids("no")

    def format_rerank_input(
        self,
        instruction: str,
        query: Dict[str, Any],
        document: Dict[str, Any],
        fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict]:
        """
        Format query and document into conversation format.
        
        Args:
            instruction: Task instruction
            query: Query dict with 'text' and/or 'image'
            document: Document dict with 'text' and/or 'image'
            fps: Frames per second for video
            max_frames: Maximum frames to extract
            
        Returns:
            Formatted conversation
        """
        content = []
        
        # Add instruction
        content.append({
            "type": "text",
            "text": f"<Instruction>: {instruction}\n\n"
        })
        
        # Add query
        query_parts = ["<Query>:"]
        
        # Query text
        if query.get('text'):
            query_parts.append(query['text'])
        
        # Query image
        if query.get('image'):
            image = query['image']
            if isinstance(image, Image.Image):
                image_content = image
            elif isinstance(image, str):
                image_content = image if image.startswith(('http', 'oss')) else 'file://' + image
            else:
                raise TypeError(f"Unrecognized image type: {type(image)}")
            
            content.append({
                "type": "image",
                "image": image_content,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels
            })
        
        content.append({
            "type": "text",
            "text": " ".join(query_parts) + "\n\n"
        })
        
        # Add document
        doc_parts = ["<Document>:"]
        
        # Document text
        if document.get('text'):
            doc_parts.append(document['text'])
        
        # Document image
        if document.get('image'):
            image = document['image']
            if isinstance(image, Image.Image):
                image_content = image
            elif isinstance(image, str):
                image_content = image if image.startswith(('http', 'oss')) else 'file://' + image
            else:
                raise TypeError(f"Unrecognized image type: {type(image)}")
            
            content.append({
                "type": "image",
                "image": image_content,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels
            })
        
        # Document video
        if document.get('video'):
            video = document['video']
            video_kwargs = {'total_pixels': self.total_pixels}
            
            if isinstance(video, list):
                # List of frames
                video_content = video
                if self.num_frames is not None or self.max_frames is not None:
                    video_content = sample_frames(
                        video_content,
                        self.num_frames,
                        self.max_frames
                    )
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele)
                    for ele in video_content
                ]
            elif isinstance(video, str):
                # Video path or URL
                video_content = video if video.startswith(('http://', 'https://')) else 'file://' + video
                video_kwargs = {
                    'fps': fps or self.fps,
                    'max_frames': max_frames or self.max_frames,
                }
            else:
                raise TypeError(f"Unrecognized video type: {type(video)}")
            
            content.append({
                'type': 'video',
                'video': video_content,
                **video_kwargs
            })
        
        content.append({
            "type": "text",
            "text": " ".join(doc_parts) + "\n\n<Question>: Is the document relevant to the query? Answer with 'yes' or 'no'.\n<Answer>:"
        })
        
        conversation = [{"role": "user", "content": content}]
        return conversation

    def _preprocess_inputs(
        self,
        conversations: List[List[Dict]]
    ) -> Dict[str, torch.Tensor]:
        """Preprocess conversations for model input"""
        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=False
        )

        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            video_inputs = None
            video_kwargs = {'do_sample_frames': False}

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors='pt',
            **video_kwargs
        )

        return inputs

    @torch.no_grad()
    def compute_scores(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> List[float]:
        """
        Compute relevance scores from model outputs.
        
        Args:
            inputs: Preprocessed model inputs
            
        Returns:
            List of relevance scores (0-1)
        """
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # Get logits for the last token
        logits = outputs.logits[:, -1, :]
        
        # Extract yes/no probabilities
        yes_logits = logits[:, self.token_yes_id]
        no_logits = logits[:, self.token_no_id]
        
        # Stack and compute softmax
        binary_logits = torch.stack([no_logits, yes_logits], dim=1)
        probs = F.softmax(binary_logits, dim=1)
        
        # Return probability of "yes"
        scores = probs[:, 1].cpu().tolist()
        return scores

    def process(
        self,
        inputs: Dict[str, Any]
    ) -> List[float]:
        """
        Process reranking input and return relevance scores.
        
        Args:
            inputs: Dict with keys:
                - instruction: str (optional)
                - query: Dict with 'text' and/or 'image'
                - documents: List of dicts with 'text' and/or 'image'
                - fps: float (optional, for video)
                - max_frames: int (optional, for video)
                
        Returns:
            List of relevance scores (one per document)
        """
        instruction = inputs.get('instruction', self.default_instruction)
        query = inputs['query']
        documents = inputs['documents']
        fps = inputs.get('fps', self.fps)
        max_frames = inputs.get('max_frames', self.max_frames)
        
        # Format each query-document pair
        conversations = [
            self.format_rerank_input(
                instruction=instruction,
                query=query,
                document=doc,
                fps=fps,
                max_frames=max_frames
            )
            for doc in documents
        ]
        
        # Preprocess
        processed_inputs = self._preprocess_inputs(conversations)
        
        # Compute scores
        scores = self.compute_scores(processed_inputs)
        
        return scores

    def rerank(
        self,
        query: Dict[str, Any],
        documents: List[Dict[str, Any]],
        instruction: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Query dict with 'text' and/or 'image'
            documents: List of document dicts
            instruction: Optional task instruction
            top_k: Return only top k results
            
        Returns:
            List of dicts with 'document', 'score', 'index'
        """
        inputs = {
            'instruction': instruction or self.default_instruction,
            'query': query,
            'documents': documents
        }
        
        scores = self.process(inputs)
        
        # Create results
        results = [
            {
                'document': doc,
                'score': score,
                'index': i
            }
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results


if __name__ == "__main__":
    # Example usage
    print("Loading Qwen3-VL-Reranker model...")
    reranker = Qwen3VLReranker(
        model_name_or_path="Qwen/Qwen3-VL-Reranker-2B",
        torch_dtype=torch.bfloat16
    )
    
    # Test reranking
    query = {"text": "A woman playing with her dog on a beach at sunset."}
    
    documents = [
        {"text": "A woman shares a joyful moment with her golden retriever on a beach."},
        {"text": "City skyline at night with bright lights."},
        {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    ]
    
    inputs = {
        "instruction": "Retrieve relevant documents for the user's query",
        "query": query,
        "documents": documents,
        "fps": 1.0
    }
    
    print("Computing relevance scores...")
    scores = reranker.process(inputs)
    
    print("\nResults:")
    for i, (doc, score) in enumerate(zip(documents, scores)):
        doc_type = "image" if "image" in doc else "text"
        print(f"{i+1}. [{doc_type}] Score: {score:.4f}")
