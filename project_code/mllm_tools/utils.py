from typing import Union, List, Dict, Any, Optional
from PIL import Image
import tempfile
import os
from .openai import OpenAIWrapper


def _prepare_text_inputs(texts: Union[str, List[str]]) -> List[Dict[str, str]]:
    """
    Converts a list of text strings into the input format for the model.

    Args:
        texts (Union[str, List[str]]): The text string(s) to be processed.

    Returns:
        List[Dict[str, str]]: A list of dictionaries formatted for the model.
    """
    inputs = []
    # Add each text string to the inputs
    if isinstance(texts, str):
        texts = [texts]
    for text in texts:
        inputs.append({
            "type": "text",
            "content": text
        })
    return inputs

def _prepare_text_image_inputs(texts: Union[str, List[str]], images: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> List[Dict[str, str]]:
    """
    Converts text strings and images into the input format for the model.

    Args:
        texts (Union[str, List[str]]): Text string(s) to be processed.
        images (Union[str, Image.Image, List[Union[str, Image.Image]]]): Image file path(s) or PIL Image object(s).
    Returns:
        List[Dict[str, str]]: A list of dictionaries formatted for the model.
    """
    inputs = []
    # Add each text string to the inputs
    if isinstance(texts, str):
        texts = [texts]
    for text in texts:
        inputs.append({
            "type": "text",
            "content": text
        })
    # Add each image to the inputs
    if isinstance(images, (str, Image.Image)):
        images = [images]
    for image in images:
        inputs.append({
            "type": "image",
            "content": image
        })
    return inputs

def _prepare_text_video_inputs(texts: Union[str, List[str]], videos: Union[str, List[str]]) -> List[Dict[str, str]]:
    """
    Converts text strings and video file paths into the input format for the Agent model.

    Args:
        texts (Union[str, List[str]]): Text string(s) to be processed.
        videos (Union[str, List[str]]): Video file path(s).
    Returns:
        List[Dict[str, str]]: A list of dictionaries formatted for the Agent model.
    """
    inputs = []
    # Add each text string to the inputs
    if isinstance(texts, str):
        texts = [texts]
    for text in texts:
        inputs.append({
            "type": "text",
            "content": text
        })
    # Add each video file path to the inputs
    if isinstance(videos, str):
        videos = [videos]
    for video in videos:
        inputs.append({
            "type": "video",
            "content": video
        })
    return inputs

def _prepare_text_audio_inputs(texts: Union[str, List[str]], audios: Union[str, List[str]]) -> List[Dict[str, str]]:
    """
    Converts text strings and audio file paths into the input format for the Agent model.

    Args:
        texts (Union[str, List[str]]): Text string(s) to be processed.
        audios (Union[str, List[str]]): Audio file path(s).
    Returns:
        List[Dict[str, str]]: A list of dictionaries formatted for the Agent model.
    """
    inputs = []
    # Add each text string to the inputs
    if isinstance(texts, str):
        texts = [texts]
    for text in texts:
        inputs.append({
            "type": "text",
            "content": text
        })
    # Add each audio file path to the inputs
    if isinstance(audios, str):
        audios = [audios]
    for audio in audios:
        inputs.append({
            "type": "audio",
            "content": audio
        })
    return inputs

def _extract_code(text: str) -> str:
    """Helper to extract code block from model response."""
    try:
        # Find code between ```python and ``` tags
        start = text.split("```python\n")[-1]
        end = start.split("```")[0]
        return end.strip()
    except IndexError:
        return text
    
def get_media_wrapper(model_name: str) -> Optional[OpenAIWrapper]:
    """Get appropriate wrapper for media handling based on model name.
    
    Args:
        model_name (str): The model name (e.g., 'openai/gpt-4o')
        
    Returns:
        Optional[OpenAIWrapper]: The wrapper for the specified model
    """
    if model_name.startswith('openai/'):
        return OpenAIWrapper(model_name=model_name.split('/')[-1])
    return None

def prepare_media_messages(prompt: str, media_path: Union[str, Image.Image], model_name: str) -> List[Dict[str, Any]]:
    """Prepare messages for media input.
    
    Args:
        prompt (str): The text prompt
        media_path (Union[str, Image.Image]): The media file path or PIL Image
        model_name (str): The model name
        
    Returns:
        List[Dict[str, Any]]: Messages formatted for the model
    """
    # For images
    if isinstance(media_path, str) and any(media_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
        media = Image.open(media_path)
    else:
        media = media_path if isinstance(media_path, Image.Image) else None
    
    messages = [
        {"type": "text", "content": prompt}
    ]
    
    if media:
        messages.append({"type": "image", "content": media})
    
    return messages