import os
import base64
import requests
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import io
import json
from dotenv import load_dotenv

load_dotenv()

class OpenAIWrapper:
    """A wrapper for OpenAI API."""

    def __init__(self, model_name="gpt-4o"):
        """Initialize with OpenAI API key and model name.
        
        Args:
            model_name (str, optional): The OpenAI model to use. Defaults to "gpt-4o".
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.model_name = model_name
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
    def __call__(self, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Call the OpenAI API with the given messages.
        
        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries with type and content keys
            metadata (Optional[Dict[str, Any]], optional): Optional metadata. Defaults to None.
            
        Returns:
            str: The model's response text
        """
        openai_messages = self._convert_to_openai_format(messages)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": 0.2,
            "max_tokens": 4000
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
        
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    
    def _convert_to_openai_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format.
        
        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries with type and content keys
            
        Returns:
            List[Dict[str, Any]]: Messages in OpenAI format
        """
        openai_messages = []
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant specialized in creating educational mathematical animations."
        }
        openai_messages.append(system_message)
        
        # Process each message based on its type
        for message in messages:
            if message["type"] == "text":
                openai_messages.append({
                    "role": "user",
                    "content": message["content"]
                })
            elif message["type"] == "image":
                # Handle image content
                image_content = message["content"]
                if isinstance(image_content, str):  # Image path
                    with open(image_content, "rb") as img_file:
                        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                elif isinstance(image_content, Image.Image):  # PIL Image
                    buffered = io.BytesIO()
                    image_content.save(buffered, format="JPEG")
                    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                else:
                    raise ValueError(f"Unsupported image type: {type(image_content)}")
                
                openai_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here's an image for reference:"},
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                })
            elif message["type"] == "video":
                # For video, just include a message that videos are not supported
                # and will need to be handled differently
                openai_messages.append({
                    "role": "user",
                    "content": "Note: There was a video here, but OpenAI does not support video inputs. Please use image frames instead."
                })
        
        return openai_messages 