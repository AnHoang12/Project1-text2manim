# This is essential for the release to work

# Import core components
from src.core.code_generator import CodeGenerator
from src.core.video_planner import VideoPlanner
from src.core.video_renderer import VideoRenderer

# Import our OpenAI wrapper
from mllm_tools.openai import OpenAIWrapper

# Version
__version__ = "0.1.0"