import re
from g4f.client import Client

client = Client()

def generate(history, new_prompt):
    messages = [
        {
            "role": "system",
            "content": """
You are an assistant that knows about Manim. Manim is a mathematical animation engine that is used to create videos programmatically.

The following is an example of the code:
\`\`\`
from manim import *
from math import *

class GenScene(Scene):
    def construct(self):
        c = Circle(color=BLUE)
        self.play(Create(c))
\`\`\`

# Rules
1. Always use GenScene as the class name, otherwise, the code will not work.
2. Always use self.play() to play the animation, otherwise, the code will not work.
3. Do not use text to explain the code, only the code.
4. Do not explain the code, only the code.
    """
        },
    ]

    messages += history
    messages.append(new_prompt)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    history.append(new_prompt)
    history.append({"role": "assistant", "content": response.choices[0].message.content})

    return response.choices[0].message.content

def extract_code(text: str) -> str:
    pattern = re.compile(r"```(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    else:
        return text

def extract_construct_code(code_str: str) -> str:
    pattern = r"def construct\(self\):([\s\S]*)"
    match = re.search(pattern, code_str)
    if match:
        return match.group(1)
    else:
        return ""

def code_static_corrector(code_response: str) -> str:
    code_response = code_response.replace("ShowCreation", "Create")
    return code_response

# Detect if the scene is 2D or 3D
def determine_scene_type(prompt: str) -> str:
    # Keywords that imply a 3D scene
    three_d_keywords = ['3d', 'sphere', 'camera', 'rotate', 'set_camera_orientation', 'z-axis']
    
    # If any of these keywords are found in the prompt, return "ThreeDScene"
    if any(keyword in prompt.lower() for keyword in three_d_keywords):
        return "ThreeDScene"
    
    # Otherwise, assume it's a 2D scene
    return "Scene"


def create_file_content(code_response: str, scene_type: str) -> str:
    # Generate the final Python file content for the Manim scene
    return f"""# Manim code generated with OpenAI GPT
# Command to generate animation: manim -pql GenScene.py GenScene 

from manim import *
from math import *

class GenScene({scene_type}):
    def construct(self):
{code_static_corrector(code_response)}"""



