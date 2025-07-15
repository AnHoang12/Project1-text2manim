import re
from openai import OpenAI
import os
from subprocess import Popen, PIPE, CalledProcessError, run
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

def wrap_prompt(prompt: str) -> str:
    return f"Animation Request: {prompt}"

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
    # Replace deprecated functions with correct Manim ones
    return code_response.replace("ShowCreation", "Create")

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

def generate(history, new_prompt, scene_type: str):
    # Generate the code using the chat API
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

    # Append the history of messages to maintain context
    messages += history
    messages.append(new_prompt)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    history.append(new_prompt)
    history.append({"role": "assistant", "content": response.choices[0].message.content})

    # Extract the generated code from the response
    code_response = response.choices[0].message.content
    extracted_code = extract_code(code_response)
    construct_code = extract_construct_code(extracted_code)
    
    # Create file content using the determined scene type (Scene or ThreeDScene)
    final_code = create_file_content(construct_code, scene_type)

    return final_code

def code_to_video(code, file_name, file_class, iteration):
    # Save the generated code to a .py file
    if not code:
        raise ValueError("No code provided")

    with open(file_name, "w") as f:
        f.write(code)

    try:
        # Execute Manim to render the video
        process = Popen(
            [
                "manim",
                file_name,
                file_class,
                "-pql",  # play and quality flags
            ],
            stdout=PIPE,
            stderr=PIPE,
            cwd=os.path.dirname(os.path.realpath(__file__)),
        )

        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("Video created successfully.")
            video_file_path = os.path.join(os.getcwd(), "media/videos", file_name.replace(".py", ""), "480p15", f"{file_class}.mp4")
            target_path = "video"
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            target_file_path = os.path.join(target_path, f"{file_class}_iteration_{iteration}.mp4")
            
            # Rename the video file to the target location
            os.rename(video_file_path, target_file_path)
            if os.path.exists(target_file_path):
                print(f"Video saved to: {target_file_path}")
                run(["ffplay", target_file_path])  # Play the video
                return {"message": "Video generation completed", "video_path": target_file_path}
            else:
                raise FileNotFoundError("Generated video file not found.")
        else:
            print(f"Manim command failed with return code {process.returncode}")
            print(f"stdout: {stdout.decode()}")
            print(f"stderr: {stderr.decode()}")
            raise RuntimeError(f"Video generation failed: {stderr.decode()}")

    except CalledProcessError as e:
        print(f"Subprocess error: {e}")
        raise RuntimeError(f"Subprocess error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise RuntimeError(f"Unexpected error occurred: {e}")

def chat(history):
    prompt = input("Prompt: ")
    new_prompt = {"role": "user", "content": wrap_prompt(prompt)}
    history.append(new_prompt)

    # Determine if the scene should be 2D or 3D based on the prompt
    scene_type = determine_scene_type(prompt)

    # Generate the Python code for the Manim scene
    code_to_execute = generate(history, new_prompt, scene_type)
    print(code_to_execute)

    file_name = "GenScene.py"
    file_class = "GenScene"
    iteration = len(history) // 2  

    try:
        result = code_to_video(code_to_execute, file_name, file_class, iteration)
        print(result)
    except Exception as e:
        print(f"Error during video generation: {e}")

def start():
    history = []
    while True:
        chat(history)

if __name__ == "__main__":
    start()
