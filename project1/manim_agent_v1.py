import re
from openai import OpenAI
import os
from subprocess import Popen, PIPE, CalledProcessError, run
from dotenv import load_dotenv
import json
import logging
from pydantic import BaseModel, Field
from typing import List, Optional


load_dotenv()

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

llm = "gpt-4"

logger = logging.getLogger(__name__)

class AnimationScenario(BaseModel):
    """Structured scenario for animation generation."""
    title: str = Field(..., description="Title of the animation")
    objects: List[str] = Field(..., description="Mathematical objects to include in the animation")
    transformations: List[str] = Field(..., description="Transformations to apply to the objects")
    equations: Optional[List[str]] = Field(None, description="Mathematical equations to visualize")
    story_board: Optional[List[dict]] = Field(None, description="Detailed storyboard sections with visual and narrative information")

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
    return code_response.replace("ShowCreation", "Create")

def determine_scene_type(prompt: str) -> str:
    three_d_keywords = ['3d', 'sphere', 'camera', 'rotate', 'set_camera_orientation', 'z-axis']
    if any(keyword in prompt.lower() for keyword in three_d_keywords):
        return "ThreeDScene"
    return "Scene"

def create_file_content(code_response: str, scene_type: str) -> str:
    return f"""# Manim code generated with OpenAI GPT
# Command to generate animation: manim -pql GenScene.py GenScene 

from manim import *
from math import *

class GenScene({scene_type}):
    def construct(self):
{code_static_corrector(code_response)}"""

def extract_scenario_direct(prompt: str, complexity: str = "medium") -> AnimationScenario:
    """Direct implementation of scenario extraction without using RunContext."""
    
    # Use OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """
Create a storyboard for a math/physics educational animation. Focus on making concepts clear for beginners.
Break down the concept into logical steps that build understanding progressively.
For each step, describe what should be shown visually and what explanatory text should appear.
The output should be structured as a valid JSON object containing:
- title: A clear, engaging title
- objects: Mathematical objects to include (e.g., "coordinate_plane", "function_graph")
- transformations: Animation types to use (e.g., "fade_in", "transform")
- equations: Mathematical equations to feature (can be null)
- storyboard: 5-7 sections, each with:
  * section_name: Section name (e.g., "Introduction")
  * time_range: Timestamp range (Must be in "0:00-2:00")
  * narration: What the narrator says
  * visuals: What appears on screen
  * animations: Specific animations
  * key_points: 1-2 main takeaways 
The Json with the following structure:
{
  "title": "Name of the animation",
  "objects": ["list of mathematical objects to be visualized"],
  "transformations": ["list of transformations or animations to apply"],
  "equations": ["Mathematical equations to feature"],
  "storyboard": [
    {
      "step": 1,
      "time": "0:00-0:30",  
      "visual": "description of what to show",
      "text": "explanation to display"
    }
  ]
}
"""},
            {"role": "user", "content": f"Create an animation storyboard for: '{prompt}'."
                                        f" Complexity level: {complexity}. Make it beginner-friendly "
                                        f" with clear explanations and visual examples."}
        ],
        temperature=0.7,
        max_tokens=2000
    )
    
    content = response.choices[0].message.content
    
    try:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            scenario_dict = json.loads(json_str)
            # Get basic scenario info
            title = scenario_dict.get('title', f"{prompt.capitalize()} Visualization")
            objects = scenario_dict.get('objects', [])
            transformations = scenario_dict.get('transformations', [])
            equations = scenario_dict.get('equations', None)
            storyboard = scenario_dict.get('storyboard', [])
        
            return AnimationScenario(
                title=title,
                objects=objects,
                transformations=transformations,
                equations=equations,
                story_board=storyboard
            )
        else:
            logger.warning("No JSON structure found in response")
            return AnimationScenario(
                title=f"{prompt.capitalize()} Visualization",
                objects=[],
                transformations=[],
                equations=None,
                story_board=[]
            )
    except Exception as e:
        logger.error(f"Error parsing scenario JSON: {e}")
        # Fallback to a basic scenario
        return AnimationScenario(
            title=f"{prompt.capitalize()} Visualization",
            objects=[],
            transformations=[],
            equations=None,
            story_board=[]
        )


def generate(history, new_prompt, scene_type: str):
    prompt_text = new_prompt["content"].replace("Animation Request: ", "")
    scenario = extract_scenario_direct(prompt_text)

    if not scenario:
        print("Failed to generate scenario. Please check the API response or input prompt.")
        return None

    print(f"Generated Scenario: {scenario.model_dump()}")

    messages = history + [
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

VISUAL STRUCTURE AND LAYOUT:
1. Structure the animation as a narrative with clear sections (introduction, explanation, conclusion)
2. Create title screens with engaging typography and animations
3. Position ALL elements with EXPLICIT coordinates using shift() or move_to() methods
4. Ensure AT LEAST 2.0 units of space between separate visual elements
5. For equations, use MathTex with proper scaling (scale(0.8) for complex equations)
6. Group related objects using VGroup and arrange them with arrange() method
7. When showing multiple equations, use arrange_in_grid() or arrange() with DOWN/RIGHT
8. For graphs, set explicit x_range and y_range with generous padding around functions
9. Scale ALL text elements appropriately (Title: 1.2, Headers: 1.0, Body: 0.8)
10. Use colors consistently and meaningfully (BLUE for emphasis, RED for important points)

CRITICAL: ELEMENT MANAGEMENT AND STEP-BY-STEP REQUIREMENTS:
1. NEVER show too many elements on screen at once - max 3-4 related elements at any time
2. ALWAYS use self.play(FadeOut(element)) to explicitly remove elements when moving to a new concept
3. DO NOT use self.clear() as it doesn't actually remove elements from the scene
4. Implement strict SEQUENTIAL animation - introduce only ONE concept or element at a time
5. Use self.wait(0.7) to 1.5 for short pauses and self.wait(2) for important concepts
6. Organize the screen into distinct regions (TOP for titles, CENTER for main content, BOTTOM for explanations)
7. For sequential steps in derivations or proofs, use transform_matching_tex() to smoothly evolve equations
8. Use MoveToTarget() for repositioning elements that need to stay on screen between steps
9. At the end of each section, EXPLICITLY remove all elements with self.play(FadeOut(elem1, elem2, ...))
10. When positioning new elements, verify they won't overlap existing elements
11. For elements that must appear together, use VGroup but animate their creation one by one

ANIMATION TECHNIQUES:
1. Use FadeIn for introductions of new elements
2. Apply TransformMatchingTex when evolving equations
3. Use Create for drawing geometric objects
4. Implement smooth transitions between different concepts with ReplacementTransform
5. Highlight important parts with Indicate or Circumscribe
6. Add appropriate pauses: self.wait(0.7) after minor steps, self.wait(1.5) after important points
7. For complex animations, break them into smaller steps with appropriate timing
8. Use MoveAlongPath for demonstrating motion or change over time
9. Create emphasis with scale_about_point or succession of animations
10. Use camera movements sparingly and smoothly

EDUCATIONAL CLARITY:
1. Begin with simple concepts and build to more complex ones
2. Reveal information progressively, not all at once
3. Use visual metaphors to represent abstract concepts
4. Include clear labels for all important elements
5. When showing equations, animate their components step by step
6. Provide visual explanations alongside mathematical notation
7. Use consistent notation throughout the animation
8. Show practical applications or examples of the concept
9. Summarize key points at the end of the animation
    """
    },
        {
            "role": "user", 
            "content": 
                    "Generate Manim code based on the following animation scenario. "
                    f"Scenario: {json.dumps(scenario.model_dump(), indent=2)}"
                    }
    ]

    response = client.chat.completions.create(
        model=llm,
        messages=messages
    )

    history.append(new_prompt)
    history.append({"role": "assistant", "content": response.choices[0].message.content})

    code_response = response.choices[0].message.content
    extracted_code = extract_code(code_response)
    construct_code = extract_construct_code(extracted_code)

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
