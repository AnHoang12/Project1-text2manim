from llm import generate, extract_code, extract_construct_code, create_file_content, determine_scene_type
from render import code_to_video
from prompt import wrap_prompt

def chat(history):
    prompt = input("Prompt: ")
    new_prompt = {"role": "user", "content": wrap_prompt(prompt)}
    history.append(new_prompt)
    scene_type = determine_scene_type(prompt)
    code_response = generate(history, new_prompt)
    extracted_code = extract_code(code_response)
    construct_code = extract_construct_code(extracted_code)
    code_to_execute = create_file_content(construct_code,scene_type)
    print(code_to_execute)

    file_name = "GenScene.py"
    file_class = "GenScene"
    iteration = len(history) // 2 

    try:
        result = code_to_video(code_to_execute, file_name, file_class, iteration)
        print(result)
    except Exception as e:
        print(e)

def start():
    history = []
    while True:
        chat(history)

if __name__ == "__main__":
    start()
