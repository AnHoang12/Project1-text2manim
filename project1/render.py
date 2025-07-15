import os
from subprocess import Popen, PIPE, CalledProcessError, run

def code_to_video(code, file_name, file_class, iteration):
    if not code:
        raise ValueError("No code provided")

    with open(file_name, "w") as f:
        f.write(code)

    try:
        process = Popen(
            [
                "manim",
                file_name,
                file_class,
                "--format=mp4",
                "--media_dir",
                ".",
                "--custom_folders",
            ],
            stdout=PIPE,
            stderr=PIPE,
            cwd=os.path.dirname(os.path.realpath(__file__)),
        )

        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("Video created")
            video_file_path = os.path.join(os.getcwd(), f"{file_class}.mp4")
            target_path = "video"
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            target_file_path = os.path.join(target_path, f"{file_class}.mp4")
            os.rename(video_file_path, target_file_path)
            if os.path.exists(target_file_path):
                print(f"Video saved to: {target_file_path}")
                run(["ffplay", target_file_path])
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
