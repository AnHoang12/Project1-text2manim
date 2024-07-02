import time
import asyncio
import g4f
import json
import random

start = time.time()

async def process_api_request(request):
    while True:
            await asyncio.sleep(random.randint(10, 20))
            response = await g4f.ChatCompletion.create_async(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": request}],
            )   

if __name__ == "__main__":

    start = time.time()
    requests = "Write me a Blender program to illutrate a sphere. Generate code, not text"
    responses = asyncio.run(process_api_request(requests))
    end = time.time()
    print(f"Time elapsed: {end - start}")
    print(responses)
