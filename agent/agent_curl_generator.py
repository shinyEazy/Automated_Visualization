import dspy
import asyncio
import re
import yaml
import chardet
import json
import os

def safe_read_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected["encoding"] or "latin-1"
            
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

class CurlGenerator(dspy.Signature):
    """You are an agent that generates mock curl commands to call the API at:
    - Return a complete mock curl command that can call the specified API.
    - If POST body require base64 image:
        {
            "data": "base64_image"
        }
    - Return mock curl command only, no other text.
    """
    model_info: str = dspy.InputField(desc="The model information")
    task_description: str = dspy.InputField(desc="The task description")
    curl_command: str = dspy.OutputField(desc="The mock curl command to call the API")

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano", cache=False, cache_in_memory=False))

async def generate_curl_for_task(task_name: str):
    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read tool data
    tools_dir = os.path.join(base_dir, "tools")
    with open(os.path.join(tools_dir, "base64_image.txt"), "r") as f:
        BASE64_IMAGE = f.read()
    
    with open(os.path.join(tools_dir, "audio_data.txt"), "r") as f:
        AUDIO_DATA = f.read()
    
    with open(os.path.join(tools_dir, "sampling_rate.txt"), "r") as f:
        SAMPLING_RATE = f.read()

    # Read task.yaml
    task_yaml_path = os.path.join(base_dir, "problems", task_name, "task.yaml")
    task_yaml_content = safe_read_file(task_yaml_path)
    task_data = yaml.safe_load(task_yaml_content)
    model_info = task_data.get("model_information", {})
    
    # Generate curl command
    curl_generator = dspy.ChainOfThought(CurlGenerator)
    result = await curl_generator.acall(
        model_info=json.dumps(model_info),
        task_description="curl"
    )
    
    # Process and save curl command
    curl_command = result.curl_command
    curl_command = re.sub(r'("data"\s*:\s*")([^"]*base64[^"]*)(")', rf'\1{BASE64_IMAGE}\3', curl_command)
    curl_command = re.sub(r'"audio_data"\s*:\s*\[[^\]]*\]', f'"audio_data": {AUDIO_DATA}', curl_command)
    curl_command = re.sub(r'"sampling_rate"\s*:\s*\d+', f'"sampling_rate": {SAMPLING_RATE}', curl_command)
    
    output_path = os.path.join(base_dir, "problems", task_name, "curl_command_generated.txt")
    with open(output_path, "w") as f:
        f.write(curl_command)
    
    return curl_command

# Original main preserved for single-task execution
async def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent_curl_generator.py <task_name>")
        sys.exit(1)
    task_name = sys.argv[1]
    await generate_curl_for_task(task_name)

if __name__ == "__main__":
    asyncio.run(main())