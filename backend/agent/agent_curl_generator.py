# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# import dspy
# import asyncio
# import re

# task_name = "text_classification"

# with open("tools/base64_image.txt", "r") as f:
#     BASE64_IMAGE = f.read()

# with open("tools/audio_data.txt", "r") as f:
#     AUDIO_DATA = f.read()

# with open("tools/sampling_rate.txt", "r") as f:
#     SAMPLING_RATE = f.read()

# MCP_SERVERS = {
#     "filesystem_server": StdioServerParameters(
#         command="npx",
#         args=[
#             "-y",
#             "@modelcontextprotocol/server-filesystem",
#             rf"C:\Users\ASUS\Desktop\workspace\Automated_Visualization\backend\problems\{task_name}"
#         ]
#     ),
# }

# async def get_all_tools():
#     all_tools = []
#     sessions = []
#     for name, params in MCP_SERVERS.items():
#         stdio_ctx = stdio_client(params)
#         read, write = await stdio_ctx.__aenter__()
#         session_ctx = ClientSession(read, write)
#         session = await session_ctx.__aenter__()
#         await session.initialize()
#         tools = await session.list_tools()
#         for tool in tools.tools:
#             all_tools.append(dspy.Tool.from_mcp_tool(session, tool))
#         sessions.append((session_ctx, stdio_ctx))
    
#     return all_tools, sessions

# async def disconnect_all(sessions):
#     for session_ctx, stdio_ctx in sessions:
#         await session_ctx.__aexit__(None, None, None)
#         await stdio_ctx.__aexit__(None, None, None)

# class CurlGenerator(dspy.Signature):
#     """You are an agent that generates mock curl commands to call the API at:
#     - Use the appropriate tools to generate the mock curl command.
#     - Consider use the example input from data folder to generate the curl command.
#     - When POST body ready, use the tool to generate the mock curl command.
#     - Return a complete mock curl command that can call the specified API.
#     - Create a mock curl command based on example input in data folder.
#     - If POST body require base64 image:
#         {
#             "data": "base64_image"
#         }
#     - Else, use the tool to generate the mock curl command.
#     - Return mock curl command only, no other text.
#     """

#     task_description: str = dspy.InputField(desc="The task description")
#     curl_command: str = dspy.OutputField(desc="The mock curl command to call the API")

# dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano"))

# async def run(task_description):
#     all_tools, sessions = await get_all_tools()
#     try:
#         react = dspy.ReAct(CurlGenerator, tools=all_tools)
#         result = await react.acall(task_description=task_description)
#         print(result)
#         curl_command = result.curl_command
#         curl_command = re.sub(r'("data"\s*:\s*")([^"]*base64[^"]*)(")', rf'\1{BASE64_IMAGE}\3', curl_command)
#         curl_command = re.sub(r'"audio_data"\s*:\s*\[[^\]]*\]', f'"audio_data": {AUDIO_DATA}', curl_command)
#         curl_command = re.sub(r'"sampling_rate"\s*:\s*\d+', f'"sampling_rate": {SAMPLING_RATE}', curl_command)
#         with open("curl_command_generated.txt", "w") as f:
#             f.write(curl_command)
#     finally:
#         await disconnect_all(sessions)

# if __name__ == "__main__":
#     asyncio.run(run("curl"))


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import dspy
import asyncio
import re
import yaml
import chardet
import json

task_name = "object_detection_in_image"

with open("tools/base64_image.txt", "r") as f:
    BASE64_IMAGE = f.read()

with open("tools/audio_data.txt", "r") as f:
    AUDIO_DATA = f.read()

with open("tools/sampling_rate.txt", "r") as f:
    SAMPLING_RATE = f.read()

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

task_yaml_content = safe_read_file(f"../problems/{task_name}/task.yaml")
task_data = yaml.safe_load(task_yaml_content)
model_info = task_data.get("model_information", {})

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

async def run(task_description):
    curl_generator = dspy.ChainOfThought(CurlGenerator)
    result = await curl_generator.acall(model_info=json.dumps(model_info), task_description=task_description)
    print(result)
    curl_command = result.curl_command
    curl_command = re.sub(r'("data"\s*:\s*")([^"]*base64[^"]*)(")', rf'\1{BASE64_IMAGE}\3', curl_command)
    curl_command = re.sub(r'"audio_data"\s*:\s*\[[^\]]*\]', f'"audio_data": {AUDIO_DATA}', curl_command)
    curl_command = re.sub(r'"sampling_rate"\s*:\s*\d+', f'"sampling_rate": {SAMPLING_RATE}', curl_command)
    with open("curl_command_generated.txt", "w") as f:
        f.write(curl_command)

if __name__ == "__main__":
    asyncio.run(run("curl"))