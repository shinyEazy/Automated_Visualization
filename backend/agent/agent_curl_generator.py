from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import dspy
import asyncio
import re

task_name = "text_classification"

with open("tools/base64_image.txt", "r") as f:
    BASE64_IMAGE = f.read()

MCP_SERVERS = {
    "filesystem_server": StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            rf"C:\Users\ASUS\Desktop\workspace\Automated_Visualization\backend\problems\{task_name}"
        ]
    ),
}

async def get_all_tools():
    all_tools = []
    sessions = []
    for name, params in MCP_SERVERS.items():
        stdio_ctx = stdio_client(params)
        read, write = await stdio_ctx.__aenter__()
        session_ctx = ClientSession(read, write)
        session = await session_ctx.__aenter__()
        await session.initialize()
        tools = await session.list_tools()
        for tool in tools.tools:
            all_tools.append(dspy.Tool.from_mcp_tool(session, tool))
        sessions.append((session_ctx, stdio_ctx))
    
    return all_tools, sessions

async def disconnect_all(sessions):
    for session_ctx, stdio_ctx in sessions:
        await session_ctx.__aexit__(None, None, None)
        await stdio_ctx.__aexit__(None, None, None)

class CurlGenerator(dspy.Signature):
    """You are an agent that generates mock curl commands to call the API at:
    - Use the appropriate tools to generate the mock curl command.
    - Consider use the example input from data folder to generate the curl command.
    - When POST body ready, use the tool to generate the mock curl command.
    - Return a complete mock curl command that can call the specified API.
    - Create a mock curl command based on example input in data folder.
    - If POST body require base64 image:
        {
            "data": "base64_image"
        }
    - Else, use the tool to generate the mock curl command.
    - Return mock curl command only, no other text.
    """

    task_description: str = dspy.InputField(desc="The task description")
    curl_command: str = dspy.OutputField(desc="The mock curl command to call the API")

dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano"))

async def run(task_description):
    all_tools, sessions = await get_all_tools()
    try:
        react = dspy.ReAct(CurlGenerator, tools=all_tools)
        result = await react.acall(task_description=task_description)
        print(result)
        curl_command = result.curl_command
        curl_command = re.sub(r'("data"\s*:\s*")([^"]*base64[^"]*)(")', rf'\1{BASE64_IMAGE}\3', curl_command)
        with open("curl_command_generated.txt", "w") as f:
            f.write(curl_command)
    finally:
        await disconnect_all(sessions)

if __name__ == "__main__":
    asyncio.run(run("curl"))