from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import dspy
import asyncio
import subprocess
import sys
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

server_params = StdioServerParameters(
    command="python",
    args=[r"C:\Users\ASUS\Desktop\workspace\Automated_Visualization\backend\agent\mcp_server.py"],
    env=None,
)

class CurlGenerator(dspy.Signature):
    """
    You are an curl generator agent. You are given a list of tools to handle user requests.
    You should decide the right tool to use in order to fulfill users' requests.
    You should generate the curl command to call the API.
    Return the curl command only.
    """

    user_request: str = dspy.InputField()
    curl_command: str = dspy.OutputField(desc="The curl command to call the API.")

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

async def run_with_retry(user_request, max_retries=3):
    """Run with retry logic and better error handling"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}")
            
            async with stdio_client(server_params) as (read, write):
                logger.info("Successfully created stdio client")
                
                async with ClientSession(read, write) as session:
                    logger.info("Created client session")
                    
                    # Add timeout to initialization
                    await asyncio.wait_for(session.initialize(), timeout=10.0)
                    logger.info("Session initialized successfully")
                    
                    tools = await session.list_tools()
                    logger.info(f"Retrieved {len(tools.tools)} tools")
                    
                    dspy_tools = []
                    for tool in tools.tools:
                        dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))
                        logger.info(f"Loaded tool: {tool.name}")
                    
                    react = dspy.ReAct(CurlGenerator, tools=dspy_tools)
                    result = await react.acall(user_request=user_request)
                    print(result.curl_command)
                    return result
                    
        except asyncio.TimeoutError:
            logger.error(f"Attempt {attempt + 1} timed out during initialization")
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logger.error("All attempts failed")
                raise

async def run(user_request):
    """Original run function with better error handling"""
    try:
        return await run_with_retry(user_request)
    except Exception as e:
        logger.error(f"Final error: {type(e).__name__}: {e}")
        print(f"ERROR: {e}")

if __name__ == "__main__":
    user_request = """
model_information:
  api_url: "http://34.87.113.245:8000/api/text-classification"
  name: j-hartmann/emotion-english-distilroberta-base
  description: The model was trained on 6 diverse datasets and predicts Ekman's 6 basic emotions, plus a neutral class.
  input_format: 
    type: json
    structure:
      texts:
        type: string
        description: A text passage written by the author.
  output_format: 
    description: A list of dict contains emotions (labels) and their corresponding scores (probabilities).
    type: List[dict]
    structure:
      label: 
        type: string
      score: 
        type: float
  parameters:
    config:
      id2label:
        "0": anger
        "1": disgust
        "2": fear
        "3": joy
        "4": neutral
        "5": sadness
        "6": surprise
      label2id:
        anger: 0
        disgust: 1
        fear: 2
        joy: 3
        neutral: 4
        sadness: 5
        surprise: 6
"""
    
    asyncio.run(run(user_request))