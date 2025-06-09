import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

async def main():
    client = MultiServerMCPClient(
        {
            "postman": {
                "command": "python",
                "args": [r"C:\Users\ASUS\Desktop\workspace\Automated_Visualization\postman_server.py"],
                "transport": "stdio",
            },
        }
    )
    
    tools = await client.get_tools()
    print("Available tools:", [tool.name for tool in tools])
    
    agent = create_react_agent(
        "gpt-4o-mini",
        tools
    )
    
    postman_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": """
        I have an emotion classification API. Please:
        1. Create a POST request to test it
        2. Use this API endpoint: http://34.87.113.245:8000/api/text-classification
        3. Send a JSON body with a 'texts' field containing the text: "I am so happy today!"
        4. The response should be input and output of the API in json format only.

        The API expects:
        - Method: POST
        - URL: http://34.87.113.245:8000/api/text-classification
        - Body: {"texts": "your text here"}
        - Content-Type: application/json
        """}]},
        config={"recursion_limit": 300}
    )
    
    with open("postman_response.txt", "w") as f:
        f.write(postman_response)


if __name__ == "__main__":
    asyncio.run(main())