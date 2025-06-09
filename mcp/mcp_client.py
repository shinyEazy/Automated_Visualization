from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio
import json
import os
import dotenv

dotenv.load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

server_params = StdioServerParameters(
    command="python",
    args=[os.getenv("MCP_SERVER_PATH")],
)

async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)

            data_path = r"C:\Users\ASUS\Desktop\workspace\Automated_Visualization\backend\problems\text_classification_verified\data\goemotions.csv"
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": f"given the data, get the first row of data. The data is in {data_path}"})
            return agent_response

if __name__ == "__main__":
    result = asyncio.run(run_agent())
    with open('response.json', 'w', encoding='utf-8') as f:
        json.dump(dict(result), f, ensure_ascii=False, indent=2, default=str)
    print(result.get('messages')[-1].content)





