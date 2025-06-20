from tools.image_to_base64 import ImageToBase64
from tools.get_data_in_csv import GetDataInCsv
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Curl Generator Agent")

@mcp.tool()
def image_to_base64(image_path: str) -> str:
    """Convert image to base64"""
    return ImageToBase64().image_to_base64(image_path)

@mcp.tool()
def get_data_in_csv(csv_path: str) -> str:
    """Get the first row of data from the CSV file (excluding header)"""
    return GetDataInCsv().get_data_in_csv(csv_path)

if __name__ == "__main__":
    mcp.run()