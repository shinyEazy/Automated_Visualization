import json
import requests
from typing import Dict, Any, Optional, List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Postman API Server")

@mcp.tool()
def execute_api_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Execute an HTTP API request
    
    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)
        url: The API endpoint URL
        headers: Optional dictionary of headers
        params: Optional dictionary of query parameters
        body: Optional request body (JSON string or raw text)
        timeout: Request timeout in seconds (default: 30)
    
    Returns:
        Dictionary containing response data including status, headers, and body
    """
    try:
        # Prepare headers
        request_headers = headers or {}
        
        # Prepare request kwargs
        request_kwargs = {
            'method': method.upper(),
            'url': url,
            'headers': request_headers,
            'timeout': timeout
        }
        
        # Add query parameters if provided
        if params:
            request_kwargs['params'] = params
        
        # Handle request body
        if body:
            # Try to parse as JSON first
            try:
                parsed_body = json.loads(body)
                request_kwargs['json'] = parsed_body
                if 'content-type' not in [k.lower() for k in request_headers.keys()]:
                    request_headers['Content-Type'] = 'application/json'
            except json.JSONDecodeError:
                # If not JSON, send as raw data
                request_kwargs['data'] = body
        
        # Execute the request
        response = requests.request(**request_kwargs)
        
        # Try to parse response as JSON
        try:
            response_body = response.json()
        except (json.JSONDecodeError, ValueError):
            response_body = response.text
        
        return {
            'status_code': response.status_code,
            'status_text': response.reason,
            'headers': dict(response.headers),
            'body': response_body,
            'url': response.url,
            'elapsed_ms': response.elapsed.total_seconds() * 1000,
            'success': response.ok
        }
        
    except requests.exceptions.Timeout:
        return {
            'error': 'Request timeout',
            'status_code': None,
            'success': False
        }
    except requests.exceptions.ConnectionError:
        return {
            'error': 'Connection error - could not reach the server',
            'status_code': None,
            'success': False
        }
    except requests.exceptions.RequestException as e:
        return {
            'error': f'Request failed: {str(e)}',
            'status_code': None,
            'success': False
        }
    except Exception as e:
        return {
            'error': f'Unexpected error: {str(e)}',
            'status_code': None,
            'success': False
        }

@mcp.tool()
def post_request(
    url: str,
    body: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Execute a POST request
    
    Args:
        url: The API endpoint URL
        body: Request body (JSON string or raw text)
        headers: Optional dictionary of headers
        params: Optional dictionary of query parameters
        timeout: Request timeout in seconds (default: 30)
    """
    return execute_api_request('POST', url, headers, params, body, timeout)

if __name__ == "__main__":
    mcp.run(transport="stdio")