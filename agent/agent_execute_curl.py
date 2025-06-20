import requests
import uncurl
import sys
import json
import os

def execute_curl_for_task(task_name: str):
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    curl_file_path = os.path.join(base_dir, "problems", task_name, "curl_command_generated.txt")
    response_file_path = os.path.join(base_dir, "problems", task_name, "response.json")
    
    # Read curl command
    try:
        with open(curl_file_path, "r") as file:
            # Join lines and strip whitespace
            curl_command = ' '.join(line.strip().rstrip('\\') for line in file if line.strip())
        if not curl_command:
            print(f"❌ Error: The file '{curl_file_path}' is empty.")
            return False
    except FileNotFoundError:
        print(f"❌ Error: The file '{curl_file_path}' was not found.")
        return False

    # Parse using uncurl
    try:
        python_code = uncurl.parse(curl_command)
    except Exception as e:
        print(f"❌ Error during cURL conversion: {e}")
        return False

    # Add assignment so we can capture the response
    python_code_with_assignment = "response = " + python_code

    # Execute in a safe environment
    local_vars = {}
    try:
        exec(python_code_with_assignment, {"requests": requests}, local_vars)
    except Exception as e:
        print(f"❌ Error executing the generated code: {e}")
        return False

    # Process the response
    response = local_vars.get("response")
    if response:
        try:
            json_response = response.json()
            with open(response_file_path, "w") as file:
                file.write(json.dumps(json_response, indent=4, ensure_ascii=False))
            print(f"✅ Response saved to {response_file_path}")
            return True
        except requests.exceptions.JSONDecodeError:
            # If not JSON, save as text
            with open(response_file_path, "w") as file:
                file.write(response.text)
            print(f"✅ Non-JSON response saved to {response_file_path}")
            return True
        except Exception as e:
            print(f"❌ Error processing response: {e}")
            return False
    else:
        print("❌ Error: No response object found after execution.")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent_execute_curl.py <task_name>")
        sys.exit(1)
    
    task_name = sys.argv[1]
    success = execute_curl_for_task(task_name)
    
    if success:
        print(f"✅ Execution completed for task: {task_name}")
        sys.exit(0)
    else:
        print(f"❌ Execution failed for task: {task_name}")
        sys.exit(1)