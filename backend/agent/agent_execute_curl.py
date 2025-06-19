import requests
import uncurl
import sys
import json

try:
    with open("curl_command_generated.txt", "r") as file:
        curl_command = file.read().strip()
    if not curl_command:
        print("Error: The file 'curl_command_generated.txt' is empty.")
        sys.exit(1)
except FileNotFoundError:
    print("Error: The file 'curl_command_generated.txt' was not found.")
    sys.exit(1)

try:
    python_code = uncurl.parse(curl_command)
except Exception as e:
    print(f"Error during cURL conversion: {e}")
    sys.exit(1)

python_code_with_assignment = "response = " + python_code

local_vars = {}
try:
    exec(python_code_with_assignment, {"requests": requests}, local_vars)
except Exception as e:
    print(f"\nError executing the generated code: {e}")
    sys.exit(1)

response = local_vars.get("response")
if response:
    try:
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))
        with open("response.json", "w") as file:
            file.write(json.dumps(response.json(), indent=4, ensure_ascii=False))
    except requests.exceptions.JSONDecodeError:
        print("Response body (text):\n", response.text)
else:
    print("\n❌ Error: No response object found after execution.")