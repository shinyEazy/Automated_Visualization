import requests
import uncurl
import sys
import json

curl_file_path = "curl_command_generated.txt"

try:
    with open(curl_file_path, "r") as file:
        # Join lines and strip whitespace
        curl_command = ' '.join(line.strip().rstrip('\\') for line in file if line.strip())
    if not curl_command:
        print(f"❌ Error: The file '{curl_file_path}' is empty.")
        sys.exit(1)
except FileNotFoundError:
    print(f"❌ Error: The file '{curl_file_path}' was not found.")
    sys.exit(1)

# Parse using uncurl
try:
    python_code = uncurl.parse(curl_command)
except Exception as e:
    print(f"❌ Error during cURL conversion: {e}")
    sys.exit(1)

# Add assignment so we can capture the response
python_code_with_assignment = "response = " + python_code

# Execute in a safe environment
local_vars = {}
try:
    exec(python_code_with_assignment, {"requests": requests}, local_vars)
except Exception as e:
    print(f"❌ Error executing the generated code: {e}")
    sys.exit(1)

# Process the response
response = local_vars.get("response")
if response:
    try:
        json_response = response.json()
        print("✅ JSON Response:\n", json.dumps(json_response, indent=4, ensure_ascii=False))
        with open("response.json", "w") as file:
            file.write(json.dumps(json_response, indent=4, ensure_ascii=False))
    except requests.exceptions.JSONDecodeError:
        print("⚠️ Response is not JSON. Text:\n", response.text)
else:
    print("❌ Error: No response object found after execution.")
