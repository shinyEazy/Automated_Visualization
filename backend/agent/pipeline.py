with open("curl_command_generated.txt", "r") as f:
    input_payload = f.read()
with open("response.json", "r") as f:
    output_payload = f.read()


print(input_payload[:400])
print(output_payload[:2000])
