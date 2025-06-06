def extract_html_block(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    in_html_block = False
    html_lines = []
    for line in lines:
        if not in_html_block and line.strip().startswith("```html"):
            in_html_block = True
            continue
        elif in_html_block and line.strip().startswith("```"):
            break  # end of html block
        if in_html_block:
            html_lines.append(line)

    if html_lines:
        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.writelines(html_lines)
        print(f"HTML block extracted to {output_path}")
    else:
        print("No HTML code block found.")

extract_html_block("response.txt", "response_extracted.html")