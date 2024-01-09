import re

def find_return_to_start_index(code):
    lines = code.split('\n')
    first_line_indentation = len(lines[0]) - len(lines[0].lstrip())

    total_char_count = 0
    for i, line in enumerate(lines):
        stripped_line = line.lstrip()

        if stripped_line and not stripped_line.startswith('#'):
            indentation = len(line) - len(stripped_line)

            if i > 0 and indentation == first_line_indentation:
                return total_char_count

        total_char_count += len(line) + 1  # Add 1 for the newline character

    return -1  # Indicates no return to start indentation           


def extract_code_block(file_path, target_string):
    with open(file_path) as f:
        content = f.read()

    # matches most python identifiers [^\d\W]\w*
    target_pattern = re.compile(fr'(\bdef\b|\bclass\b)\s+{re.escape(target_string)}\(')
    matches = re.finditer(target_pattern, content)
    for match in matches:
        start_index = match.start()

        code_block = content[start_index:]
        index = find_return_to_start_index(code_block)
        if index == -1:
            return code_block.strip()

        return code_block[:index].strip()

    return None
    
def update_md_file(md_filepath, python_filepath):
    with open(md_filepath) as file:
        content = file.read()

    # find all occurrences of '<-- python: name -->' and extract the names
    pattern = r'<!-- python: (\w+) -->'
    matches =re.finditer(pattern, content)
    names = [match.group(1) for match in matches]

    # extract the code blocks using the names
    code_blocks = {
        name: extract_code_block(python_filepath, name)
        for name in names
    }
    print(code_blocks)

    # update the markdown file
    for name, code_block in code_blocks.items():
        if code_block is None:
            print(f"Failed to find python function or class {name=} in {python_filepath}")
            continue
        placeholder = rf'<!-- python: {name} -->\n```([\s\S]*?)```'
        old_content = content
        content = re.sub(placeholder, lambda _ : f'<!-- python: {name} -->\n```python\n{code_block}\n```', content)
        print(old_content == content)

    with open(md_filepath, 'w') as file:
        file.write(content)
