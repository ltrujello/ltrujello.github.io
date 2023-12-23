import sys
import re
import yaml
from pathlib import Path
import shutil
import tempfile
import subprocess
import textwrap
import numpy as np
import argparse 
import logging

from PIL import Image
from pdf2image import convert_from_path

"""
Converts a LaTeX source file into markdown files.
- renders math with MathJax
- converts TikZ pictures to PNGs
- converts LaTeX tables to Markdown tables 
- converts LateX list environments to Markdown lists
- converts amsthm pkg environments (theorem, definition, proof, etc.) appropriately
- converts LaTeX chapter, sections to Markdown headers 

# TODO: don't replace \textbf{} in math environments. 
# TODO: address statement, description, minipage, figure, center environments
# To address description env
# - Replace each itemize env. If an itemize env contains a description env, then stop and replace the description env.
# - Replace each description env. If a description env contained an itemize env, then it already go replaced so ignore it. 
# TODO: consider compiling TikzPictures with surrounding begin{center} or minipage code
# TODO: handle section references
# DONE: handle isomarrow macro
# TODO: remove footnotes
"""

GENERATE_PNGS = False
COMPILE_TIKZS = False
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


begin_document = re.compile("\\\\begin{document}")
end_document = re.compile("\\\\end{document}")
chapter = re.compile(r"\\chapter{(.*?)}")
section = re.compile(r"\\section{(.*?)}")
textbf = re.compile("\\\\textbf{(.*?)}")
textit = re.compile("\\\\textit{(.*?)}")
emph_cmd = re.compile("\\\\emph{(.*?)}")
latex_quotes = re.compile("``(.*?)''")
display_math = re.compile("(\\\\\\[([\s\S]*?)\\\\\\])")
align_stmt = re.compile("(\\\\begin{align}([\s\S]*?)\\\\end{align})")
align_topbot_stmt = re.compile("\\\\begin{align_topbot}([\s\S]*?)\\\\end{align_topbot}")

align_star_stmt = re.compile("(\\\\begin{align\*}([\s\S]*?)\\\\end{align\*})")
gather_env = re.compile("(\\\\begin{gather}([\s\S]*?)\\\\end{gather})")
gather_star_env = re.compile("(\\\\begin{gather\*}([\s\S]*?)\\\\end{gather\*})")
newpage_cmd = re.compile("\\\\newpage")
tabular_env = re.compile("\\\\begin{tabular}([\s\S]*?)\\\\end{tabular}")
indent_space = re.compile("^\\s*")
ampersand = re.compile("&")
end_tabular_env = re.compile("\\\\end{tabular}")
description_env = re.compile("\\\\begin{description}([\s\S]*?)\\\\end{description}")
remark_env = re.compile("\\\\begin{remark}([\s\S]*?)\\\\end{remark}")

# tex comment
tex_comment = re.compile("(?<!\\\)%(.)*")

# tikz regex
center_env = re.compile("(\\\\begin{center}([\s\S]*?)\\\\end{center})")
tikz_stmt = re.compile("\\\\begin{tikzpicture}([\s\S]*?)\\\\end{tikzpicture}")
tikz_cd_stmt = re.compile("\\\\begin{tikzcd}([\s\S]*?)\\\\end{tikzcd}")
minipage_env = re.compile("(\\\\begin{minipage}([\s\S]*?)\\\\end{minipage})")

# itemize environment
itemize_env = re.compile("\\\\begin{itemize}([\s\S]*?)\\\\end{itemize}")
item_stmt = re.compile("\\\\item")
end_itemize_env = re.compile("\\\\end{itemize}")
end_description_env = re.compile("\\\\end{description}")

# label statement
label_stmt = re.compile(r"\\label{(.*?)}")
label_stmt2 = re.compile(r"label{(.*?)}")

# amsthm environments
definition_stmt = re.compile("\\\\begin{definition}([\s\S]*?)\\\\end{definition}")
proposition_stmt = re.compile("\\\\begin{proposition}([\s\S]*?)\\\\end{proposition}")
lemma_stmt = re.compile("\\\\begin{lemma}([\s\S]*?)\\\\end{lemma}")
corollary_stmt = re.compile("\\\\begin{corollary}([\s\S]*?)\\\\end{corollary}")
theorem_stmt = re.compile("\\\\begin{theorem}([\s\S]*?)\\\\end{theorem}")
thm_env = re.compile("\\\\begin{thm}([\s\S]*?)\\\\end{thm}")
example_stmt = re.compile("\\\\begin{example}([\s\S]*?)\\\\end{example}")
proof_stmt = re.compile("\\\\begin{prf}([\s\S]*?)\\\\end{prf}")

repl_def = lambda x: repl_asmthm_statement(x, "definition")
repl_prop = lambda x: repl_asmthm_statement(x, "proposition" )
repl_lemma = lambda x: repl_asmthm_statement(x, "lemma")
repl_corollary = lambda x: repl_asmthm_statement(x, "corollary")
repl_theorem = lambda x: repl_asmthm_statement(x, "theorem")
repl_example = lambda x: repl_asmthm_statement(x, "example")
repl_proof = lambda x: repl_asmthm_statement(x, "proof")
repl_remark = lambda x: repl_asmthm_statement(x, "remark")

EXTRA_MKDOCS_CSS = """<style>
.md-content {
    max-width: 80em;
}
</style>
"""

KNOWN_ENVS = [
    "align",
    "align*",
    "bmatrix",
    "pmatrix",
    "cases",
    "equation",
    "gather",
    "gather*",
]

def repl_asmthm_statement(match, class_name):
    env_contents = match.group(1)
    env_contents = textwrap.dedent(env_contents)
    repl = f"\n<span style=\"display:block\" class=\"{class_name}\">{env_contents}</span>"
    return repl


def repl_listing_env(match, env):
    code = match.group(0)

    items = []
    ind = 0
    while True:
        curr_match = re.search(item_stmt, code[ind:])
        # Early out if env contains no items
        if curr_match is None:
            return code

        next_match = re.search(item_stmt, code[ind + curr_match.end() :])

        if next_match is not None:
            items.append(code[ind + curr_match.end():ind + curr_match.end() + next_match.start()])
            ind = ind + curr_match.end() + next_match.start()
        else:
            if env == "itemize":
                next_match = re.search(end_itemize_env, code[ind + curr_match.end():])
            else:
                next_match = re.search(end_description_env, code[ind + curr_match.end():])

            if next_match is not None:
                items.append(code[ind + curr_match.end(): ind + curr_match.end() + next_match.start()])
            else:
                LOGGER.error(f"Failed to find \\item or \\end{{itemize}} in {code[ind + curr_match.end():]} in {chapter=} {section=}")
            break
    res = "\n"
    for item in items:
        res += f"* {item}\n\n"
    return res

repl_itemize_env = lambda x: repl_listing_env(x, "itemize")
repl_description_env = lambda x: repl_listing_env(x, "description")


def repl_tabular_env(match):
    contents = match.group(1)
    contents = re.sub("\\\\hline", "", contents)
    # find the index 
    num_brackets_seen = 0
    j = 0
    for ind, ch in enumerate(contents):
        if ch == "{":
            num_brackets_seen += 1
        elif ch == "}":
            num_brackets_seen -= 1
            if num_brackets_seen == 0:
                j = ind + 1
                break
    rows = []
    while j < len(contents):
        code = contents[j:]
        ind = 0
        row = []
        while True:
            curr_match = re.search(ampersand, code[ind:])
            next_match = re.search("\\\\\\\\", code[ind:])
            if curr_match is not None and next_match is not None:
                if next_match.start() < curr_match.start():
                    curr_match = None

            if curr_match is not None:
                row.append(re.sub("\\n", " ", code[ind: ind + curr_match.start()]))
                ind += curr_match.end() + 1
                continue
            else:
                curr_match = re.search("\\\\\\\\", code[ind:])
                if curr_match is not None:
                    row.append(re.sub("\\n", " ", code[ind: ind + curr_match.start()]))
                    # update index in the outer while loop
                    j += ind + curr_match.end() + 1
                else:
                    curr_match = re.search(end_tabular_env, code[ind:])
                    if curr_match is None:
                        LOGGER.debug(f"Warning: failed to find an ampersand or \\\\\\\\ in {code[ind:]}")
                    j = len(contents)
                break
        rows.append(row)

    output = "\n"
    first_row = rows[0]
    output += "|"
    output += "|".join(first_row)
    output += "|\n"

    for elem in first_row:
        output += "|"
        output += "-"*len(elem)
    output += "|\n"

    for row in rows[1:]:
        output += "|"
        output += "|".join(row)
        output += "|\n"
    return output


def repl_align_topbot(match):
    code = match.group(1)
    return "\n\\begin{align}" + code + "\\end{align}\n"


def find_image_start_boundary(img_data):
    ind = 0
    while ind < len(img_data):
        row = img_data[ind]
        found = False
        for col in row:
            if col < 255:
                found = True
                break
        if found:
            break
        ind += 1
    return ind

def find_image_end_boundary(img_data):
    ind = len(img_data) - 1
    while ind > 0:
        row = img_data[ind]
        found = False
        for col in row:
            if col < 255:
                found = True
                break
        if found:
            break
        ind -= 1
    return ind


def find_balanced_delimeters(code: str, start_delimeter: str, end_delimeter: str):
    """ Finds the balanced start and end indices in the code string given a start delimeter 
    and end delimeter. 
    """

    ind = 0 
    num_seen = 0
    start = 0
    balanced_matches: list[tuple[int, int]] = []
    while ind < len(code):
        if code[ind : ind + len(start_delimeter)] == start_delimeter:
            LOGGER.debug(f"found start delimeter at {ind=}")
            if num_seen == 0: 
                start = ind
            num_seen += 1
            ind += len(start_delimeter)

        elif code[ind : ind + len(end_delimeter)] == end_delimeter:
            # only care about end delimieter if we've already seen a start delimeter
            if num_seen > 0:
                LOGGER.debug(f"found end delimeter at {ind=}")
                if num_seen == 1:
                    end = ind + len(end_delimeter) - 1
                    balanced_matches.append((start, end))
                num_seen -= 1
                ind += len(end_delimeter)
        ind += 1
    return balanced_matches




class Latex2Md:
    def __init__(self, tex_file: Path, markdown_dest: Path, page_name: str, create_dirs: bool = False):
        self.tex_file = tex_file
        self.markdown_dest = markdown_dest
        self.page_name = page_name
        self.pdf_dir = Path(f"docs/pdf/{markdown_dest.name}")
        self.png_dir = Path(f"docs/png/{markdown_dest.name}")
        self.tikz_dir = Path(f"docs/tikz/{markdown_dest.name}")
        self.num_figures = 0

        if create_dirs is True:
            self.pdf_dir.mkdir(exist_ok=True)
            self.png_dir.mkdir(exist_ok=True)
            self.tikz_dir.mkdir(exist_ok=True)

    def write_tikz_to_disk(self,raw_tikz_code: str, chapter_num:int, section_num:int, figure_num:int):
        with open(f"{self.tikz_dir}/chapter_{chapter_num}/tikz_code_{section_num}_{figure_num}.tex", "w") as f:
            f.write(raw_tikz_code)

    def repl_center_env(self, match, chapter, section):
        code = match.group(0)
        tikz_code_match = re.search(tikz_stmt, code)
        tikz_cd_code_match = re.search(tikz_cd_stmt, code)
        if tikz_code_match is None and tikz_cd_code_match is None:
            LOGGER.debug(f"Found a center environment but it contains no tikz code, returning {code=}")
            return code, False
        self.write_tikz_to_disk(code, chapter, section, self.num_figures)
        img_url = f"\n<img src=\"../../../png/{self.markdown_dest.name}/chapter_{chapter}/tikz_code_{section}_{self.num_figures}.png\" width=\"99%\" style=\"display: block; margin-left: auto; margin-right: auto;\"/>\n"
        self.num_figures += 1
        return img_url, True
    
    def repl_include_graphics(self, code, chapter):
        # parse the width option, if any
        img_options = find_balanced_delimeters(code, "[", "]")
        # parse the relative png path
        img_path_inds = find_balanced_delimeters(code, "{", "}")
        if len(img_path_inds) == 0:
            LOGGER.error(f"Failed to find any img path associated with {code=}, returning original code")
            return code

        if len(img_path_inds) > 1:
            LOGGER.error(f"Found multiple img paths associated with {code=}, using the first match")
            return code

        start = img_path_inds[0][0] + 1  # don't include { 
        end = img_path_inds[0][1]  # don't include } 
        rel_img_name = code[start: end]

        img_path = (self.tex_file.parent / rel_img_name).resolve()
        if not img_path.exists():
            LOGGER.error(f"{self.tex_file.parent} / {rel_img_name}")
            LOGGER.error(f"Failed to find image with destination {img_path=}")
            return code
        img_name = img_path.name  

        # copy the png over to our png directory
        img_dest = f"docs/png/{self.markdown_dest.name}/chapter_{chapter}/{img_name}"
        LOGGER.info(f"Copying {img_path} to {img_dest}")
        shutil.copy(img_path, img_dest)

        # return the img url 
        img_url = f"\n<img src=\"../../../png/{self.markdown_dest.name}/chapter_{chapter}/{img_name}\" width=\"99%\" style=\"display: block; margin-left: auto; margin-right: auto;\"/>\n"
        return img_url
    
    def repl_all_include_graphics(self, code, chapter):
        begin = "\\includegraphics"
        end = "}"

        balanced_delimeters = find_balanced_delimeters(code, begin, end)

        new_code = code
        offset = 0
        for start, end in balanced_delimeters:
            # Associate tikz_code on disk with img_url eventually holding png of tikz code 
            include_graphics_code = new_code[start + offset: end + offset + 1]
            repl = self.repl_include_graphics(include_graphics_code, chapter)

            # Replace tikz code with img tag
            new_code = new_code[:start + offset] + repl + new_code[end + 1 + offset:]
            offset += len(repl) - (end - start + 1)

        return new_code
    

    def repl_tikzpicture(self, code, chapter, section):
        begin_tikzpicture= "\\begin{tikzpicture}"
        end_tikzpicture= "\\end{tikzpicture}"

        balanced_delimeters = find_balanced_delimeters(code, begin_tikzpicture, end_tikzpicture)

        new_code = code
        offset = 0
        for start, end in balanced_delimeters:
            # Associate tikz_code on disk with img_url eventually holding png of tikz code 
            self.write_tikz_to_disk(new_code[start + offset: end + offset + 1], chapter, section, self.num_figures)
            img_url = f"\n<img src=\"../../../png/{self.markdown_dest.name}/chapter_{chapter}/tikz_code_{section}_{self.num_figures}.png\" width=\"99%\" style=\"display: block; margin-left: auto; margin-right: auto;\"/>\n"

            # Replace tikz code with img tag
            new_code = new_code[:start + offset] + img_url + new_code[end + 1 + offset:]
            offset += len(img_url) - (end - start + 1)

            self.num_figures += 1
        return new_code
    
    def repl_tikzcd(self, code, chapter, section):
        begin_tikzcd = "\\begin{tikzcd}"
        end_tikzcd = "\\end{tikzcd}"

        balanced_delimeters = find_balanced_delimeters(code, begin_tikzcd, end_tikzcd)

        new_code = code
        offset = 0
        for start, end in balanced_delimeters:
            # Associate tikz_code on disk with img_url eventually holding png of tikz code 
            self.write_tikz_to_disk(new_code[start + offset: end + offset + 1], chapter, section, self.num_figures)
            img_url = f"\n<img src=\"../../../png/{self.markdown_dest.name}/chapter_{chapter}/tikz_code_{section}_{self.num_figures}.png\" width=\"99%\" style=\"display: block; margin-left: auto; margin-right: auto;\"/>\n"

            # Replace tikz code with img tag
            new_code = new_code[:start + offset] + img_url + new_code[end + 1 + offset:]
            offset += len(img_url) - (end - start + 1)

            self.num_figures += 1
        return new_code

    def repl_surround(self, content, start_delimeter, end_delimeter, new_start_delimeter, new_end_delimeter):
        """ Finds and replaces the start, and end delimeters with new delimeters. 
        E.g. repl_surround("\\textbf{Meowmix}", "\\textbf{", "}", "*", "*") = *Meowmix*
        """

        balanced_delimeters = find_balanced_delimeters(content, start_delimeter, end_delimeter)

        new_code = content
        offset = 0
        for start, end in balanced_delimeters:
            captured_content = new_code[start + offset + len(start_delimeter): end + 1 + offset - len(end_delimeter)]
            replacement = f"{new_start_delimeter}{captured_content}{new_end_delimeter}"
            LOGGER.debug(f"replacing {captured_content=} with {replacement=}")

            new_code = new_code[:start + offset] + replacement + new_code[end + 1 + offset:]
            offset += len(replacement) - (end - start + 1)

        return new_code

    def extract_itemize_items(self, code):
        # look for nested itemize or description environments
        begin_itemize = "\\begin{itemize}"
        end_itemize = "\\end{itemize}"
        begin_description = "\\begin{description}"
        end_description = "\\end{description}"
        
        # chop off \begin{itemize}, \end{itemize}
        code = code.strip()
        if code.startswith(begin_itemize):
            code = code[len(begin_itemize):]
        elif code.startswith(begin_description):
            code = code[len(begin_description):]

        if code.endswith(end_itemize):
            code = code[:len(code) - len(end_itemize)]
        elif code.endswith(end_description):
            code = code[:len(code) - len(end_description)]

        print("running")
        list_items = [] 
        ind = 0
        while ind < len(code):
            # find the next \item and \begin{itemize} match; see which is first
            next_item = re.search("\\\\item", code[ind:])
            next_begin_itemize = re.search("\\\\begin{itemize}|\\\\begin{description}", code[ind:])

            # if \begin{itemize} is sooner
            if next_begin_itemize is not None and (next_begin_itemize.start() < next_item.start()):
                if next_begin_itemize.group(0) == begin_itemize:
                    balanced_itemize_envs = find_balanced_delimeters(code[ind:], begin_itemize, end_itemize)
                else:
                    balanced_itemize_envs = find_balanced_delimeters(code[ind:], begin_description, end_description)
                print("XXXXX", code)
                start, end = balanced_itemize_envs[0]
                list_items.append(self.extract_itemize_items(code[ind + start: ind + end +1]))
                ind += end + 1
            # if \item is sooner
            else:
                if next_item is None:
                    break
                start = ind + next_item.end()
                # check for \item[option] content
                item_content = ""
                if (text := re.match("\\[(.*?)\\]", code[start:])):
                    item_content += f"**{text.group(1)}** " # terminal space is important
                    start += text.end()

                # determine how far ahead to look
                next_item = re.search("\\\\item", code[start:])
                next_begin_itemize = re.search("\\\\begin{itemize}|\\\\begin{description}", code[start:])
                if next_item is None and next_begin_itemize is None:
                    end = len(code)
                elif next_begin_itemize is not None and (next_begin_itemize.start() < next_item.start()):
                    end = start + next_begin_itemize.start()
                else:
                    end = start + next_item.start()

                item_content += code[start:end].strip()
                if len(item_content) > 0:
                    LOGGER.debug(f"Adding item {item_content=} with {start=} {end=}")
                    list_items.append(item_content.strip())
                ind = end
        return list_items
            
    def format_list_items_as_markdown(self,list_items):
        output = ""
        for item in list_items:
            if isinstance(item, str):
                output += f"* {item}\n"
            else:
                sublist = self.format_list_items_as_markdown(item)
                output += textwrap.indent(sublist, prefix="    ")
        return output

    def repl_itemize_env(self, code):
        items = self.extract_itemize_items(code)
        markdown_list = self.format_list_items_as_markdown(items)
        markdown_list = f"\n{markdown_list}"
        return markdown_list

    def repl_all_itemize_envs(self, code):
        begin_itemize = "\\begin{itemize}"
        end_itemize = "\\end{itemize}"
        begin_description = "\\begin{description}"
        end_description = "\\end{description}"

        new_code = code
        offset = 0
        ind = 0
        while True:
            next_itemize_env = re.search("\\\\begin{itemize}", new_code[ind:])
            next_description_env = re.search("\\\\begin{description}", new_code[ind:])
            LOGGER.info(f"{next_itemize_env=} {next_description_env=}")
            if next_itemize_env is not None and next_description_env is not None:
                if next_itemize_env.start() < next_description_env.start():
                    balanced_delimeters = find_balanced_delimeters(new_code[ind:], begin_itemize, end_itemize)
                else:
                    balanced_delimeters = find_balanced_delimeters(new_code[ind:], begin_description, end_description)

            elif next_itemize_env is None and next_description_env is not None:
                balanced_delimeters = find_balanced_delimeters(new_code[ind:], begin_description, end_description)
                print(balanced_delimeters)

            elif next_itemize_env is not None and next_description_env is None:
                balanced_delimeters = find_balanced_delimeters(new_code[ind:], begin_itemize, end_itemize)

            else:
                break

            start, end = balanced_delimeters[0] 
            start += ind
            end += ind
            # Associate tikz_code on disk with img_url eventually holding png of tikz code 
            itemize_code = new_code[start : end + 1]
            print("YYY", itemize_code)
            repl = self.repl_itemize_env(itemize_code)

            # Replace itemize code 
            new_code = new_code[:start] + repl + new_code[end + 1:]
            # ind = end + 1 + offset
            ind = end + 1 + offset
            offset += len(repl) - (end - start + 1)

        return new_code
    

    def clean_code(self, code: str, chapter:int, section: int) -> str:
        LOGGER.info(f"doing {chapter=} {section=}")
        # remove comments
        code = re.sub(tex_comment, "", code)
        new_code = code

        # de indent everything
        final_code = ""
        for line in new_code.split("\n"):
            final_code += re.sub(indent_space, "", line)
            final_code += "\n"
        new_code = final_code

        ind = 0
        center_env_match = re.search(center_env, new_code)
        while center_env_match is not None:
            i = ind + center_env_match.start()
            replaced_code, _ = self.repl_center_env(center_env_match, chapter, section)
            j = ind + center_env_match.end() + 1

            # look for next instance of \begin{center} before updating new_code
            center_env_match = re.search(center_env, new_code[j:])
            # update new code
            new_code = new_code[:i] + replaced_code + new_code[j:]
            ind = i + len(replaced_code)
        # replace all of the tikzpictures 
        new_code = self.repl_tikzpicture(new_code, chapter, section)
        new_code = self.repl_tikzcd(new_code, chapter, section)
        # replace all of the tikzcd
        self.num_figures = 0

        # get rid of labels for now. alg might be chopping off leading backslash after label.
        new_code = re.sub(label_stmt, "", new_code)
        new_code = re.sub(label_stmt2, "", new_code)

        # amsthm env
        new_code = re.sub(definition_stmt, repl_def, new_code)
        new_code = re.sub(proposition_stmt, repl_prop, new_code)
        new_code = re.sub(lemma_stmt, repl_lemma, new_code)
        new_code = re.sub(corollary_stmt, repl_corollary, new_code)
        new_code = re.sub(theorem_stmt, repl_theorem, new_code)
        new_code = re.sub(thm_env, repl_theorem, new_code)
        new_code = re.sub(example_stmt, repl_example, new_code)
        new_code = re.sub(proof_stmt, repl_proof, new_code)
        new_code = re.sub(remark_env, repl_remark, new_code)

        # replace latex bolding with ** syntax
        new_code = self.repl_surround(new_code, "\\textbf{", "}", "**", "**")
        # replace latex italics with * syntax
        new_code = self.repl_surround(new_code, "\\emph{", "}", "*", "*")
        new_code = self.repl_surround(new_code, "\\textit{", "}", "*", "*")
        # replace latex quotes `` '' with " syntax
        new_code = self.repl_surround(new_code, "``", "''", "\"", "\"")
        # set display math on newlines
        new_code = re.sub(display_math, "\n\\1\n", new_code)
        # set align_topbot environments on newlines
        new_code = re.sub(align_topbot_stmt, repl_align_topbot, new_code)
        # set align environments on newlines
        new_code = re.sub(align_stmt, "\n\\1\n", new_code)
        # set align* environments on newlines 
        new_code = re.sub(align_star_stmt, "\n\\1\n", new_code)
        # set gather environments on newlines
        new_code = re.sub(gather_env, "\n\\1\n", new_code)
        # set gather environments on newlines
        new_code = re.sub(gather_star_env, "\n\\1\n", new_code)
        # replace itemize environments
        # new_code = re.sub(itemize_env, repl_itemize_env, new_code)
        new_code = self.repl_all_itemize_envs(new_code)
        # replace description environments
        # new_code = re.sub(description_env, repl_description_env, new_code)
        # remove \newpage
        new_code = re.sub(newpage_cmd, "", new_code)
        # replace tabular environments with markdown tables
        new_code = re.sub(tabular_env, repl_tabular_env, new_code)
        # replace minipage environment

        # replace includegraphics
        new_code = self.repl_all_include_graphics(new_code, chapter)
        # change \[ ... \] delimeters to $$ $$
        new_code = self.repl_surround(new_code, "\\[", "\\]", "$$", "$$")

        # log unrecognized environments
        start_env = re.compile("\\\\begin{([\S\s]+?)}")
        for env_match in re.finditer(start_env, new_code):
            env = env_match.group(1)
            if env not in KNOWN_ENVS:
                LOGGER.warning(f"Unrecognized environment {env_match.group(1)} remains in {chapter=} {section=}")

        return new_code


    def convert_pdf_to_png(self, pdf_file):
        pdf_file = Path(pdf_file)
        chapter_dir = pdf_file.parts[-2]
        png_filename = Path(pdf_file.parts[-1]).with_suffix(".png")
        png_destination = Path(f"{self.png_dir}/{chapter_dir}/{png_filename}")
        if CACHE_PNGS:
            if png_destination.exists():
                return

        pdf_fp = str(pdf_file.resolve())
        page_pngs = convert_from_path(pdf_fp)

        # Create the png of the pdf
        total = len(page_pngs)
        ind = 0
        if total > 1:
            LOGGER.error(f"WARNING! {pdf_file=} has more than two pages, expected only one. Going to use"
                " the last page. ")
            ind = len(page_pngs) - 1

        page_pngs = list(page_pngs)
        image = page_pngs[ind]
        LOGGER.debug(f"Converting page {ind}/{total}")
        # easier to find boundaries of a grayscale image
        grayscale_image = image.convert("L")
        img_data = np.asarray(grayscale_image)
        y_0 = find_image_start_boundary(img_data)
        y_1 = find_image_end_boundary(img_data)
        x_0 = find_image_start_boundary(img_data.T)
        x_1 = find_image_end_boundary(img_data.T)
        horizontal_len = len(img_data.T)
        vertical_len = len(img_data.T)
        # Zoom in the picture 
        x_0 = int(min(.20*horizontal_len, x_0))
        x_1 = int(max(.80*horizontal_len, x_1))
        # Add vertical whitespace padding
        y_0 = int(max(0, y_0 - 0.02*vertical_len))
        y_1 = int(min(vertical_len, y_1 + 0.02*vertical_len))
        
        true_img = image.convert("RGB")
        img_data = np.asarray(true_img)
        cropped_img_data= img_data[y_0:y_1, x_0:x_1]
        cropped_img = Image.fromarray(np.uint8(cropped_img_data))
        cropped_img.save(str(png_destination), "PNG")

            
        # png -> np.array
        # numpy.asarray(PIL.Image.open('test.jpg'))

        # np.array -> img
        # Image.fromarray(np.uint8(img_data))

    def latex_to_markdown(self, tex_file: Path, markdown_dir: Path):
        with open(tex_file) as f:
            contents = f.read()

        begin_match = re.search(begin_document, contents)
        end_match = re.search(end_document, contents)

        tex_code = contents[begin_match.end() : end_match.start()]

        chapters = {}
        for chapter_match in re.finditer(chapter, tex_code):
            chapter_start = chapter_match.end() + 1
            chapter_name = chapter_match.group(1)
            next_chapter_match = re.search(chapter, tex_code[chapter_start:])
            if next_chapter_match is None:
                chapter_code = tex_code[chapter_start:]
            else:
                chapter_code = tex_code[chapter_start:chapter_start + next_chapter_match.start()]

            sections = {}
            chapters[chapter_name] = sections
            for section_match in re.finditer(section, chapter_code):
                section_start = section_match.end() + 1
                section_name = section_match.group(1)
                next_section_match = re.search(section, chapter_code[section_start:])
                if next_section_match is None:
                    section_code = chapter_code[section_start:]
                else:
                    section_code = chapter_code[section_start:section_start + next_section_match.start()]

                sections[section_name] = section_code

        chapter_num = 1
        for chapter_name, sections in chapters.items():
            chapter_dir = Path(f"docs/{markdown_dir.name}/{chapter_name}")
            chapter_pdf_dir = Path(f"{self.pdf_dir}/chapter_{chapter_num}")
            chapter_png_dir = Path(f"{self.png_dir}/chapter_{chapter_num}")
            chapter_tikz_dir = Path(f"{self.tikz_dir}/chapter_{chapter_num}")
            chapter_dir.mkdir(exist_ok=True)
            chapter_pdf_dir.mkdir(exist_ok=True)
            chapter_png_dir.mkdir(exist_ok=True)
            chapter_tikz_dir.mkdir(exist_ok=True)

            section_num = 1
            for section_name, code in sections.items():
                with open(chapter_dir / f"{section_name}.md", "w") as f:
                    code = self.clean_code(code, chapter_num, section_num)
                    f.write(EXTRA_MKDOCS_CSS)
                    f.write(f"#{chapter_num}.{section_num}. {section_name}\n")
                    f.write(code)
                    f.write("\n<script src=\"../../mathjax_helper.js\"></script>")
                section_num += 1
            chapter_num += 1
        return chapters 


    def create_mkdocs_yaml_nav(self, chapters):
        page_nav = {self.page_name: [{"Home": f"{self.markdown_dest.name}/index.md"}]}
        chapter_num = 1
        for chapter in chapters:
            nav_chapter = {chapter: []}
            sections = chapters[chapter]
            section_num = 1
            for section_name in sections:
                nav_chapter[chapter].append({f"{chapter_num}.{section_num} {section_name}": f"{self.markdown_dest.name}/{chapter}/{section_name}.md"})
                section_num += 1

            page_nav[self.page_name].append(nav_chapter)
            chapter_num += 1
        
        with open("mkdocs.yml") as f:
            yaml_contents = f.read()

        yaml_data = yaml.load(yaml_contents, Loader=yaml.Loader)
        replaced = False
        for ind, elem in enumerate(yaml_data["nav"]):
            if next(iter(elem.keys())) == self.page_name:
                yaml_data["nav"][ind] = page_nav
                replaced = True
        if not replaced:
            yaml_data["nav"].append(page_nav)
            
        yaml_output = yaml.dump(yaml_data, Dumper=yaml.Dumper)

        with open("mkdocs.yml", "w") as f:
            f.write(yaml_output)

    def compile_tikz_blocks(self, tikz_code_path):
        chapter_dir = tikz_code_path.parts[-2]
        pdf_filename = Path(tikz_code_path.parts[-1]).with_suffix(".pdf")
        pdf_destination = Path(f"{self.pdf_dir}/{chapter_dir}/{pdf_filename}")
        if CACHE_TIKZ:
            if pdf_destination.exists():
                return

        with open(tikz_code_path) as f:
            raw_tikz_code = f.read()

        with tempfile.TemporaryDirectory() as tmpdirname:
            if (self.markdown_dest / "tikz_template.tex").exists():
                with open(self.markdown_dest / "tikz_template.tex") as f:
                    tikz_code = f.read()
            else:
                with open("tikz_template.tex") as f:
                    tikz_code = f.read()

            tikz_code = re.sub("fillme", lambda x: raw_tikz_code, tikz_code)

            with open(f"{tmpdirname}/tikz_code.tex", "w") as f:
                f.write(tikz_code)

            proc_res = subprocess.run(
                f"latexmk -pdf -quiet -output-directory={tmpdirname} {tmpdirname}/tikz_code.tex",
                shell=True,
            )
            if proc_res.returncode != 0:
                LOGGER.error(f"Experience an error while processing {tikz_code_path=}")

            pdf_file = Path(f"{tmpdirname}/tikz_code.pdf")
            pdf_file.replace(pdf_destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="latex2md")
    parser.add_argument("--tex-file", required=True)
    parser.add_argument("--markdown-dest", required=True)
    parser.add_argument("--page-name", required=True)
    parser.add_argument("--compile-tikz", action='store_true', default=False)
    parser.add_argument("--generate-pngs", action='store_true', default=False)
    parser.add_argument("--cache-tikz", action='store_true', default=False)
    parser.add_argument("--cache-pngs", action='store_true', default=False)
    parser.add_argument("--create-mkdocs-nav", action='store_true', default=False)
    args = parser.parse_args()

    tex_file = Path(args.tex_file)
    markdown_dest = Path(args.markdown_dest)
    page_name = args.page_name
    COMPILE_TIKZ = args.compile_tikz
    GENERATE_PNGS = args.generate_pngs
    CACHE_TIKZ = args.cache_tikz
    CACHE_PNGS = args.cache_pngs
    CREATE_MKDOCS_NAV = args.create_mkdocs_nav

    if not tex_file.exists():
        LOGGER.error(f"Error: {args.tex_file} does not exist")
        sys.exit()

    if not markdown_dest.exists():
        LOGGER.error(f"Error: {args.markdown_dest} does not exist")
        sys.exit()

    handler = Latex2Md(
        tex_file=tex_file,
        markdown_dest=markdown_dest,
        page_name=page_name,
        create_dirs=True,
    )    

    chapters: dict = handler.latex_to_markdown(tex_file, markdown_dir=markdown_dest)
    if CREATE_MKDOCS_NAV:
        handler.create_mkdocs_yaml_nav(chapters)
    
    if COMPILE_TIKZ:
        # compile tikz drawings to pdfs
        for tikz_file in Path(".").glob(f"docs/tikz/{markdown_dest.name}/**/*.tex"):
            LOGGER.info(tikz_file)
            handler.compile_tikz_blocks(tikz_file)

    if GENERATE_PNGS:
        # convert tikz pdfs to pngs
        for pdf_file in Path(".").glob(f"docs/pdf/{markdown_dest.name}/**/*.pdf"):
            LOGGER.info(pdf_file)
            handler.convert_pdf_to_png(str(pdf_file))

