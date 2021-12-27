#!/usr/bin/python3
"""Make static website/blog with Python."""
import os
import shutil
import re
import glob
import sys
import json
import datetime
import markdown
from pathlib import Path

# Special tags
def css_tag(css_file):
    """Create CSS tag given path relative to CSS folder."""
    return f'<link rel="stylesheet" type="text/css" href="/css/{css_file}">'

def src_tag(link):
    """Create script tag based on supplied https link"""
    return f'<script src="{link}"></script>'

def syntax_highlighting_tag():
    """Returns the CSS tag to perform syntax highlighting"""
    code_theme = '<link rel="stylesheet" type="text/css" href="/css/material-darker.css">' 
    syntax_highlighting = '<script src="/highlight/highlight.min.js"></script>\n<script>hljs.highlightAll();</script>'
    return code_theme + '\n\t' + syntax_highlighting

def preview_img_tag(img_path):
    """Returns an image tag images on the home page"""
    if img_path is None:
        return ''
    return f'<img class="preview-img" src="{img_path}"/>'

def html_to_md(content):
    return markdown.markdown(content, extensions=["fenced_code", "tables"])

def fread(filename):
    """Read file and close the file."""
    with open(filename, 'r') as f:
        return f.read()


def fwrite(filename, text):
    """Write content to file and close the file."""
    basedir = Path(filename).parent
    if not basedir.exists():
        basedir.mkdir(parents=True)

    with open(filename, 'w') as f:
        f.write(text)

def currently_ignoring(path):
    with open(".gitignore") as f:
        for line in f.readlines():
            if path == line.strip():
                return True
    return False


def log(msg, *args):
    """Log message with specified arguments."""
    sys.stderr.write(msg.format(*args) + '\n')


def truncate(text, words=25):
    """Remove tags and truncate text to the specified number of words."""
    return ' '.join(re.sub('(?s)<.*?>', ' ', text).split()[:words])


def read_headers(text):
    """Parse headers in text and yield (key, value, end-index) tuples."""
    for match in re.finditer(r'\s*<!--\s*(.+?)\s*:\s*(.+?)\s*-->\s*|.+', text):
        if not match.group(1):
            break
        yield match.group(1), match.group(2), match.end()

def rfc_2822_format(date_str):
    """Convert yyyy-mm-dd date string to RFC 2822 format date string."""
    d = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return d.strftime('%a, %d %b %Y %H:%M:%S +0000')


def read_content(filename):
    """Read content and metadata from file into a dictionary."""
    # Read file content.
    text = fread(filename)

    # Read metadata and save it in a dictionary.
    date_slug = os.path.basename(filename).split('.')[0]
    match = re.search(r'^(?:(\d\d\d\d-\d\d-\d\d)-)?(.+)$', date_slug)
    content = {
        'date': match.group(1) or '1970-01-01',
        'slug': match.group(2),
    }

    # Read headers.
    end = 0
    for key, val, end in read_headers(text):
        if key == 'css' or key == 'src':
            if content.get(key) is None:
                content[key] = [val]
            else:
                content[key].append(val)
        else:
            content[key] = val

    # Separate content from headers.
    text = text[end:]

    # Convert Markdown content to HTML.
    if filename.endswith(('.md', '.mkd', '.mkdn', '.mdown', '.markdown')):
        try:
            if _test == 'ImportError':
                raise ImportError('Error forced by test')
            text = html_to_md(text)
        except ImportError as e:
            log('WARNING: Cannot render Markdown in {}: {}', filename, str(e))

    # Update the dictionary with content and RFC 2822 date.
    content.update({
        'content': text,
        'rfc_2822_date': rfc_2822_format(content['date'])
    })

    return content


def render(template, **params):
    """Replace placeholders in template with values from params."""
    return re.sub(r'{{\s*([^}\s]+)\s*}}',
                  lambda match: str(params.get(match.group(1), match.group(0))),
                  template)


def make_pages(src_pattern, dst_parent, layout, **params):
    """Generate pages from page content."""
    items = []
    dst_parent = Path(f"{dst_parent}")

    for src_path in Path(".").glob(src_pattern):
        print(f"{src_pattern=} has {src_path=}")
        if currently_ignoring(str(src_path)):
            print(f"Ignoring {src_path}")
            continue
        if src_path.is_file():
            content = read_content(str(src_path))
        elif src_path.is_dir():
            for item in src_path.iterdir():
                if item.suffix == ".md":
                    content = read_content(str(item))
                else:  # Copy directory contents to destination
                    if item.is_dir():
                        dst = (dst_parent / src_path.name / item.name)
                        dst.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(item, dst, dirs_exist_ok=True)
                    else:
                        dst = (dst_parent / src_path.name)
                        dst.mkdir(parents=True, exist_ok=True)
                        shutil.copy(item, dst)

        page_params = dict(params, **content)

        # Assign a type to a post
        content['post_type'] = dst_parent.parts[-1]

        # Populate placeholders in content if content-rendering is enabled.
        if page_params.get('render') == 'yes':
            rendered_content = render(page_params['content'], **page_params)
            page_params['content'] = rendered_content
            content['content'] = rendered_content

        items.append(content)

        # Create CSS tags.
        css_tags = ''
        for css_file in page_params['css']:
            css_tags += css_tag(css_file) + '\n\t'
        if page_params['syntax_highlighting'] == 'on':
            css_tags += syntax_highlighting_tag() + '\n\t'
        page_params['css'] = css_tags
        
        # Create script tags.
        src_links = ''
        for link in page_params['src']:
            src_links += src_tag(link) + '\n\t'
        if page_params['mathjax'] == 'on':
            src_links += '\n' + fread('layout/mathjax.html')
        page_params['src'] = src_links
        

        # Create page.
        if src_path.is_dir():
            dst_path = dst_parent / src_path.stem / 'index.html'
        else:
            dst_path = dst_parent / "index.html"
        output = render(layout, **page_params)
        log('Rendering {} => {} ...', src_path, dst_path)
        fwrite(dst_path, output)

    return items


def make_list(posts, dst, list_layout, item_layout, **params):
    """Generate list page for a blog."""
    items = []
    for post in posts:
        item_params = dict(params, **post)
        item_params['summary'] = truncate(post['content'])
        item_params['image'] = preview_img_tag(post.get('preview_img'))
        item = render(item_layout, **item_params)
        items.append(item)

    # Create CSS tags.
    css_tags = ''
    for css_file in params['css']:
        css_tags += css_tag(css_file) + '\n\t'
    if params['syntax_highlighting'] == 'on':
        css_tags += syntax_highlighting_tag() + '\n\t'
    params['css'] = css_tags
    
    # Create script tags.
    src_links = ''
    for link in params['src']:
        src_links += src_tag(link) + '\n\t'
    if params['mathjax'] == 'on':
        src_links += '\n' + fread('layout/mathjax.html')
    params['src'] = src_links

    params['content'] = ''.join(items)
    dst_path = render(dst, **params)
    output = render(list_layout, **params)

    log('Rendering list => {} ...', dst_path)
    fwrite(dst_path, output)


def main():
    # Create a new _site directory from scratch.
    # Default parameters.
    params = {
        'base_path': '',
        'site_url': 'http://localhost:8000',
        'current_year': datetime.datetime.now().year,
        'css': [],
        'src': [],
        'syntax_highlighting': 'off',
        'mathjax': 'off',
    }

    # Load layouts.
    page_layout = fread('layout/page.html')
    post_layout = fread('layout/post.html')
    list_layout = fread('layout/list.html')
    item_layout = fread('layout/preview_item.html')

    # Combine layouts to form final layouts.
    post_layout = render(page_layout, content=post_layout)
    list_layout = render(page_layout, content=list_layout)

     # Create about page
    make_pages('content/about/about.md', 
               'about',
               page_layout, 
               **params)

    # Load associahedra app.
    associahedra_app = fread('layout/associahedra_app.html')

    # Create blogs.
    blog_posts = make_pages('content/math/*',
                            'math',
                            post_layout, 
                            blog='math',
                            associahedra_app=associahedra_app,
                            **params) 
    
    blog_posts += make_pages('content/cs/*',
                            'cs',
                            post_layout, 
                            blog='cs',
                            associahedra_app=associahedra_app,
                            **params)
    # Sort the blog posts
    blog_posts = sorted(blog_posts, key=lambda x: x['date'], reverse=True) 

    # Create page intro.
    intro = html_to_md(fread('content/index.md'))

    # Create blog list pages.
    make_list(blog_posts, 'index.html',
                list_layout,
                item_layout,
                intro=intro, 
                **params)


# Test parameter to be set temporarily by unit tests.
_test = None

def clear_metadata() -> int:
    """Remove metadata from photos."""
    photo_extensions = [".png", ".jpg", "jpeg"]
    website_dir = Path().resolve()

    assert website_dir.parts[-1] == "website"
    for file in website_dir.rglob("*"):
        if file.suffix in photo_extensions:
            subprocess.Popen(["exiftool", "-all=", f"{file}"])
            print(f"{file}")
    return 0


if __name__ == '__main__':
    main()
