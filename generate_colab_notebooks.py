#!/usr/bin/env python3
"""
Generate Google Colab notebooks from Marimo notebooks with proper formatting and badges.
"""
import json
import re
from pathlib import Path

GITHUB_REPO = "harunpirim/IME775"
GITHUB_BRANCH = "main"

def extract_marimo_content(marimo_file):
    """Extract cells from Marimo notebook."""
    with open(marimo_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the file to extract cells
    cells = []
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Skip import marimo and app definition
        if 'import marimo' in line or 'app = marimo.App' in line or '__generated_with' in line:
            i += 1
            continue
        
        # Check for @app.cell decorator
        if '@app.cell' in line:
            cell_type = 'markdown' if 'hide_code=True' in line else 'code'
            i += 1
            
            # Skip function definition line
            if i < len(lines) and lines[i].strip().startswith('def '):
                i += 1
            
            # Collect cell content
            cell_lines = []
            base_indent = None
            
            while i < len(lines):
                current_line = lines[i]
                
                # Stop at next decorator or end of file
                if current_line.strip().startswith('@app.cell'):
                    break
                
                # Stop at blank line followed by another function
                if (current_line.strip() == '' and 
                    i + 1 < len(lines) and 
                    (lines[i + 1].strip() == '' or lines[i + 1].strip().startswith('@app.cell'))):
                    break
                
                # Skip return statements
                if current_line.strip().startswith('return'):
                    i += 1
                    break
                
                # Process line
                if current_line.strip():
                    # Detect base indentation from first non-empty line
                    if base_indent is None:
                        base_indent = len(current_line) - len(current_line.lstrip())
                    
                    # Remove base indentation
                    if current_line.startswith(' ' * base_indent):
                        dedented = current_line[base_indent:]
                        cell_lines.append(dedented)
                    else:
                        cell_lines.append(current_line.lstrip())
                
                i += 1
            
            # Process cell content
            cell_text = '\n'.join(cell_lines).strip()
            
            # Check if it's markdown (contains mo.md)
            if 'mo.md(' in cell_text or cell_type == 'markdown':
                cell_type = 'markdown'
                # Extract markdown content from mo.md(r"""...""")
                match = re.search(r'mo\.md\s*\(\s*r?"""(.*?)"""\s*\)', cell_text, re.DOTALL)
                if match:
                    cell_text = match.group(1).strip()
                else:
                    # Try single quotes
                    match = re.search(r"mo\.md\s*\(\s*r?'''(.*?)'''\s*\)", cell_text, re.DOTALL)
                    if match:
                        cell_text = match.group(1).strip()
            
            # Skip empty cells and marimo imports
            if cell_text and 'import marimo' not in cell_text:
                cells.append({
                    'type': cell_type,
                    'content': cell_text
                })
            
            continue
        
        i += 1
    
    return cells

def create_colab_badge_cell(notebook_path):
    """Create a Colab badge cell."""
    # Convert local path to GitHub path
    github_path = str(notebook_path).replace('/Users/harunpirim/Documents/GitHub/IME775/', '')
    colab_url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{github_path}"
    
    badge_markdown = f"""<a href="{colab_url}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---
"""
    
    return {
        "cell_type": "markdown",
        "metadata": {
            "id": "colab_badge"
        },
        "source": badge_markdown.split('\n')
    }

def create_colab_notebook(cells, notebook_path, title="Notebook"):
    """Create Jupyter/Colab notebook JSON structure with badge."""
    notebook_cells = []
    
    # Add Colab badge as first cell
    notebook_cells.append(create_colab_badge_cell(notebook_path))
    
    # Add all other cells
    for idx, cell in enumerate(cells):
        if cell['type'] == 'markdown':
            notebook_cells.append({
                "cell_type": "markdown",
                "metadata": {
                    "id": f"markdown_{idx}"
                },
                "source": [line + '\n' for line in cell['content'].split('\n')]
            })
        else:
            # Code cell - preserve line breaks
            source_lines = cell['content'].split('\n')
            # Add newline to each line except the last one
            formatted_lines = [line + '\n' for line in source_lines[:-1]]
            if source_lines[-1]:  # Add last line without newline
                formatted_lines.append(source_lines[-1])
            
            notebook_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {
                    "id": f"code_{idx}"
                },
                "outputs": [],
                "source": formatted_lines
            })
    
    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "colab": {
                "name": notebook_path.stem,
                "provenance": [],
                "toc_visible": True,
                "collapsed_sections": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    return notebook

def generate_notebook(marimo_path, output_path):
    """Generate a Colab notebook from a Marimo notebook."""
    print(f"Generating {output_path.name}...", end=' ')
    
    try:
        cells = extract_marimo_content(marimo_path)
        
        if not cells:
            print("⚠ No cells extracted")
            return False
        
        notebook = create_colab_notebook(cells, output_path, title=marimo_path.stem)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print(f"✓ ({len(cells) + 1} cells with badge)")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Generate all Colab notebooks from Marimo notebooks."""
    base_dir = Path(__file__).parent
    
    # Find all Marimo notebooks
    notebooks = sorted(base_dir.glob("week-*/notebook*.py"))
    
    print(f"Generating {len(notebooks)} Colab notebooks with badges\n")
    print("=" * 60)
    
    generated = 0
    for notebook_path in notebooks:
        # Create output path
        week_dir = notebook_path.parent
        notebook_name = notebook_path.stem
        output_path = week_dir / f"{notebook_name}.ipynb"
        
        if generate_notebook(notebook_path, output_path):
            generated += 1
    
    print("=" * 60)
    print(f"\n✓ Successfully generated {generated}/{len(notebooks)} notebooks")
    print(f"\nEach notebook includes:")
    print(f"  • Google Colab badge linking to GitHub")
    print(f"  • Proper markdown formatting")
    print(f"  • GPU acceleration enabled")
    print(f"  • Table of contents enabled")

if __name__ == "__main__":
    main()
