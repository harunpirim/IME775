#!/usr/bin/env python3
"""
Convert Marimo notebooks to Google Colab (Jupyter) format.
"""
import json
import re
import ast
from pathlib import Path

def extract_marimo_content(marimo_file):
    """Extract cells from Marimo notebook."""
    with open(marimo_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the file to extract cells
    cells = []
    lines = content.split('\n')
    current_cell = []
    in_cell = False
    cell_type = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for @app.cell decorator
        if '@app.cell' in line:
            in_cell = True
            current_cell = []
            # Check if it's markdown (hide_code=True or mo.md inside)
            cell_type = 'markdown' if 'hide_code=True' in line else 'code'
            i += 1
            continue
        
        # Check for function definition
        if in_cell and line.strip().startswith('def '):
            # Skip function definition line
            i += 1
            # Get function body
            indent_level = len(line) - len(line.lstrip())
            while i < len(lines):
                next_line = lines[i]
                # Check if we've reached the end of function
                if next_line.strip() and not next_line.startswith(' ' * (indent_level + 1)) and not next_line.startswith('\t'):
                    if next_line.strip().startswith('return') or next_line.strip() == '':
                        # Include return statement
                        if next_line.strip().startswith('return'):
                            current_cell.append(next_line)
                        break
                
                # Skip return statements at end
                if next_line.strip().startswith('return') and i < len(lines) - 2:
                    if lines[i+1].strip() == '' and lines[i+2].strip() == '':
                        break
                
                # Add line to cell
                if next_line.strip() and not next_line.strip().startswith('def '):
                    # Remove function-level indentation
                    if next_line.startswith(' ' * (indent_level + 4)):
                        current_cell.append(next_line[(indent_level + 4):])
                    elif next_line.startswith(' ' * (indent_level + 1)):
                        current_cell.append(next_line[(indent_level + 1):])
                    else:
                        current_cell.append(next_line)
                i += 1
            
            # Process cell content
            cell_text = '\n'.join(current_cell).strip()
            
            # Check if it's actually markdown (contains mo.md)
            if 'mo.md(' in cell_text:
                cell_type = 'markdown'
                # Extract markdown content
                match = re.search(r'mo\.md\(r?"""(.*?)"""\)', cell_text, re.DOTALL)
                if match:
                    cell_text = match.group(1).strip()
                else:
                    # Try single quotes
                    match = re.search(r"mo\.md\(r?'''(.*?)'''\)", cell_text, re.DOTALL)
                    if match:
                        cell_text = match.group(1).strip()
            
            if cell_text:
                cells.append({
                    'type': cell_type,
                    'content': cell_text
                })
            
            current_cell = []
            in_cell = False
            cell_type = None
            continue
        
        i += 1
    
    return cells

def create_colab_notebook(cells, title="Notebook"):
    """Create Jupyter/Colab notebook JSON structure."""
    notebook_cells = []
    
    for cell in cells:
        if cell['type'] == 'markdown':
            notebook_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": cell['content'].split('\n')
            })
        else:
            # Code cell
            source_lines = cell['content'].split('\n')
            notebook_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source_lines
            })
    
    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    return notebook

def convert_notebook(marimo_path, output_path):
    """Convert a single Marimo notebook to Colab format."""
    print(f"Converting {marimo_path}...")
    
    try:
        cells = extract_marimo_content(marimo_path)
        notebook = create_colab_notebook(cells)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"  ✓ Created {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ Error converting {marimo_path}: {e}")
        return False

def main():
    """Convert all Marimo notebooks to Colab format."""
    base_dir = Path(__file__).parent
    
    # Find all Marimo notebooks
    notebooks = list(base_dir.glob("week-*/notebook*.py"))
    
    print(f"Found {len(notebooks)} notebooks to convert\n")
    
    converted = 0
    for notebook_path in sorted(notebooks):
        # Create output path
        week_dir = notebook_path.parent
        notebook_name = notebook_path.stem  # e.g., "notebook-math-dl" or "notebook"
        output_path = week_dir / f"{notebook_name}.ipynb"
        
        if convert_notebook(notebook_path, output_path):
            converted += 1
    
    print(f"\n✓ Successfully converted {converted}/{len(notebooks)} notebooks")

if __name__ == "__main__":
    main()
