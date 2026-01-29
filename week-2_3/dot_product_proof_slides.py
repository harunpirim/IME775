# /// script
# requires-python = "==3.12"
# dependencies = [
#     "manim==0.19.1",
#     "manim-slides==5.5.2",
#     "mohtml==0.1.11",
#     "moterm==0.1.0",
# ]
# ///

"""
Dot Product and Cosine Proof - Animated Slides
================================================
From "Math and Architectures of Deep Learning" - Appendix A

This marimo notebook creates an animated proof showing that:
    a · b = ||a|| ||b|| cos(θ)

Uses manim-slides for dynamic mathematical visualization.
"""

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="columns")

with app.setup:
    from manim import (
        Dot,
        Circle,
        Arrow,
        VGroup,
        Text,
        MathTex,
        Axes,
        FadeIn,
        FadeOut,
        Create,
        Write,
        Transform,
        ReplacementTransform,
        BLUE,
        RED,
        GREEN,
        WHITE,
        YELLOW,
        BLACK,
        ORIGIN,
        UP,
        DOWN,
        LEFT,
        RIGHT,
        Line,
        Arc,
        config,
        TexTemplate,
    )
    from manim_slides import Slide
    import numpy as np


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    # Check if manim is available
    try:
        import manim
        import manim_slides
        MANIM_AVAILABLE = True
    except ImportError:
        MANIM_AVAILABLE = False

    return Path, mo, MANIM_AVAILABLE


@app.cell
def _(mo):
    mo.md(r"""
    # Dot Product Proof: $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$

    This animated proof shows why the dot product equals the product of magnitudes
    times the cosine of the angle between vectors.

    **From "Math and Architectures of Deep Learning" - Appendix A**

    ---

    ## The Proof Steps

    We want to prove that for two vectors $\mathbf{a}$ and $\mathbf{b}$:

    $$\mathbf{a} \cdot \mathbf{b} = a_x b_x + a_y b_y = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$$

    where $\theta$ is the angle between the vectors.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 1: Express Vector Components

    For vector $\\mathbf{a}$ at angle $(\\theta + \\phi)$ from x-axis:

    $$a_x = \\|\\mathbf{a}\\| \\cos(\\theta + \\phi)$$

    $$a_y = \\|\\mathbf{a}\\| \\sin(\\theta + \\phi)$$

    For vector $\\mathbf{b}$ at angle $\\phi$ from x-axis:

    $$b_x = \\|\\mathbf{b}\\| \\cos(\\phi)$$

    $$b_y = \\|\\mathbf{b}\\| \\sin(\\phi)$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: Apply Angle Addition Formulas

    Using trigonometric identities:

    $$\cos(\theta + \phi) = \cos\phi \cos\theta - \sin\phi \sin\theta = \frac{a_x}{\|\mathbf{a}\|}$$

    $$\sin(\theta + \phi) = \sin\phi \cos\theta + \cos\phi \sin\theta = \frac{a_y}{\|\mathbf{a}\|}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3: Set Up the Linear System

    Substituting $\cos\phi = \frac{b_x}{\|\mathbf{b}\|}$ and $\sin\phi = \frac{b_y}{\|\mathbf{b}\|}$:

    $$\frac{b_x}{\|\mathbf{b}\|} \cos\theta - \frac{b_y}{\|\mathbf{b}\|} \sin\theta = \frac{a_x}{\|\mathbf{a}\|}$$

    $$\frac{b_y}{\|\mathbf{b}\|} \cos\theta + \frac{b_x}{\|\mathbf{b}\|} \sin\theta = \frac{a_y}{\|\mathbf{a}\|}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4: Matrix-Vector Form

    In matrix form:

    $$\frac{1}{\|\mathbf{b}\|} \begin{bmatrix} b_x & -b_y \\ b_y & b_x \end{bmatrix} \begin{bmatrix} \cos\theta \\ \sin\theta \end{bmatrix} = \frac{1}{\|\mathbf{a}\|} \begin{bmatrix} a_x \\ a_y \end{bmatrix}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5: Solve for cos(θ)

    Solving the system yields:

    $$\cos\theta = \frac{a_x b_x + a_y b_y}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

    $$\sin\theta = \frac{a_y b_x - a_x b_y}{\|\mathbf{a}\| \|\mathbf{b}\|}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Final Result ✓

    Rearranging the cosine equation:

    $$\boxed{\|\mathbf{a}\| \|\mathbf{b}\| \cos\theta = a_x b_x + a_y b_y = \mathbf{a} \cdot \mathbf{b}}$$

    **Q.E.D.** □
    """)
    return


@app.class_definition
class DotProductProof(Slide):
    """Animated proof that a·b = ||a|| ||b|| cos(θ)"""
    
    def construct(self):
        # Configure LaTeX to use article instead of standalone (avoids missing standalone.cls error)
        # Create a minimal template without babel to avoid compatibility issues
        tex_template = TexTemplate()
        tex_template.documentclass = r"\documentclass{article}"
        # Override preamble to remove babel dependency
        tex_template.preamble = r"""
\usepackage{amsmath}
\usepackage{amssymb}
"""
        # Keep DVI output (default) - will convert to SVG with dvisvgm
        MathTex.set_default(tex_template=tex_template)
        
        # Set black background
        self.camera.background_color = BLACK

        # ===== Title Slide =====
        title = Text("Dot Product Proof", font_size=48, color=WHITE)
        subtitle = MathTex(r"\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta", 
                          font_size=36, color=YELLOW)
        subtitle.next_to(title, DOWN, buff=0.5)

        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(subtitle))

        # ===== Slide 1: Vector Setup =====
        comment1 = Text("Consider two vectors a and b in 2D", font_size=24, color=WHITE).to_edge(UP, buff=0.3)

        # Create coordinate axes
        axes = Axes(
            x_range=[-0.5, 4, 1],
            y_range=[-0.5, 3, 1],
            x_length=6,
            y_length=4,
            axis_config={"color": WHITE, "include_tip": True},
        ).shift(LEFT * 1)

        origin = axes.c2p(0, 0)

        # Vector a (at angle θ+φ)
        a_end = axes.c2p(3, 2)
        vec_a = Arrow(origin, a_end, color=BLUE, buff=0)
        label_a = MathTex(r"\mathbf{a}", color=BLUE, font_size=30).next_to(a_end, UP+RIGHT, buff=0.1)

        # Vector b (at angle φ)
        b_end = axes.c2p(3, 1)
        vec_b = Arrow(origin, b_end, color=RED, buff=0)
        label_b = MathTex(r"\mathbf{b}", color=RED, font_size=30).next_to(b_end, DOWN+RIGHT, buff=0.1)

        # Angle arc between vectors
        angle_arc = Arc(radius=0.5, start_angle=0.32, angle=0.32, arc_center=origin, color=YELLOW)
        theta_label = MathTex(r"\theta", color=YELLOW, font_size=24).move_to(
            origin + 0.8 * (axes.c2p(1, 0.5) - origin) / 1.1
        )

        self.play(Write(comment1))
        self.play(Create(axes))
        self.play(Create(vec_a), Write(label_a))
        self.play(Create(vec_b), Write(label_b))
        self.play(Create(angle_arc), Write(theta_label))
        self.next_slide()

        # ===== Slide 2: Component Equations =====
        comment2 = Text("Express components using angles", font_size=24, color=WHITE).to_edge(UP, buff=0.3)

        eq_group1 = VGroup(
            MathTex(r"a_x = \|\mathbf{a}\| \cos(\theta + \phi)", font_size=28, color=BLUE),
            MathTex(r"a_y = \|\mathbf{a}\| \sin(\theta + \phi)", font_size=28, color=BLUE),
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(RIGHT, buff=0.5).shift(UP * 1)

        eq_group2 = VGroup(
            MathTex(r"b_x = \|\mathbf{b}\| \cos(\phi)", font_size=28, color=RED),
            MathTex(r"b_y = \|\mathbf{b}\| \sin(\phi)", font_size=28, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT).next_to(eq_group1, DOWN, buff=0.5)

        self.play(Transform(comment1, comment2))
        self.play(Write(eq_group1))
        self.play(Write(eq_group2))
        self.next_slide()

        # Clear and move to algebraic proof
        self.play(
            FadeOut(axes), FadeOut(vec_a), FadeOut(vec_b),
            FadeOut(label_a), FadeOut(label_b),
            FadeOut(angle_arc), FadeOut(theta_label),
            FadeOut(eq_group1), FadeOut(eq_group2)
        )

        # ===== Slide 3: Trigonometric Identity =====
        comment3 = Text("Apply angle addition formulas", font_size=24, color=WHITE).to_edge(UP, buff=0.3)

        trig_id = VGroup(
            MathTex(r"\cos(\theta + \phi) = \cos\phi \cos\theta - \sin\phi \sin\theta", font_size=32, color=WHITE),
            MathTex(r"\sin(\theta + \phi) = \sin\phi \cos\theta + \cos\phi \sin\theta", font_size=32, color=WHITE),
        ).arrange(DOWN, buff=0.3)

        self.play(Transform(comment1, comment3))
        self.play(Write(trig_id))
        self.next_slide()

        # ===== Slide 4: Substitution =====
        comment4 = Text("Substitute angle expressions", font_size=24, color=WHITE).to_edge(UP, buff=0.3)

        subst = VGroup(
            MathTex(r"\cos\phi = \frac{b_x}{\|\mathbf{b}\|}", font_size=28, color=RED),
            MathTex(r"\sin\phi = \frac{b_y}{\|\mathbf{b}\|}", font_size=28, color=RED),
        ).arrange(RIGHT, buff=1).next_to(trig_id, DOWN, buff=0.5)

        self.play(Transform(comment1, comment4))
        self.play(Write(subst))
        self.next_slide()
        self.play(FadeOut(trig_id), FadeOut(subst))

        # ===== Slide 5: Matrix Form =====
        comment5 = Text("Write as matrix equation", font_size=24, color=WHITE).to_edge(UP, buff=0.3)

        matrix_eq = MathTex(
            r"\frac{1}{\|\mathbf{b}\|} \begin{bmatrix} b_x & -b_y \\ b_y & b_x \end{bmatrix}"
            r"\begin{bmatrix} \cos\theta \\ \sin\theta \end{bmatrix}"
            r"= \frac{1}{\|\mathbf{a}\|} \begin{bmatrix} a_x \\ a_y \end{bmatrix}",
            font_size=32, color=WHITE
        )

        self.play(Transform(comment1, comment5))
        self.play(Write(matrix_eq))
        self.next_slide()
        self.play(FadeOut(matrix_eq))

        # ===== Slide 6: Solution =====
        comment6 = Text("Solve for cos(θ)", font_size=24, color=WHITE).to_edge(UP, buff=0.3)

        solution = MathTex(
            r"\cos\theta = \frac{a_x b_x + a_y b_y}{\|\mathbf{a}\| \|\mathbf{b}\|}",
            font_size=40, color=GREEN
        )

        self.play(Transform(comment1, comment6))
        self.play(Write(solution))
        self.next_slide()

        # ===== Slide 7: Final Result =====
        comment7 = Text("Final Result", font_size=24, color=WHITE).to_edge(UP, buff=0.3)

        final = MathTex(
            r"\|\mathbf{a}\| \|\mathbf{b}\| \cos\theta = a_x b_x + a_y b_y = \mathbf{a} \cdot \mathbf{b}",
            font_size=44, color=YELLOW
        )

        qed = Text("Q.E.D. □", font_size=36, color=GREEN).next_to(final, DOWN, buff=0.5)

        self.play(
            Transform(comment1, comment7),
            ReplacementTransform(solution, final)
        )
        self.play(Write(qed))
        self.next_slide()


@app.cell
def _(MANIM_AVAILABLE, mo):
    _install_header = ""
    if not MANIM_AVAILABLE:
        _install_header = """
## 📦 Installation

To enable animated visualizations, install the required packages using the button below.
"""
    mo.md(_install_header) if _install_header else None
    return


@app.cell
def _(MANIM_AVAILABLE, mo):
    import subprocess as _subprocess
    import sys as _sys
    
    if not MANIM_AVAILABLE:
        install_button = mo.ui.run_button(label="🔧 Install manim, manim-slides & moterm")
        _display = install_button
    else:
        install_button = None
        _display = None
    _display
    return install_button, subprocess, sys


@app.cell
def _(install_button, MANIM_AVAILABLE, mo, _subprocess, _sys):
    _install_status = ""
    
    if not MANIM_AVAILABLE and install_button and install_button.value:
        _install_status = "**Installing packages... This may take a few minutes.**"
        
        try:
            _result = _subprocess.run(
                [_sys.executable, "-m", "pip", "install", "manim==0.19.1", "manim-slides==5.5.2", "moterm==0.1.0"],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if _result.returncode == 0:
                _install_status = """
✅ **Installation complete!**

Packages installed:
- `manim==0.19.1` - Mathematical animation engine
- `manim-slides==5.5.2` - Slide presentation extension  
- `moterm==0.1.0` - Terminal command integration

**Please restart this notebook** to use the animations.
"""
            else:
                _error_output = _result.stderr if _result.stderr else _result.stdout
                _install_status = f"""
❌ **Installation failed**

Error output:
```
{_error_output[:500] if len(_error_output) > 500 else _error_output}
```

Try running manually:
```bash
pip install manim==0.19.1 manim-slides==5.5.2 moterm==0.1.0
```
"""
        except _subprocess.TimeoutExpired:
            _install_status = """
⏱️ **Installation timed out**

The installation is taking longer than expected. Please run manually:
```bash
pip install manim==0.19.1 manim-slides==5.5.2 moterm==0.1.0
```
"""
        except Exception as _e:
            _install_status = f"""
❌ **Installation error**

{str(_e)}

Try running manually:
```bash
pip install manim==0.19.1 manim-slides==5.5.2 moterm==0.1.0
```
"""
    
    _output = mo.md(_install_status) if _install_status else None
    _output
    return


@app.cell
def _(MANIM_AVAILABLE, mo):
    _status_msg = """
## ⚠️ Manim Not Installed

Install the packages using the button above, or run manually:
```bash
pip install manim==0.19.1 manim-slides==5.5.2 moterm==0.1.0
```

Then restart this notebook.
"""
    if MANIM_AVAILABLE:
        _status_msg = """
## ✅ Manim Available

The slides are defined above. To render them, run the cells below.

**Note:** First time rendering may take a minute.
"""
    mo.md(_status_msg)
    return


@app.cell
def _(Path, mo):
    import subprocess
    from moterm import Kmd

    DotProductProof

    # Check if LaTeX standalone package is available
    _latex_check = subprocess.run(
        ["pdflatex", "-version"],
        capture_output=True,
        text=True
    )
    
    _latex_available = _latex_check.returncode == 0
    
    if not _latex_available:
        mo.md("""
        ## ⚠️ LaTeX Not Found
        
        Manim requires LaTeX to render mathematical equations. Please install LaTeX:
        
        ```bash
        brew install --cask basictex
        ```
        
        Then install the required package:
        ```bash
        sudo tlmgr install standalone
        ```
        """)
    else:
        mo.md("""
        ### Step 1: Rendering Slides
        
        This will render all animation frames. **This may take 5-10 minutes on the first run.**
        
        **Note:** If you see LaTeX errors about missing `standalone.cls`, run:
        ```bash
        sudo tlmgr install standalone
        ```
        """)

    # Render the slides - this may take several minutes
    if _latex_available:
        # Set TeX environment variables so dvisvgm can find PostScript files
        import os
        os.environ['TEXMFHOME'] = '/opt/homebrew/share/texmf'
        os.environ['TEXMFVAR'] = '/opt/homebrew/var/texlive/2025/texmf-var'
        os.environ['TEXMFCONFIG'] = '/opt/homebrew/var/texlive/2025/texmf-config'
        # Also set TEXMFDIST so dvisvgm can find PostScript files
        os.environ['TEXMFDIST'] = '/opt/homebrew/Cellar/texlive/20250308_2/share/texmf-dist'
        render_output = Kmd("TEXMFHOME=/opt/homebrew/share/texmf TEXMFDIST=/opt/homebrew/Cellar/texlive/20250308_2/share/texmf-dist manim-slides render dot_product_proof_slides.py DotProductProof")
        render_output
    else:
        render_output = None

    if _latex_available and render_output:
        mo.md("### Step 2: Converting to HTML")
        
        # Convert to HTML
        convert_output = Kmd("manim-slides convert DotProductProof -c controls=true dot_product_proof.html --one-file")
        convert_output
    else:
        convert_output = None
        mo.md("**Skipping conversion - LaTeX not available or rendering failed.**")

    # Check for HTML file - try multiple possible locations
    html_paths = [
        Path("dot_product_proof.html"),
        Path.cwd() / "dot_product_proof.html",
        Path("/Users/harunpirim/Documents/GitHub/IME775/week-01/dot_product_proof.html"),
    ]
    
    html_path = None
    for path in html_paths:
        if path.exists():
            html_path = path
            break
    
    if html_path and html_path.exists():
        try:
            _display = mo.md("### Step 3: Displaying Slides")
            _display = mo.iframe(html_path.read_text())
        except Exception as e:
            _display = mo.md(f"""
            **Error reading HTML file:**
            
            {str(e)}
            
            File found at: `{html_path}`
            """)
    else:
        _display = mo.md(f"""
        ### Step 3: Waiting for HTML File
        
        **Rendering in progress...** The HTML file will appear here once both steps complete.
        
        **Note:** Rendering can take 5-10 minutes. Check the output above to see progress.
        
        **Current status:**
        - Current directory: `{Path.cwd()}`
        - HTML file exists: `{html_paths[0].exists() if html_paths else False}`
        
        **Tip:** If rendering seems stuck, check the terminal output above. Once you see "Rendered successfully" or similar, refresh this cell.
        """)
    
    _display
    return


if __name__ == "__main__":
    app.run()
