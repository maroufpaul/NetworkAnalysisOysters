#!/usr/bin/env python3
"""
Generate a PowerPoint presentation for the oyster reef site selection project.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
OUT_PATH = ROOT / "presentation.pptx"

# Colors
DARK_BLUE = RGBColor(0x1A, 0x23, 0x7E)
MEDIUM_BLUE = RGBColor(0x21, 0x96, 0xF3)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
LIGHT_GRAY = RGBColor(0x66, 0x66, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT_GREEN = RGBColor(0x4C, 0xAF, 0x50)
BG_LIGHT = RGBColor(0xF5, 0xF5, 0xF5)


def set_slide_bg(slide, color):
    """Set solid background color for a slide."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 bold=False, color=DARK_GRAY, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    """Add a text box to a slide."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return tf


def add_bullet_slide(slide, left, top, width, height, bullets, font_size=16,
                     color=DARK_GRAY, spacing=Pt(8)):
    """Add bulleted text to a slide."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, (text, bold, indent_level) in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.level = indent_level
        p.space_after = spacing
    return tf


def add_image_centered(slide, img_path, top_inches, max_width=8.5, max_height=4.5):
    """Add an image centered horizontally on the slide."""
    from PIL import Image
    img = Image.open(img_path)
    img_w, img_h = img.size
    aspect = img_w / img_h

    # Fit within bounds
    if max_width / aspect <= max_height:
        w = max_width
        h = w / aspect
    else:
        h = max_height
        w = h * aspect

    left = (10 - w) / 2
    slide.shapes.add_picture(str(img_path), Inches(left), Inches(top_inches),
                             Inches(w), Inches(h))


def make_title_slide(prs):
    """Slide 1: Title slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_slide_bg(slide, DARK_BLUE)

    add_text_box(slide, 0.8, 1.5, 8.4, 1.5,
                 "Optimal Oyster Reef Site Selection\nin the Chesapeake Bay",
                 font_size=32, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    add_text_box(slide, 0.8, 3.3, 8.4, 0.8,
                 "Combining ODE Modeling, Heuristic Optimization, and\nMixed-Integer Quadratic Programming",
                 font_size=18, color=RGBColor(0xBB, 0xDE, 0xFB), alignment=PP_ALIGN.CENTER)

    add_text_box(slide, 0.8, 4.8, 8.4, 0.5,
                 "Paul Marouf",
                 font_size=20, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)


def make_problem_slide(prs):
    """Slide 2: The problem."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.3, 9, 0.6,
                 "The Problem: Oyster Reef Restoration",
                 font_size=28, bold=True, color=DARK_BLUE)

    bullets = [
        ("Chesapeake Bay has lost ~99% of its oyster population", True, 0),
        ("From ~10 billion to fewer than 100 million", False, 1),
        ("Restoration is expensive and resource-constrained", True, 0),
        ("We can only restore 25 of 48 candidate reef sites", False, 1),
        ("Key question:", True, 0),
        ("Which 25 sites maximize total adult oyster biomass,", False, 1),
        ("given the larval connectivity network between reefs?", False, 1),
        ("This is a combinatorial optimization problem", True, 0),
        ("C(48, 25) = 1.4 trillion possible selections", False, 1),
        ("Brute force is impossible - need smart algorithms", False, 1),
    ]
    add_bullet_slide(slide, 0.7, 1.1, 8.5, 5.5, bullets, font_size=16)


def make_model_slide(prs):
    """Slide 3: The JARS ODE model (given)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.3, 9, 0.6,
                 "The Biological Model: JARS ODE (Provided by Dr. Shaw)",
                 font_size=26, bold=True, color=DARK_BLUE)

    bullets = [
        ("4-state ODE system simulates oyster life stages across reef patches:", True, 0),
        ("J = Juveniles  |  A = Adults (biomass)  |  R = Reef/shell  |  S = Sediment", False, 1),
        ("Sites are coupled through larval dispersal:", True, 0),
        ("P0 = external larvae input (from outside the network)", False, 1),
        ("P1 = internal connectivity matrix (larvae exchanged between sites)", False, 1),
        ("Larval input = (P0 + P1' * |A|^1.72) * L(A,R) * f(A,R,S)", False, 1),
        ("Key property: nonlinear interactions between sites", True, 0),
        ("A site's value depends on which OTHER sites are also active", False, 1),
        ("Evaluating one subset = solving a 4N-dimensional ODE to t=1000", False, 1),
        ("Objective: maximize total adult biomass sum(A) at equilibrium", True, 0),
    ]
    add_bullet_slide(slide, 0.5, 1.1, 9.0, 5.5, bullets, font_size=15)


def make_connectivity_slide(prs):
    """Slide 4: Connectivity heatmap."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.2, 9, 0.5,
                 "Larval Connectivity Network",
                 font_size=26, bold=True, color=DARK_BLUE)

    add_image_centered(slide, FIGURES_DIR / "fig8_connectivity_heatmap.png",
                       0.8, max_width=6.0, max_height=5.5)

    add_text_box(slide, 6.3, 1.5, 3.2, 4.0,
                 "Data: 56-site connectivity matrix from oceanographic modeling\n\n"
                 "Strong connectivity clusters visible (hot cells)\n\n"
                 "Sites 10, 15, 27-29, 31-32, 40-41 form a highly connected core\n\n"
                 "Sparse regions indicate isolated sites with low restoration value",
                 font_size=12, color=DARK_GRAY)


def make_my_contribution_slide(prs):
    """Slide 5: What I built."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.3, 9, 0.6,
                 "My Contribution: Three Solution Approaches",
                 font_size=26, bold=True, color=DARK_BLUE)

    bullets = [
        ("Approach 1: Heuristic Search (uses full ODE)", True, 0),
        ("Forward greedy: add best site one at a time (25 rounds)", False, 1),
        ("Backward elimination: start with all 48, remove least impactful", False, 1),
        ("Local search: 1-swap hill climbing to refine greedy output", False, 1),
        ("Approach 2: MIQP Surrogate (my key contribution)", True, 0),
        ("ODE too expensive for exact optimization", False, 1),
        ("Built a quadratic surrogate: max sum(Pe_i * x_i) + sum(W_ij * x_i * x_j)", False, 1),
        ("W_ij = P1_ij * (A*)^1.72 approximates the network effect", False, 1),
        ("Solved exactly with Gurobi in 0.32 seconds", False, 1),
        ("Approach 3: Constrained MIQP Variants", True, 0),
        ("Community constraints: minimum reefs per geographic region", False, 1),
        ("Sizing constraints: optimize reef area allocation under budget", False, 1),
    ]
    add_bullet_slide(slide, 0.5, 1.0, 9.0, 5.5, bullets, font_size=15)


def make_greedy_curve_slide(prs):
    """Slide 6: Greedy selection curve."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.2, 9, 0.5,
                 "Greedy Site Selection: Marginal Returns",
                 font_size=26, bold=True, color=DARK_BLUE)

    add_image_centered(slide, FIGURES_DIR / "fig7_greedy_curve.png",
                       0.7, max_width=8.5, max_height=5.0)

    add_text_box(slide, 0.5, 5.9, 9.0, 0.8,
                 "Each added site increases biomass, with a jump at sites 17-18 (adding hub sites 40, 41). "
                 "Greedy+Local Search ultimately achieves the best ODE score (1.862).",
                 font_size=12, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)


def make_score_comparison_slide(prs):
    """Slide 7: Score comparison bar chart."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.2, 9, 0.5,
                 "ODE-Validated Performance Comparison",
                 font_size=26, bold=True, color=DARK_BLUE)

    add_image_centered(slide, FIGURES_DIR / "fig1_score_comparison.png",
                       0.7, max_width=7.5, max_height=4.8)

    add_text_box(slide, 0.5, 5.8, 9.0, 1.0,
                 "All methods validated through the full JARS ODE (tmax=1000). "
                 "Greedy+Local achieves the highest biomass. MIQP is 8% behind but solves in 0.3s vs 32 min. "
                 "Community constraints reduce score by 26% - the cost of geographic fairness.",
                 font_size=12, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)


def make_timing_slide(prs):
    """Slide 8: Timing comparison."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.2, 9, 0.5,
                 "Computational Time: Heuristics vs MIQP",
                 font_size=26, bold=True, color=DARK_BLUE)

    add_image_centered(slide, FIGURES_DIR / "fig9_timing.png",
                       0.7, max_width=7.0, max_height=4.8)

    add_text_box(slide, 0.5, 5.8, 9.0, 1.0,
                 "MIQP solves 6,000x faster than Greedy+Local. The surrogate trades ~8% accuracy "
                 "for near-instant solutions, enabling rapid what-if analysis and constraint exploration.",
                 font_size=12, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)


def make_gap_slide(prs):
    """Slide 9: Optimality gap."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.2, 9, 0.5,
                 "Optimality Gap Analysis",
                 font_size=26, bold=True, color=DARK_BLUE)

    add_image_centered(slide, FIGURES_DIR / "fig10_optimality_gap.png",
                       0.7, max_width=8.0, max_height=4.8)

    add_text_box(slide, 0.5, 5.8, 9.0, 1.0,
                 "Backward greedy is within 2.3% of the best. MIQP's 8% gap reflects the surrogate "
                 "approximation - it optimizes a linearized proxy, not the true nonlinear ODE. "
                 "Community constraints show the explicit cost of policy requirements.",
                 font_size=12, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)


def make_overlap_slide(prs):
    """Slide 10: Site overlap heatmap."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.2, 9, 0.5,
                 "Agreement Between Methods",
                 font_size=26, bold=True, color=DARK_BLUE)

    add_image_centered(slide, FIGURES_DIR / "fig2_site_overlap_heatmap.png",
                       0.6, max_width=5.5, max_height=5.2)

    add_text_box(slide, 6.0, 1.2, 3.6, 4.5,
                 "Jaccard similarity measures overlap.\n\n"
                 "MIQP variants agree strongly with each other (0.72)\n\n"
                 "Backward agrees with MIQP (0.72) - similar ranking of site importance\n\n"
                 "Greedy+Local is most different (0.35-0.56) - local search swaps change the set\n\n"
                 "Despite different selections, all methods share a consensus core of ~10 sites",
                 font_size=12, color=DARK_GRAY)


def make_consensus_slide(prs):
    """Slide 11: Consensus analysis."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.2, 9, 0.5,
                 "Consensus Core: Sites All Methods Agree On",
                 font_size=26, bold=True, color=DARK_BLUE)

    add_image_centered(slide, FIGURES_DIR / "fig4_consensus_sites.png",
                       0.7, max_width=9.0, max_height=3.5)

    add_text_box(slide, 0.5, 4.5, 9.0, 2.5,
                 "10 sites selected by ALL 5 methods: 10, 15, 31, 32, 37, 40, 41, 49, 51, 52, 53\n"
                 "These are the highest-confidence restoration targets regardless of methodology.\n\n"
                 "7 more sites selected by 4/5 methods: 16, 17, 21, 27, 33, 36, 47, 59\n"
                 "Strong candidates with broad algorithmic support.\n\n"
                 "Only 8 sites are contested (selected by 1-2 methods) - these are the "
                 "edge cases where methodology choice matters most.",
                 font_size=13, color=DARK_GRAY)


def make_network_slide(prs):
    """Slide 12: Network centrality."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.2, 9, 0.5,
                 "Network Analysis: Why These Sites?",
                 font_size=26, bold=True, color=DARK_BLUE)

    add_image_centered(slide, FIGURES_DIR / "fig5_network_centrality.png",
                       0.7, max_width=9.0, max_height=3.5)

    add_text_box(slide, 0.5, 4.5, 9.0, 2.5,
                 "MIQP-selected sites (blue) dominate all network centrality metrics:\n"
                 "  - In-Strength: these sites receive the most larvae from the network\n"
                 "  - Out-Strength: these sites export the most larvae to other sites\n"
                 "  - PageRank: these sites are the most 'important' in the connectivity graph\n\n"
                 "Top network hubs (sites 10, 31, 32, 40, 41) are consensus picks across all methods.\n"
                 "The optimization naturally selects network-central sites.",
                 font_size=13, color=DARK_GRAY)


def make_reef_sizes_slide(prs):
    """Slide 13: Reef size allocation."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.2, 9, 0.5,
                 "MIQP Extension: Optimal Reef Size Allocation",
                 font_size=26, bold=True, color=DARK_BLUE)

    add_image_centered(slide, FIGURES_DIR / "fig6_reef_sizes.png",
                       0.7, max_width=9.0, max_height=3.8)

    add_text_box(slide, 0.5, 4.8, 9.0, 2.0,
                 "The MIQP framework naturally extends to variable reef sizing.\n"
                 "Network hubs (sites 10-17, 26-33) get maximum allocation.\n"
                 "Peripheral sites (20, 36, 37) get minimal allocation - selected for connectivity, not size.\n"
                 "Community constraints force inclusion of smaller sites (42, 47, 48) at minimum size.",
                 font_size=13, color=DARK_GRAY)


def make_selection_matrix_slide(prs):
    """Slide 14: Full site selection matrix."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.2, 9, 0.5,
                 "Complete Site Selection Matrix",
                 font_size=26, bold=True, color=DARK_BLUE)

    add_image_centered(slide, FIGURES_DIR / "fig3_site_selection_matrix.png",
                       0.8, max_width=9.0, max_height=4.0)

    add_text_box(slide, 0.5, 5.2, 9.0, 1.5,
                 "Blue = selected. Count row at top shows consensus level per site. "
                 "The dense blue columns (sites 10, 15, 31, 32, 40, 41, 49, 51-53) form the "
                 "indisputable core. Greedy+Local uniquely selects sites 6, 24, 30, 38, 39, 55, 60.",
                 font_size=12, color=LIGHT_GRAY, alignment=PP_ALIGN.CENTER)


def make_summary_table_slide(prs):
    """Slide 15: Summary results table."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.3, 9, 0.6,
                 "Summary of Results",
                 font_size=28, bold=True, color=DARK_BLUE)

    # Create a table
    rows, cols = 6, 5
    table_shape = slide.shapes.add_table(rows, cols, Inches(0.5), Inches(1.1), Inches(9.0), Inches(3.0))
    table = table_shape.table

    headers = ["Method", "ODE Score", "Time", "Overlap w/ Best", "Key Trait"]
    data = [
        ["Greedy+Local", "1.862 (best)", "32 min", "-", "Best ODE score"],
        ["Backward", "1.819 (-2.3%)", "4.7 min", "18/25", "Fast, good quality"],
        ["MIQP", "1.713 (-8.0%)", "0.32 sec", "14/25", "Near-instant, exact surrogate"],
        ["MIQP+Comm", "1.383 (-25.7%)", "<1 sec", "15/25", "Geographic fairness"],
        ["MIQP+Comm+Size", "N/A*", "<1 sec", "13/25", "Full policy model"],
    ]

    # Style header
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = Pt(13)
            paragraph.font.bold = True
            paragraph.font.color.rgb = WHITE
            paragraph.font.name = "Calibri"
            paragraph.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_BLUE

    # Style data
    for i, row_data in enumerate(data):
        for j, val in enumerate(row_data):
            cell = table.cell(i + 1, j)
            cell.text = val
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(12)
                paragraph.font.name = "Calibri"
                paragraph.font.color.rgb = DARK_GRAY
                paragraph.alignment = PP_ALIGN.CENTER
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xE3, 0xF2, 0xFD)

    add_text_box(slide, 0.5, 4.3, 9.0, 0.4,
                 "* MIQP+Comm+Size uses a different objective (total larvae) so ODE score is not directly comparable.",
                 font_size=11, color=LIGHT_GRAY)

    add_text_box(slide, 0.5, 4.8, 9.0, 2.0,
                 "Key Findings:\n"
                 "  1. Greedy+Local Search achieves the best ODE biomass but takes 32 minutes\n"
                 "  2. MIQP solves in 0.3 seconds and provides a provably optimal surrogate solution\n"
                 "  3. The 8% gap between MIQP and Greedy+Local quantifies the surrogate approximation error\n"
                 "  4. All methods agree on a consensus core of ~10 high-value sites\n"
                 "  5. MIQP framework naturally extends to handle real-world policy constraints",
                 font_size=13, bold=False, color=DARK_GRAY)


def make_takeaways_slide(prs):
    """Slide 16: Takeaways."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, DARK_BLUE)

    add_text_box(slide, 0.5, 0.5, 9, 0.8,
                 "Key Takeaways",
                 font_size=32, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

    bullets = [
        ("The MIQP surrogate provides a fast, exact benchmark", True, 0),
        ("Solves in 0.3s what heuristics need 32 minutes for", False, 1),
        ("Enables rapid exploration of constraints and scenarios", False, 1),
        ("", False, 0),
        ("Heuristics outperform the surrogate on the true ODE", True, 0),
        ("Greedy+Local Search achieves 8% better ODE score than MIQP", False, 1),
        ("The surrogate approximation is the limiting factor, not the solver", False, 1),
        ("", False, 0),
        ("All methods converge on a consensus core of ~10 sites", True, 0),
        ("Sites 10, 15, 31, 32, 37, 40, 41, 49, 51, 52, 53 are robust picks", False, 1),
        ("These are the network hubs with highest centrality metrics", False, 1),
        ("", False, 0),
        ("The MIQP framework is extensible to real policy needs", True, 0),
        ("Community fairness, reef sizing, budget constraints all integrate naturally", False, 1),
        ("Quantifies the cost of policy: geographic fairness costs ~26% biomass", False, 1),
    ]
    add_bullet_slide(slide, 0.7, 1.4, 8.5, 5.5, bullets, font_size=14,
                     color=RGBColor(0xE3, 0xF2, 0xFD), spacing=Pt(4))


def make_future_work_slide(prs):
    """Slide 17: Future work / questions."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, WHITE)

    add_text_box(slide, 0.5, 0.5, 9, 0.8,
                 "Future Work & Questions",
                 font_size=28, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)

    bullets = [
        ("Improve the surrogate approximation", True, 0),
        ("Better A* estimate (per-site instead of median)", False, 1),
        ("Higher-order terms or piecewise linearization", False, 1),
        ("Robustness analysis", True, 0),
        ("Test with both connectivity matrices (different tidal conditions)", False, 1),
        ("Sensitivity to P1 scaling factor and P0 mode", False, 1),
        ("Scalability", True, 0),
        ("Parallelize ODE evaluations for faster heuristic runs", False, 1),
        ("Extend to larger candidate sets or multi-objective formulations", False, 1),
    ]
    add_bullet_slide(slide, 0.7, 1.5, 8.5, 4.5, bullets, font_size=16)

    add_text_box(slide, 0.5, 5.5, 9.0, 1.0,
                 "Thank you! Questions?",
                 font_size=28, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)


def main():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # Build slides in order
    make_title_slide(prs)           # 1
    make_problem_slide(prs)         # 2
    make_model_slide(prs)           # 3
    make_connectivity_slide(prs)    # 4
    make_my_contribution_slide(prs) # 5
    make_greedy_curve_slide(prs)    # 6
    make_score_comparison_slide(prs)# 7
    make_timing_slide(prs)          # 8
    make_gap_slide(prs)             # 9
    make_overlap_slide(prs)         # 10
    make_consensus_slide(prs)       # 11
    make_network_slide(prs)         # 12
    make_reef_sizes_slide(prs)      # 13
    make_selection_matrix_slide(prs)# 14
    make_summary_table_slide(prs)   # 15
    make_takeaways_slide(prs)       # 16
    make_future_work_slide(prs)     # 17

    prs.save(str(OUT_PATH))
    print(f"Presentation saved to: {OUT_PATH}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
