"""
plot_architecture.py — Generate high-resolution publication-quality diagram
of the proposed Edge-Native Anchor-Free Drone Detection Architecture (Model-D).
Pristine alignment, pointed vector arrows, explicit Concat fusion, zero overlaps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Configure publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "mathtext.fontset": "cm",
})

def draw_block(ax, x, y, w, h, title, subtitle="", fc="#e1f5fe", ec="#0288d1", lw=1.6, text_color="#01579b", rx=0.25):
    """Draw a rounded stylized architectural block with crisp contrast."""
    box = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.1,rounding_size={rx}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=0.98, zorder=3
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x, y + h*0.18, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color=text_color, zorder=4)
        ax.text(x, y - h*0.22, subtitle, ha="center", va="center", fontsize=7.4, color="#37474f", zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color=text_color, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color="#37474f", lw=1.8, label="", label_pos=0.5, label_offset=0.22, ls="-", label_color=None):
    """Draw a prominent directed connection arrow with a clear pointed head."""
    if label_color is None:
        label_color = color
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            ls=ls,
            mutation_scale=14,
            shrinkA=2,
            shrinkB=2,
        ),
        zorder=2
    )
    if label:
        lx = x1 + (x2 - x1) * label_pos
        ly = y1 + (y2 - y1) * label_pos + label_offset
        ax.text(
            lx, ly, label,
            ha="center", va="center",
            fontsize=7.2, color=label_color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="#e0e0e0", alpha=0.9, lw=0.6),
            zorder=6
        )

def generate_architecture_diagram(output_path: Path):
    fig, ax = plt.subplots(figsize=(20.5, 9.5), dpi=300)
    ax.set_xlim(-0.8, 24.2)
    ax.set_ylim(-0.8, 10.2)
    ax.axis("off")

    # -------------------------------------------------------------
    # 0. Container Background Sections (Swimlanes)
    # -------------------------------------------------------------
    # Input Stage
    ax.add_patch(patches.FancyBboxPatch((-0.4, 0.4), 2.8, 9.2, boxstyle="round,pad=0.2,rounding_size=0.25",
                                        facecolor="#fafafa", edgecolor="#cfd8dc", ls="--", lw=1.2, zorder=0))
    ax.text(1.0, 9.2, "Input Stage", ha="center", fontweight="bold", fontsize=10.5, color="#37474f")

    # 4-Stage Backbone
    ax.add_patch(patches.FancyBboxPatch((2.8, 0.4), 4.8, 9.2, boxstyle="round,pad=0.2,rounding_size=0.25",
                                        facecolor="#f3e5f5", edgecolor="#ba68c8", ls="--", lw=1.2, alpha=0.4, zorder=0))
    ax.text(5.2, 9.2, "4-Stage CSPDarknet Backbone", ha="center", fontweight="bold", fontsize=10.5, color="#6a1b9a")

    # 4-Level High-Res Feature Neck
    ax.add_patch(patches.FancyBboxPatch((8.0, 0.4), 8.6, 9.2, boxstyle="round,pad=0.2,rounding_size=0.25",
                                        facecolor="#e8f5e9", edgecolor="#81c784", ls="--", lw=1.2, alpha=0.4, zorder=0))
    ax.text(12.3, 9.2, "4-Level High-Res P2-P5 Neck (FPN + CBAM + PAN)", ha="center", fontweight="bold", fontsize=10.5, color="#1b5e20")

    # Decoupled Heads & Loss
    ax.add_patch(patches.FancyBboxPatch((17.0, 0.4), 6.8, 9.2, boxstyle="round,pad=0.2,rounding_size=0.25",
                                        facecolor="#fff3e0", edgecolor="#ffb74d", ls="--", lw=1.2, alpha=0.4, zorder=0))
    ax.text(20.4, 9.2, "Decoupled Anchor-Free Heads & Loss", ha="center", fontweight="bold", fontsize=10.5, color="#e65100")

    # Row Y-Coordinates for 4 Scale Levels
    y_p2 = 7.7   # Stride 4  (104x104) -> Micro-Drones (<20px)
    y_p3 = 5.8   # Stride 8  (52x52)   -> Tiny Drones
    y_p4 = 3.9   # Stride 16 (26x26)   -> Small Drones
    y_p5 = 2.0   # Stride 32 (13x13)   -> Medium Drones

    # -------------------------------------------------------------
    # 1. Input Image Block (Aligned horizontally with Stage 1 / P2)
    # -------------------------------------------------------------
    draw_block(ax, 1.0, y_p2, 2.0, 1.5, "UAV Image", "3 × 416 × 416\nRGB Frame", fc="#eceff1", ec="#78909c", text_color="#263238")
    
    # Arrow: Input Image -> Focus/Stem -> Stage 1 (C2) [Straight Horizontal Line]
    draw_arrow(ax, 2.0, y_p2, 3.7, y_p2, color="#455a64", lw=1.8, label="Focus Stem (s2)", label_pos=0.5, label_offset=0.22)

    # -------------------------------------------------------------
    # 2. Backbone Stages (C2, C3, C4, C5) + SPPF
    # -------------------------------------------------------------
    x_bb = 4.8
    draw_block(ax, x_bb, y_p2, 2.2, 1.1, "Stage 1 (C2)", "Stride 4: 104×104 (32c)", fc="#ede7f6", ec="#7e57c2", text_color="#4527a0")
    draw_block(ax, x_bb, y_p3, 2.2, 1.1, "Stage 2 (C3)", "Stride 8: 52×52 (64c)", fc="#ede7f6", ec="#7e57c2", text_color="#4527a0")
    draw_block(ax, x_bb, y_p4, 2.2, 1.1, "Stage 3 (C4)", "Stride 16: 26×26 (128c)", fc="#ede7f6", ec="#7e57c2", text_color="#4527a0")
    draw_block(ax, x_bb, y_p5, 2.2, 1.1, "Stage 4 (C5)", "Stride 32: 13×13 (256c)", fc="#ede7f6", ec="#7e57c2", text_color="#4527a0")

    # Downsampling arrows between backbone stages
    draw_arrow(ax, x_bb, y_p2 - 0.55, x_bb, y_p3 + 0.55, color="#5e35b1", lw=1.8, label="Conv 3×3 (s2)", label_offset=0.0)
    draw_arrow(ax, x_bb, y_p3 - 0.55, x_bb, y_p4 + 0.55, color="#5e35b1", lw=1.8, label="Conv 3×3 (s2)", label_offset=0.0)
    draw_arrow(ax, x_bb, y_p4 - 0.55, x_bb, y_p5 + 0.55, color="#5e35b1", lw=1.8, label="Conv 3×3 (s2)", label_offset=0.0)

    # SPPF Block attached to Stage 4 (C5)
    x_sppf = 6.6
    draw_block(ax, x_sppf, y_p5, 1.5, 1.1, "SPPF", "Pool {5,9,13}", fc="#d1c4e9", ec="#512da8", text_color="#311b92")
    draw_arrow(ax, x_bb + 1.1, y_p5, x_sppf - 0.75, y_p5, color="#512da8", lw=1.8)

    # -------------------------------------------------------------
    # 3. 4-Level High-Res Feature Neck (FPN + CBAM + PAN)
    # -------------------------------------------------------------
    x_fpn = 9.4      # FPN Node Column
    x_cbam = 12.1    # CBAM Attention Column
    x_pan = 14.8     # PAN Node Column

    # --- FPN Top-Down Column ---
    draw_block(ax, x_fpn, y_p2, 1.6, 1.1, "P2 Node", "104×104 (s4)", fc="#e0f2f1", ec="#26a69a", text_color="#004d40")
    draw_block(ax, x_fpn, y_p3, 1.6, 1.1, "P3 Node", "52×52 (s8)", fc="#e0f2f1", ec="#26a69a", text_color="#004d40")
    draw_block(ax, x_fpn, y_p4, 1.6, 1.1, "P4 Node", "26×26 (s16)", fc="#e0f2f1", ec="#26a69a", text_color="#004d40")
    draw_block(ax, x_fpn, y_p5, 1.6, 1.1, "P5 Node", "13×13 (s32)", fc="#e0f2f1", ec="#26a69a", text_color="#004d40")

    # Lateral connections (Backbone -> FPN)
    draw_arrow(ax, x_bb + 1.1, y_p2, x_fpn - 0.8, y_p2, color="#00897b", lw=1.8, label="1×1 Lateral C2", label_pos=0.5, label_offset=0.22)
    draw_arrow(ax, x_bb + 1.1, y_p3, x_fpn - 0.8, y_p3, color="#00897b", lw=1.8, label="1×1 Lateral C3", label_pos=0.5, label_offset=0.22)
    draw_arrow(ax, x_bb + 1.1, y_p4, x_fpn - 0.8, y_p4, color="#00897b", lw=1.8, label="1×1 Lateral C4", label_pos=0.5, label_offset=0.22)
    draw_arrow(ax, x_sppf + 0.75, y_p5, x_fpn - 0.8, y_p5, color="#00897b", lw=1.8, label="Lateral C5", label_pos=0.5, label_offset=0.22)

    # Top-Down FPN Upsampling Arrows (from P5 up to P2)
    draw_arrow(ax, x_fpn, y_p5 + 0.55, x_fpn, y_p4 - 0.55, color="#00796b", lw=1.8, label="Upsample 2×", label_offset=0.0)
    draw_arrow(ax, x_fpn, y_p4 + 0.55, x_fpn, y_p3 - 0.55, color="#00796b", lw=1.8, label="Upsample 2×", label_offset=0.0)
    draw_arrow(ax, x_fpn, y_p3 + 0.55, x_fpn, y_p2 - 0.55, color="#00796b", lw=1.8, label="Upsample 2×", label_offset=0.0)

    # --- CBAM Dual Attention Column ---
    draw_block(ax, x_cbam, y_p2, 1.7, 1.1, "CBAM", "Ch + Spat Attn", fc="#c8e6c9", ec="#43a047", text_color="#1b5e20")
    draw_block(ax, x_cbam, y_p3, 1.7, 1.1, "CBAM", "Ch + Spat Attn", fc="#c8e6c9", ec="#43a047", text_color="#1b5e20")
    draw_block(ax, x_cbam, y_p4, 1.7, 1.1, "CBAM", "Ch + Spat Attn", fc="#c8e6c9", ec="#43a047", text_color="#1b5e20")
    draw_block(ax, x_cbam, y_p5, 1.7, 1.1, "CBAM", "Ch + Spat Attn", fc="#c8e6c9", ec="#43a047", text_color="#1b5e20")

    # Arrows from FPN -> CBAM
    draw_arrow(ax, x_fpn + 0.8, y_p2, x_cbam - 0.85, y_p2, color="#2e7d32", lw=1.8)
    draw_arrow(ax, x_fpn + 0.8, y_p3, x_cbam - 0.85, y_p3, color="#2e7d32", lw=1.8)
    draw_arrow(ax, x_fpn + 0.8, y_p4, x_cbam - 0.85, y_p4, color="#2e7d32", lw=1.8)
    draw_arrow(ax, x_fpn + 0.8, y_p5, x_cbam - 0.85, y_p5, color="#2e7d32", lw=1.8)

    # --- PAN Bottom-Up Column ---
    draw_block(ax, x_pan, y_p2, 1.6, 1.1, "N2 Fusion", "104×104 (P2)", fc="#e8f8f5", ec="#1abc9c", text_color="#0e6251")
    draw_block(ax, x_pan, y_p3, 1.6, 1.1, "N3 Fusion", "52×52 (P3)", fc="#e8f8f5", ec="#1abc9c", text_color="#0e6251")
    draw_block(ax, x_pan, y_p4, 1.6, 1.1, "N4 Fusion", "26×26 (P4)", fc="#e8f8f5", ec="#1abc9c", text_color="#0e6251")
    draw_block(ax, x_pan, y_p5, 1.6, 1.1, "N5 Fusion", "13×13 (P5)", fc="#e8f8f5", ec="#1abc9c", text_color="#0e6251")

    # Arrows from CBAM -> PAN
    draw_arrow(ax, x_cbam + 0.85, y_p2, x_pan - 0.8, y_p2, color="#0e6251", lw=1.8)
    draw_arrow(ax, x_cbam + 0.85, y_p3, x_pan - 0.8, y_p3, color="#0e6251", lw=1.8)
    draw_arrow(ax, x_cbam + 0.85, y_p4, x_pan - 0.8, y_p4, color="#0e6251", lw=1.8)
    draw_arrow(ax, x_cbam + 0.85, y_p5, x_pan - 0.8, y_p5, color="#0e6251", lw=1.8)

    # Bottom-Up PAN Downsampling Arrows (from N2 down to N5)
    draw_arrow(ax, x_pan, y_p2 - 0.55, x_pan, y_p3 + 0.55, color="#0e6251", lw=1.8, label="Conv 3×3 (s2)", label_offset=0.0)
    draw_arrow(ax, x_pan, y_p3 - 0.55, x_pan, y_p4 + 0.55, color="#0e6251", lw=1.8, label="Conv 3×3 (s2)", label_offset=0.0)
    draw_arrow(ax, x_pan, y_p4 - 0.55, x_pan, y_p5 + 0.55, color="#0e6251", lw=1.8, label="Conv 3×3 (s2)", label_offset=0.0)

    # -------------------------------------------------------------
    # 4. Decoupled Anchor-Free Detection Heads & Subnets
    # -------------------------------------------------------------
    x_head = 18.0

    draw_block(ax, x_head, y_p2, 1.7, 1.1, "Head (P2)", "104×104 (Micro)", fc="#fff8e1", ec="#ffa000", text_color="#e65100")
    draw_block(ax, x_head, y_p3, 1.7, 1.1, "Head (P3)", "52×52 (Tiny)", fc="#fff8e1", ec="#ffa000", text_color="#e65100")
    draw_block(ax, x_head, y_p4, 1.7, 1.1, "Head (P4)", "26×26 (Small)", fc="#fff8e1", ec="#ffa000", text_color="#e65100")
    draw_block(ax, x_head, y_p5, 1.7, 1.1, "Head (P5)", "13×13 (Medium)", fc="#fff8e1", ec="#ffa000", text_color="#e65100")

    # Direct 1-to-1 horizontal arrows from PAN -> Heads
    draw_arrow(ax, x_pan + 0.8, y_p2, x_head - 0.85, y_p2, color="#e65100", lw=1.8)
    draw_arrow(ax, x_pan + 0.8, y_p3, x_head - 0.85, y_p3, color="#e65100", lw=1.8)
    draw_arrow(ax, x_pan + 0.8, y_p4, x_head - 0.85, y_p4, color="#e65100", lw=1.8)
    draw_arrow(ax, x_pan + 0.8, y_p5, x_head - 0.85, y_p5, color="#e65100", lw=1.8)

    # --- Decoupled Output Subnets (Classification, Regression, Quality) ---
    x_out = 21.6
    draw_block(ax, x_out, 7.5, 2.7, 1.2, r"$\mathbf{H}_{cls}$ (Sigmoid Focal)", r"$\hat{P}_{cls} \in [0, 1]^{1 \times H \times W}$", fc="#ffe0b2", ec="#fb8c00", text_color="#bf360c")
    draw_block(ax, x_out, 5.5, 2.7, 1.2, r"$\mathbf{H}_{reg}$ (CIoU + NWD)", r"$\hat{\mathbf{t}} = (l, t, r, b) \in \mathbb{R}_+^4$", fc="#ffe0b2", ec="#fb8c00", text_color="#bf360c")
    draw_block(ax, x_out, 3.5, 2.7, 1.2, r"$\mathbf{H}_{obj}$ (Centerness)", r"$\hat{\mathbf{q}} \in [0, 1]^{1 \times H \times W}$", fc="#ffe0b2", ec="#fb8c00", text_color="#bf360c")

    # Arrows from Head stack into the 3 Decoupled Subnets
    draw_arrow(ax, x_head + 0.85, y_p2, x_out - 1.35, 7.5, color="#bf360c", lw=1.5)
    draw_arrow(ax, x_head + 0.85, y_p3, x_out - 1.35, 5.5, color="#bf360c", lw=1.5)
    draw_arrow(ax, x_head + 0.85, y_p4, x_out - 1.35, 3.5, color="#bf360c", lw=1.5)
    draw_arrow(ax, x_head + 0.85, y_p5, x_out - 1.35, 3.5, color="#bf360c", lw=1.5)

    # --- Model EMA Shadow Tracking Box (Placed cleanly at bottom-right with ZERO overlap) ---
    draw_block(ax, x_out, 1.4, 3.1, 1.2,
               r"$\mathbf{Model\ EMA}$",
               r"$\beta = 0.9999\ \text{Shadow Weights}$",
               fc="#e8eaf6", ec="#3f51b5", text_color="#1a237e")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Architecture diagram saved to: {output_path}")

if __name__ == "__main__":
    out1 = Path("paper/figures/vanilla_drone_architecture.png")
    out2 = Path("paper/vanilla_drone_architecture.png")
    generate_architecture_diagram(out1)
    generate_architecture_diagram(out2)
