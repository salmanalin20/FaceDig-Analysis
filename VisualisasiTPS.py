#!/usr/bin/env python3
"""
FaceDig TPS Viewer — Visualize facial landmarks from TPS files.

Supports batch processing of multiple TPS files across year directories
with automatic output routing to dedicated output folders.
"""

import os
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend for batch processing
plt.switch_backend('Agg')


# ══════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════

@dataclass
class Specimen:
    """A single specimen parsed from a TPS file."""
    landmarks: np.ndarray
    image: Optional[str] = None
    id: Optional[int] = None

    @property
    def n_landmarks(self) -> int:
        return len(self.landmarks)


@dataclass
class LandmarkGroup:
    """Definition of a named group of landmark indices."""
    name: str
    indices: list[int]
    color: str


# ══════════════════════════════════════════════
# Facial Landmark Configuration (72 landmarks)
# ══════════════════════════════════════════════

LANDMARK_GROUPS: list[LandmarkGroup] = [
    LandmarkGroup('Chin / Jawline',     [0],                                    '#E74C3C'),
    LandmarkGroup('Forehead',           [1],                                    '#F39C12'),
    LandmarkGroup('Nose Bridge',        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],      '#2ECC71'),
    LandmarkGroup('Nose Tip / Base',    [12, 13, 14, 15, 16, 17, 18],           '#1ABC9C'),
    LandmarkGroup('Left Eye',           [19, 20, 21, 22, 23, 24, 25, 70],       '#3498DB'),
    LandmarkGroup('Right Eye',          [26, 27, 28, 29, 30, 31, 32, 71],       '#9B59B6'),
    LandmarkGroup('Left Eyebrow',       [33, 34, 35],                           '#E67E22'),
    LandmarkGroup('Right Eyebrow',      [36, 37],                               '#D35400'),
    LandmarkGroup('Upper Lip',          [38, 39, 40, 41, 42, 43],               '#E91E63'),
    LandmarkGroup('Lower Lip',          [44, 45, 46, 47, 48, 49],               '#C2185B'),
    LandmarkGroup('Left Jaw Contour',   [50, 51, 52, 53, 54, 55, 56, 57, 58],  '#FF7043'),
    LandmarkGroup('Right Jaw Contour',  [59, 60, 61, 62, 63, 64, 65, 66, 67],  '#AB47BC'),
    LandmarkGroup('Nose Midline',       [68, 69],                               '#26A69A'),
]

# Pre-compute all grouped indices for quick lookup
_GROUPED_INDICES: set[int] = {idx for g in LANDMARK_GROUPS for idx in g.indices}

# Wireframe connections (pairs of landmark indices)
WIREFRAME_CONNECTIONS: list[tuple[int, int]] = [
    # Nose bridge vertical
    (2, 7), (7, 12),
    # Nose bridge horizontal top
    (4, 3), (3, 2), (2, 5), (5, 6),
    # Nose bridge horizontal mid
    (10, 8), (8, 7), (7, 9), (9, 11),
    # Nose tip
    (14, 13), (13, 12), (12, 15), (15, 16),
    (17, 14), (18, 16),
    # Nose wings
    (68, 69), (68, 70), (69, 7),
    # Left eye
    (20, 23), (23, 21), (21, 24), (24, 22), (22, 25), (25, 20), (19, 70),
    # Right eye
    (27, 30), (30, 28), (28, 31), (31, 29), (29, 32), (32, 27), (26, 71),
    # Eyebrows
    (33, 34), (36, 37),
    # Upper lip
    (38, 41), (41, 39), (39, 42), (42, 40), (40, 43),
    # Lower lip
    (44, 47), (47, 45), (45, 48), (48, 46), (46, 49),
    # Connect lips
    (38, 44), (43, 49),
    # Left jaw contour
    (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58),
    # Right jaw contour
    (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67),
    # Connect jaw to chin
    (50, 0), (59, 0),
]


# ══════════════════════════════════════════════
# TPS Parser
# ══════════════════════════════════════════════

def parse_tps(filepath: str) -> list[Specimen]:
    """Parse a TPS file and return a list of Specimen objects."""
    specimens: list[Specimen] = []
    current_landmarks: list[list[float]] = []
    current_image: Optional[str] = None
    current_id: Optional[int] = None

    def _save_current():
        """Save the current specimen if landmarks exist."""
        nonlocal current_landmarks, current_image, current_id
        if current_landmarks:
            specimens.append(Specimen(
                landmarks=np.array(current_landmarks),
                image=current_image,
                id=current_id,
            ))

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('LM='):
                _save_current()
                current_landmarks = []
                current_image = None
                current_id = None
            elif line.startswith('IMAGE='):
                current_image = line.split('=', 1)[1].strip()
            elif line.startswith('ID='):
                current_id = int(line.split('=', 1)[1].strip())
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        current_landmarks.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        pass

    _save_current()  # Save last specimen
    return specimens


# ══════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════

def _draw_wireframe(ax: plt.Axes, landmarks: np.ndarray) -> None:
    """Draw wireframe connections between landmarks."""
    n = len(landmarks)
    for i, j in WIREFRAME_CONNECTIONS:
        if i < n and j < n:
            ax.plot(
                [landmarks[i, 0], landmarks[j, 0]],
                [landmarks[i, 1], landmarks[j, 1]],
                color='#00E5FF', linewidth=0.8, alpha=0.5, zorder=2,
            )


def _draw_landmarks(ax: plt.Axes, landmarks: np.ndarray, point_size: int) -> None:
    """Draw landmarks colored by group, plus any ungrouped landmarks."""
    n = len(landmarks)

    # Grouped landmarks
    for group in LANDMARK_GROUPS:
        indices = [i for i in group.indices if i < n]
        if indices:
            pts = landmarks[indices]
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=point_size, c=group.color, edgecolors='white',
                linewidths=0.5, zorder=3, label=group.name,
            )

    # Ungrouped landmarks
    remaining = [i for i in range(n) if i not in _GROUPED_INDICES]
    if remaining:
        pts = landmarks[remaining]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=point_size, c='#FFFFFF', edgecolors='gray',
            linewidths=0.5, zorder=3, label='Other',
        )


def _draw_labels(ax: plt.Axes, landmarks: np.ndarray) -> None:
    """Draw 1-based landmark number labels."""
    for i, (x, y) in enumerate(landmarks):
        ax.annotate(
            str(i + 1), (x, y),
            textcoords='offset points', xytext=(4, 4),
            fontsize=5, color='yellow', fontweight='bold', zorder=4,
            bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5, lw=0),
        )


def plot_specimen(
    specimen: Specimen,
    tps_dir: str,
    ax: plt.Axes,
    show_labels: bool = True,
    show_wireframe: bool = True,
    show_image: bool = True,
    point_size: int = 30,
) -> None:
    """Plot a single specimen's landmarks on its image."""
    landmarks = specimen.landmarks

    # Load and display background image
    if show_image and specimen.image:
        img_path = os.path.join(tps_dir, specimen.image)
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            h, w = img.shape[:2]
            ax.imshow(img, extent=[0, w, 0, h])
        else:
            print(f"  ⚠ Image not found: {img_path}")

    if show_wireframe:
        _draw_wireframe(ax, landmarks)

    _draw_landmarks(ax, landmarks, point_size)

    if show_labels:
        _draw_labels(ax, landmarks)

    # Title
    parts = []
    if specimen.id is not None:
        parts.append(f"ID: {specimen.id}")
    if specimen.image:
        parts.append(f"Image: {specimen.image}")
    parts.append(f"LM: {specimen.n_landmarks}")
    ax.set_title(' │ '.join(parts), fontsize=11, fontweight='bold',
                 color='#333333', pad=10)
    ax.set_xlabel('X (pixels)', fontsize=9)
    ax.set_ylabel('Y (pixels)', fontsize=9)
    ax.set_aspect('equal')


def _create_legend_handles() -> list[mpatches.Patch]:
    """Create legend handles for all landmark groups."""
    return [mpatches.Patch(color=g.color, label=g.name) for g in LANDMARK_GROUPS]


# ══════════════════════════════════════════════
# Coordinate Table Printer
# ══════════════════════════════════════════════

def print_coordinate_table(specimens: list[tuple[int, Specimen]]) -> None:
    """Print landmark coordinate tables for the given specimens."""
    sep = '=' * 80
    print(f"\n{sep}")
    for idx, specimen in specimens:
        img_name = specimen.image or 'N/A'
        print(f"\n📋 Landmark Coordinates — Specimen {idx} ({img_name})")
        print('-' * 50)
        print(f"{'#':<5} {'X':>12} {'Y':>12}")
        print('-' * 50)
        for i, (x, y) in enumerate(specimen.landmarks):
            print(f"{i+1:<5} {x:>12.2f} {y:>12.2f}")
        print('-' * 50)
        print(f"Total landmarks: {specimen.n_landmarks}")
    print(sep)


# ══════════════════════════════════════════════
# Main Visualization Pipeline
# ══════════════════════════════════════════════

def visualize_tps(
    tps_path: str,
    out_dir: Optional[str] = None,
    specimen_index: Optional[int] = None,
    show_labels: bool = True,
    show_wireframe: bool = True,
    save_output: bool = True,
    show_plot: bool = False,
) -> None:
    """
    Visualize specimens from a TPS file.

    Parameters
    ----------
    tps_path : str
        Path to the TPS file.
    out_dir : str or None
        Directory for output images (default: same as TPS file).
    specimen_index : int or None
        0-based index of a single specimen to visualize (None = all).
    show_labels : bool
        Show landmark number labels.
    show_wireframe : bool
        Draw wireframe connections.
    save_output : bool
        Save output images to disk.
    show_plot : bool
        Display interactive plot window.
    """
    tps_dir = os.path.dirname(os.path.abspath(tps_path))
    specimens = parse_tps(tps_path)

    if not specimens:
        print(f"⚠ No specimens found in {tps_path}")
        return

    # Print summary
    basename = os.path.basename(tps_path)
    print(f"\n╔══════════════════════════════════════════════╗")
    print(f"║  FaceDig TPS Viewer                          ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  File      : {basename:<31}║")
    print(f"║  Specimens : {len(specimens):<31}║")
    for i, sp in enumerate(specimens):
        img = sp.image or 'N/A'
        sid = sp.id if sp.id is not None else 'N/A'
        print(f"║  [{i:>2}] ID={sid}, LM={sp.n_landmarks}, Image={img}")
    print(f"╚══════════════════════════════════════════════╝")

    # Determine which specimens to plot
    if specimen_index is not None:
        if not 0 <= specimen_index < len(specimens):
            print(f"⚠ Specimen index {specimen_index} out of range (0-{len(specimens)-1})")
            return
        to_plot = [(specimen_index, specimens[specimen_index])]
    else:
        to_plot = list(enumerate(specimens))

    # Setup output directory
    save_dir = Path(out_dir) if out_dir else Path(tps_dir)
    if save_output:
        save_dir.mkdir(parents=True, exist_ok=True)

    legend_handles = _create_legend_handles()

    # ── Individual specimen plots ──
    for idx, specimen in to_plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        fig.patch.set_facecolor('#F5F5F5')

        plot_specimen(specimen, tps_dir, ax=ax, show_labels=show_labels,
                      show_wireframe=show_wireframe, point_size=40)

        ax.legend(
            handles=legend_handles, loc='upper left', fontsize=6,
            framealpha=0.8, fancybox=True, ncol=2,
            title='Landmark Groups', title_fontsize=7,
        )
        plt.tight_layout()

        if save_output:
            out_path = save_dir / f"tps_landmarks_specimen_{idx}.png"
            fig.savefig(str(out_path), dpi=200, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"  ✓ Saved: {out_path}")

        if not show_plot:
            plt.close(fig)

    # ── Comparison plot (if multiple specimens) ──
    if len(to_plot) > 1:
        n = len(to_plot)
        fig, axes = plt.subplots(1, n, figsize=(10 * n, 12))
        fig.patch.set_facecolor('#F5F5F5')
        if n == 1:
            axes = [axes]

        for ax_idx, (sp_idx, specimen) in enumerate(to_plot):
            plot_specimen(specimen, tps_dir, ax=axes[ax_idx],
                          show_labels=show_labels, show_wireframe=show_wireframe,
                          point_size=30)

        fig.suptitle(f'FaceDig TPS — Specimen Comparison ({basename})',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_output:
            out_path = save_dir / "tps_landmarks_comparison.png"
            fig.savefig(str(out_path), dpi=200, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            print(f"  ✓ Saved: {out_path}")

        if not show_plot:
            plt.close(fig)

    # ── Print coordinate tables ──
    print_coordinate_table(to_plot)

    if show_plot:
        plt.show()

    print("\n✅ Done!")


# ══════════════════════════════════════════════
# Batch Processing
# ══════════════════════════════════════════════

# Define the batch jobs: (TPS file path relative to base, Output folder path relative to base)
BATCH_JOBS = [
    ('Data/2023/2023_AI.tps', 'Data/2023/Output 2023'),
    ('Data/2024/2024_AI.tps', 'Data/2024/Output 2024'),
    ('Data/2025/2025_AI.tps', 'Data/2025/Output 2025'),
]


def run_batch(base_dir: str, show_labels: bool = True, show_wireframe: bool = True) -> None:
    """Process all TPS files defined in BATCH_JOBS."""
    print("=" * 60)
    print("  🔬 FaceDig TPS Batch Processing")
    print("=" * 60)

    for tps_rel, out_rel in BATCH_JOBS:
        tps_path = os.path.join(base_dir, tps_rel)
        out_dir = os.path.join(base_dir, out_rel)

        if not os.path.exists(tps_path):
            print(f"\n⚠ TPS file not found, skipping: {tps_path}")
            continue

        print(f"\n{'─' * 60}")
        print(f"  📂 Processing: {tps_rel}")
        print(f"  📁 Output to : {out_rel}")
        print(f"{'─' * 60}")

        visualize_tps(
            tps_path=tps_path,
            out_dir=out_dir,
            show_labels=show_labels,
            show_wireframe=show_wireframe,
            save_output=True,
            show_plot=False,
        )

    print(f"\n{'=' * 60}")
    print("  🎉 Batch processing complete!")
    print(f"{'=' * 60}")


# ══════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='FaceDig TPS Viewer — Visualize facial landmarks from TPS files',
    )
    parser.add_argument(
        '--tps', type=str, default=None,
        help='Path to a single TPS file. If omitted, runs batch mode on all years.',
    )
    parser.add_argument(
        '--specimen', type=int, default=None,
        help='Index of specimen to visualize (0-based). Omit to show all.',
    )
    parser.add_argument('--no-labels', action='store_true', help='Hide landmark labels')
    parser.add_argument('--no-wireframe', action='store_true', help='Hide wireframe connections')
    parser.add_argument('--no-save', action='store_true', help='Do not save output images')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory for images')
    parser.add_argument('--no-show', action='store_true', help='Do not show interactive plot')
    parser.add_argument('--batch', action='store_true', help='Force batch mode (all years)')

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.batch or args.tps is None:
        # Batch mode: process all TPS files
        run_batch(
            base_dir=base_dir,
            show_labels=not args.no_labels,
            show_wireframe=not args.no_wireframe,
        )
    else:
        # Single file mode
        if not os.path.exists(args.tps):
            print(f"⚠ TPS file not found: {args.tps}")
            sys.exit(1)

        visualize_tps(
            tps_path=args.tps,
            out_dir=args.out_dir,
            specimen_index=args.specimen,
            show_labels=not args.no_labels,
            show_wireframe=not args.no_wireframe,
            save_output=not args.no_save,
            show_plot=not args.no_show,
        )


if __name__ == '__main__':
    main()
