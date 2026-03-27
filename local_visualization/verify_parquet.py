"""
Verification and visualization script for AutoBrep parquet files.

Checks:
  1. Schema — all required columns present with correct dtypes
  2. Array shapes — matches AutoBrep expected (F,32,32,3), (E,32,3), (F,6), (E,6), (F,E)
  3. Value ranges — points and bboxes in [-1, 1]
  4. Topology — every edge touches exactly 2 faces (manifold), no face with 0 edges
  5. Bbox consistency — bbox matches actual point extents
  6. Normalization — global point cloud fits in [-1, 1]
  7. AutoBrep filter simulation — applies the same pre_filter checks as the real data loader
  8. Visual plots — 3D scatter of decoded face/edge geometry

Usage:
    python verify_parquet.py --parquet /path/to/output.parquet [--n_samples 5] [--plot]
    python verify_parquet.py --parquet /path/to/train/ [--n_samples 5] [--plot]
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Deserialization (mirrors autobrep.data.serialize)
# ─────────────────────────────────────────────────────────────────────────────


def deserialize(blob) -> np.ndarray:
    if isinstance(blob, (bytes, bytearray)):
        buf = io.BytesIO(blob)
    else:
        buf = io.BytesIO(bytes(blob))
    return np.load(buf)


# ─────────────────────────────────────────────────────────────────────────────
# AutoBrep pre_filter logic (same as abc_data.py)
# ─────────────────────────────────────────────────────────────────────────────

BIT = 10
TOL = 1.0 / (2 ** (BIT - 1))  # ≈ 0.00195, minimum visible box size
MAX_EDGE = 1000
MIN_FACE = 2
MAX_FACE = 200


def autobrep_pre_filter(face_bbox, edge_bbox, face_edge_adj):
    """
    Returns (passed: bool, reason: str)
    """
    if face_edge_adj is None or face_edge_adj.size == 0:
        return False, "empty incidence matrix"
    if face_edge_adj.ndim == 0:
        return False, "0-dim incidence matrix"

    # [3] face with no edges
    if np.any(np.all(~face_edge_adj.astype(bool), axis=1)):
        return False, "face with zero edges"

    # [4] non-manifold (edge not shared by exactly 2 faces)
    edge_face_counts = face_edge_adj.astype(int).sum(axis=0)
    if np.any(edge_face_counts != 2):
        n_bad = int(np.sum(edge_face_counts != 2))
        return False, f"non-manifold: {n_bad} edges not shared by exactly 2 faces"

    # [5] too many edges
    if face_edge_adj.shape[1] > MAX_EDGE:
        return False, f"too many edges: {face_edge_adj.shape[1]} > {MAX_EDGE}"

    # [6] tiny face
    face_sizes = np.abs(face_bbox[:, 3:6] - face_bbox[:, 0:3])
    if np.any(np.all(face_sizes < TOL, axis=1)):
        return False, "face bbox too small in all dims"

    # [7] tiny edge
    edge_sizes = np.abs(edge_bbox[:, 3:6] - edge_bbox[:, 0:3])
    if np.any(np.all(edge_sizes < TOL, axis=1)):
        return False, "edge bbox too small in all dims"

    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample verification
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "face_points_normalized",
    "edge_points_normalized",
    "face_bbox_world",
    "edge_bbox_world",
    "face_edge_incidence",
    "num_faces_after_splitting",
    "scaled_unique",
]


def check_sample(row, idx):
    """
    Run all checks on a single parquet row.
    Returns (issues: list[str], stats: dict)
    """
    issues = []
    stats = {}

    # ── Deserialize ──────────────────────────────────────────────────────────
    try:
        face_pts = deserialize(row["face_points_normalized"])
    except Exception as e:
        return [f"[{idx}] Cannot deserialize face_points_normalized: {e}"], {}

    try:
        edge_pts = deserialize(row["edge_points_normalized"])
    except Exception as e:
        return [f"[{idx}] Cannot deserialize edge_points_normalized: {e}"], {}

    try:
        face_bbox = deserialize(row["face_bbox_world"])
    except Exception as e:
        return [f"[{idx}] Cannot deserialize face_bbox_world: {e}"], {}

    try:
        edge_bbox = deserialize(row["edge_bbox_world"])
    except Exception as e:
        return [f"[{idx}] Cannot deserialize edge_bbox_world: {e}"], {}

    try:
        face_edge_adj = deserialize(row["face_edge_incidence"])
    except Exception as e:
        return [f"[{idx}] Cannot deserialize face_edge_incidence: {e}"], {}

    nf = face_pts.shape[0]
    ne = edge_pts.shape[0]

    stats = {
        "num_faces": nf,
        "num_edges": ne,
        "face_pts_shape": face_pts.shape,
        "edge_pts_shape": edge_pts.shape,
        "face_bbox_shape": face_bbox.shape,
        "edge_bbox_shape": edge_bbox.shape,
        "adj_shape": face_edge_adj.shape,
        "face_pts_dtype": str(face_pts.dtype),
        "edge_pts_dtype": str(edge_pts.dtype),
    }

    # ── Shape checks ─────────────────────────────────────────────────────────
    if face_pts.ndim != 4 or face_pts.shape[3] != 3:
        issues.append(f"  face_points shape {face_pts.shape} != (F, 32, 32, 3)")
    elif face_pts.shape[1] != 32 or face_pts.shape[2] != 32:
        issues.append(
            f"  face_points grid size {face_pts.shape[1]}x{face_pts.shape[2]} != 32x32"
        )

    if edge_pts.ndim != 3 or edge_pts.shape[2] != 3:
        issues.append(f"  edge_points shape {edge_pts.shape} != (E, 32, 3)")
    elif edge_pts.shape[1] != 32:
        issues.append(f"  edge_points grid size {edge_pts.shape[1]} != 32")

    if face_bbox.shape != (nf, 6):
        issues.append(f"  face_bbox shape {face_bbox.shape} != ({nf}, 6)")

    if edge_bbox.shape != (ne, 6):
        issues.append(f"  edge_bbox shape {edge_bbox.shape} != ({ne}, 6)")

    if face_edge_adj.shape != (nf, ne):
        issues.append(
            f"  face_edge_incidence shape {face_edge_adj.shape} != ({nf}, {ne})"
        )

    # ── Dtype checks ─────────────────────────────────────────────────────────
    if face_pts.dtype != np.float32:
        issues.append(f"  face_points dtype={face_pts.dtype} (expected float32)")
    if edge_pts.dtype != np.float32:
        issues.append(f"  edge_points dtype={edge_pts.dtype} (expected float32)")
    if face_bbox.dtype != np.float32:
        issues.append(f"  face_bbox dtype={face_bbox.dtype} (expected float32)")
    if edge_bbox.dtype != np.float32:
        issues.append(f"  edge_bbox dtype={edge_bbox.dtype} (expected float32)")

    # ── Value range [-1, 1] ──────────────────────────────────────────────────
    fp_min, fp_max = float(face_pts.min()), float(face_pts.max())
    ep_min, ep_max = float(edge_pts.min()), float(edge_pts.max())
    stats["face_pts_range"] = (fp_min, fp_max)
    stats["edge_pts_range"] = (ep_min, ep_max)

    if fp_min < -1.01 or fp_max > 1.01:
        issues.append(
            f"  face_points out of [-1,1]: min={fp_min:.4f}, max={fp_max:.4f}"
        )
    if ep_min < -1.01 or ep_max > 1.01:
        issues.append(
            f"  edge_points out of [-1,1]: min={ep_min:.4f}, max={ep_max:.4f}"
        )

    fb_min, fb_max = float(face_bbox.min()), float(face_bbox.max())
    eb_min, eb_max = float(edge_bbox.min()), float(edge_bbox.max())
    if fb_min < -1.01 or fb_max > 1.01:
        issues.append(f"  face_bbox out of [-1,1]: min={fb_min:.4f}, max={fb_max:.4f}")
    if eb_min < -1.01 or eb_max > 1.01:
        issues.append(f"  edge_bbox out of [-1,1]: min={eb_min:.4f}, max={eb_max:.4f}")

    # ── Bbox vs actual points consistency ────────────────────────────────────
    for fi in range(nf):
        pts = face_pts[fi].reshape(-1, 3)
        actual_min = pts.min(axis=0)
        actual_max = pts.max(axis=0)
        stored_min = face_bbox[fi, :3]
        stored_max = face_bbox[fi, 3:]
        if np.any(np.abs(actual_min - stored_min) > 1e-3) or np.any(
            np.abs(actual_max - stored_max) > 1e-3
        ):
            issues.append(
                f"  face {fi} bbox mismatch: stored [{stored_min}..{stored_max}] vs actual [{actual_min}..{actual_max}]"
            )
            break  # report first occurrence only

    # ── Global normalization — combined cloud should be in [-1, 1] ───────────
    all_pts = np.concatenate([face_pts.reshape(-1, 3), edge_pts.reshape(-1, 3)], axis=0)
    global_min = all_pts.min()
    global_max = all_pts.max()
    stats["global_range"] = (float(global_min), float(global_max))
    if global_min < -1.01 or global_max > 1.01:
        issues.append(
            f"  global point cloud [{global_min:.4f}, {global_max:.4f}] not in [-1,1]"
        )

    # ── AutoBrep pre_filter ───────────────────────────────────────────────────
    passed, reason = autobrep_pre_filter(face_bbox, edge_bbox, face_edge_adj)
    stats["autobrep_filter"] = (passed, reason)
    if not passed:
        issues.append(f"  AutoBrep pre_filter FAIL: {reason}")

    # ── Scalar fields ─────────────────────────────────────────────────────────
    n_faces_meta = row.get("num_faces_after_splitting", None)
    if n_faces_meta is not None and int(n_faces_meta) != nf:
        issues.append(
            f"  num_faces_after_splitting={n_faces_meta} != actual faces={nf}"
        )

    scaled = row.get("scaled_unique", None)
    if scaled is not None and not bool(scaled):
        issues.append(
            "  scaled_unique=False (will be filtered out by AutoBrep data loader)"
        )

    return issues, stats


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


def plot_bbox_edges(ax, bbox, color, label=None, linewidth=0.5, alpha=0.4):
    """Plot AABB wireframe edges."""
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ]
    )
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # bottom
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # top
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # vertical
    ]
    for i, e in enumerate(edges):
        pts = corners[e]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label if i == 0 else None,
        )


def plot_sample(row, sample_idx, title=""):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        face_pts = deserialize(row["face_points_normalized"])  # (F, 32, 32, 3)
        edge_pts = deserialize(row["edge_points_normalized"])  # (E, 32, 3)
        face_bbox = deserialize(row["face_bbox_world"])  # (F, 6)
        edge_bbox = deserialize(row["edge_bbox_world"])  # (E, 6)

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"Sample {sample_idx}  {title}", fontsize=14, fontweight="bold")
        ax = fig.add_subplot(111, projection="3d")

        # Colormap for faces
        cmap = plt.get_cmap("tab20")

        # ── Plot face point clouds (colored, all points) ────────────────────
        for fi in range(len(face_pts)):
            pts = face_pts[fi].reshape(-1, 3)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=[cmap(fi % 20)],
                s=2,
                alpha=0.7,
                label=None,
            )

        # ── Plot edge points as crosses (purple, all points) ────────────────────
        if len(edge_pts) > 0:
            edge_pts_flat = edge_pts.reshape(-1, 3)
            ax.scatter(
                edge_pts_flat[:, 0],
                edge_pts_flat[:, 1],
                edge_pts_flat[:, 2],
                c="purple",
                s=4,
                alpha=0.6,
                marker="x",
                label=f"Edge points ({len(edge_pts)} edges)",
            )

        # ── Plot face bounding boxes (green wireframe) ────────────────────
        if len(face_bbox) > 0:
            for fi, bbox in enumerate(face_bbox):
                plot_bbox_edges(
                    ax,
                    bbox,
                    color="green",
                    label="Face bboxes" if fi == 0 else None,
                    linewidth=0.4,
                    alpha=0.3,
                )

        # ── Plot edge bounding boxes (red wireframe) ─────────────────────
        if len(edge_bbox) > 0:
            for ei, bbox in enumerate(edge_bbox):
                plot_bbox_edges(
                    ax,
                    bbox,
                    color="red",
                    label="Edge bboxes" if ei == 0 else None,
                    linewidth=0.4,
                    alpha=0.3,
                )

        # ── Axes setup ───────────────────────────────────────────────────
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_zlabel("Z", fontsize=10)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        # ── Legend ───────────────────────────────────────────────────────
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(0),
                markersize=6,
                label=f"Face points ({len(face_pts)} faces)",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="x",
                color="purple",
                markersize=10,
                linestyle="None",
                label=f"Edge points ({len(edge_pts)} edges)",
            ),
            plt.Line2D(
                [0],
                [0],
                color="green",
                linewidth=2,
                alpha=0.6,
                label=f"Face bboxes ({len(face_bbox)})",
            ),
            plt.Line2D(
                [0],
                [0],
                color="red",
                linewidth=2,
                alpha=0.6,
                label=f"Edge bboxes ({len(edge_bbox)})",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=11,
            framealpha=0.95,
            fancybox=True,
            shadow=True,
        )

        ax.set_title(
            f"{len(face_pts)} faces × {len(edge_pts)} edges", fontsize=10, pad=10
        )

        plt.tight_layout()
        out_path = Path(f"parquet_sample_{sample_idx:03d}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Plot saved: {out_path}")
        plt.close(fig)

    except ImportError:
        print("  matplotlib not available, skipping plots")
    except Exception as e:
        print(f"  Plot error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level statistics
# ─────────────────────────────────────────────────────────────────────────────


def print_dataset_stats(df):
    print("\n" + "=" * 60)
    print(f"  Dataset: {len(df)} rows")
    print(f"  Columns: {sorted(df.columns.tolist())}")

    # Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"  MISSING REQUIRED COLUMNS: {missing}")
    else:
        print("  ✅ All required columns present")

    # Scalar column stats
    if "num_faces_after_splitting" in df.columns:
        nf_vals = df["num_faces_after_splitting"].dropna()
        print(
            f"\n  num_faces_after_splitting: min={nf_vals.min()}, max={nf_vals.max()}, "
            f"mean={nf_vals.mean():.1f}, median={nf_vals.median():.0f}"
        )
        n_too_many = (nf_vals > MAX_FACE).sum()
        n_too_few = (nf_vals < MIN_FACE).sum()
        if n_too_many:
            print(
                f"    ⚠️  {n_too_many} rows exceed max_face={MAX_FACE} (will be filtered)"
            )
        if n_too_few:
            print(
                f"    ⚠️  {n_too_few} rows below min_face={MIN_FACE} (will be filtered)"
            )

    if "scaled_unique" in df.columns:
        n_true = df["scaled_unique"].sum()
        n_false = len(df) - n_true
        print(f"\n  scaled_unique: {n_true} True / {n_false} False")
        if n_false:
            print(
                f"    ⚠️  {n_false} rows have scaled_unique=False (will be filtered by AutoBrep)"
            )

    if "filename" in df.columns:
        print(f"\n  Filenames (first 5): {df['filename'].head(5).tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Verify AutoBrep parquet files")
    parser.add_argument(
        "--parquet",
        type=str,
        required=True,
        help="Path to a .parquet file or a directory containing parquet files",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples to verify in detail (default: 10)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save 3D scatter plots for each verified sample",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet)

    if parquet_path.is_dir():
        files = sorted(parquet_path.rglob("*.parquet"))
        print(f"Found {len(files)} parquet files in {parquet_path}")
        df = pd.read_parquet(parquet_path, engine="pyarrow")
    elif parquet_path.is_file():
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        files = [parquet_path]
    else:
        print(f"ERROR: {parquet_path} does not exist")
        sys.exit(1)

    print_dataset_stats(df)

    # Sample selection — spread across the dataframe
    n = min(args.n_samples, len(df))
    indices = np.linspace(0, len(df) - 1, n, dtype=int)

    print(f"\n{'=' * 60}")
    print(f"  Checking {n} samples in detail...")
    print(f"{'=' * 60}")

    n_pass = 0
    n_fail = 0

    for i, row_idx in enumerate(indices):
        row = df.iloc[row_idx]
        fname = (
            row.get("filename", f"row_{row_idx}")
            if "filename" in df.columns
            else f"row_{row_idx}"
        )

        issues, stats = check_sample(row, row_idx)

        status = "✅" if not issues else "❌"
        nf = stats.get("num_faces", "?")
        ne = stats.get("num_edges", "?")
        f_range = stats.get("face_pts_range", ("?", "?"))
        e_range = stats.get("edge_pts_range", ("?", "?"))
        filt = stats.get("autobrep_filter", (None, ""))

        print(
            f"\n  [{i + 1}/{n}] {status} {fname} | faces={nf}, edges={ne} | "
            f"pts_range=[{f_range[0]:.3f}, {f_range[1]:.3f}] | filter={filt[1]}"
        )

        if stats.get("face_pts_shape"):
            print(
                f"         shapes: face_pts{stats['face_pts_shape']}, "
                f"edge_pts{stats['edge_pts_shape']}, "
                f"adj{stats['adj_shape']}"
            )

        for issue in issues:
            print(f"         ⚠️  {issue}")

        if not issues:
            n_pass += 1
        else:
            n_fail += 1

        if args.plot:
            title = fname if isinstance(fname, str) else ""
            plot_sample(row, row_idx, title=title)

    print(f"\n{'=' * 60}")
    print(f"  Results: {n_pass}/{n} passed, {n_fail}/{n} failed")
    print(f"{'=' * 60}")

    # ── Summary of what AutoBrep will actually see ────────────────────────
    print("\n  AutoBrep data loader simulation:")
    print(f"  - min_face={MIN_FACE}, max_face={MAX_FACE}, max_edge={MAX_EDGE}")
    print("  - scaled_unique filter: expects True")
    print(f"  - bit={BIT} → TOL={TOL:.5f} (minimum bbox size)")
    print("  - 90% of samples with <25 faces are dropped during training (normal!)")
    print(
        "  - Valid rows available to loader won't be visible until actual training run\n"
    )


if __name__ == "__main__":
    main()
