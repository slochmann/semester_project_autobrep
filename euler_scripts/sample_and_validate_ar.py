"""
Sample + validate AutoBrep base model on a fresh HF ABC-1M sample.

Steps performed (no training):
    1. Load base checkpoints (ar.ckpt, surf-fsq.ckpt, edge-fsq.ckpt)
    2. Sample one unconditional batch and visualize
    3. Stream one sample from HuggingFace (ADSKAILab/ABC-1M)
    4. Visualize the real sample (reconstruct via AutoBrepBuilder)
    5. Tokenize the sample with ARDataModule.map_func + encode_fsq_code
    6. Compute teacher-forced perplexity on the full sequence
    7. Take first 50% of tokens as prompt, autocomplete the rest
    8. Visualize the autocomplete result
"""

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from autobrep.data.abc_data import ARDataModule
from autobrep.data.token_mapping import MMTokenIndex
from autobrep.inference.brepgen_brep_builder import AutoBrepBuilder
from autobrep.inference.inference_common import (
    reconstruct_compound,
    save_debug_images,
)
from autobrep.models.autoregressive import AutoBrepModel, AutoRegressiveSampler
from autobrep.models.vaes import EdgeFSQVAE, SurfaceFSQVAE
from datasets import load_dataset
from einops import rearrange
from occwl.io import save_step as save_step_func
from PIL import Image

job_name = os.environ.get("SLURM_JOB_NAME", "local_run")


def parse_args():
    p = argparse.ArgumentParser(description="Sample + validate AutoBrep on one HF sample")
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--hf_repo", type=str, default="ADSKAILab/ABC-1M")
    p.add_argument("--num_train_samples", type=int, default=1,
                   help="How many samples to draw from the HF train split")
    p.add_argument("--num_val_samples", type=int, default=1,
                   help="How many samples to draw from the HF validation split")
    p.add_argument("--hf_sample_skip", type=int, default=0,
                   help="Skip N rows in each stream before taking samples")

    # Unconditional sampling — defaults follow configs/sample.json from the
    # AutoBrep repo (T=1.0, top_p=0.9, complexity=hard).
    p.add_argument("--sample_temperature", type=float, default=1.0)
    p.add_argument("--sample_threshold", type=float, default=0.9)
    p.add_argument("--sample_complexity", type=int, default=16,
                   help="14=easy, 15=medium, 16=hard, 17=random")

    # Sequence completion (NOT the paper's B-Rep autocomplete — see notes
    # in complete_sequence_from_prefix below: geom_tokens were commented out
    # of full_seq during base-model pretraining in abc_data.py:811).
    p.add_argument("--prefix_ratio", type=float, default=0.5,
                   help="Fraction of the real sequence to feed as prompt")
    p.add_argument("--num_completions", type=int, default=4,
                   help="How many independent completions to sample from the same prefix")

    # Reconstruction tolerances — defaults follow configs/sample.json.
    p.add_argument("--z_threshold", type=float, default=0.0)
    p.add_argument("--vertex_threshold", type=float, default=0.002)
    p.add_argument("--sewing_tolerance", type=float, default=0.002)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers shared with train_lora_ar.py
# ---------------------------------------------------------------------------

def decode_and_reconstruct(
    sample_seq,
    model,
    surface_fsq,
    edge_fsq,
    device,
    out_dir: Path,
    stem: str,
    args,
):
    """
    Decode a single generated/teacher-forced token sequence (1D np.ndarray) into a
    CAD body, save debug PNGs + STEP file, return success bool.
    """
    out_dir.mkdir(exist_ok=True, parents=True)

    geom_tokens = None
    if MMTokenIndex.BOGEOM.value in sample_seq:
        gs = np.where(sample_seq == MMTokenIndex.BOGEOM.value)[0][0]
        ge = np.where(sample_seq == MMTokenIndex.EOGEOM.value)[0][0]
        geom_tokens = sample_seq[gs + 1:ge]

    if MMTokenIndex.BOC.value not in sample_seq:
        print(f"  ⚠️ {stem}: no BOC token in sequence")
        return False
    cs = np.where(sample_seq == MMTokenIndex.BOC.value)[0][0]
    if MMTokenIndex.EOC.value not in sample_seq:
        print(f"  ⚠️ {stem}: no EOC token in sequence")
        return False
    ce = np.where(sample_seq == MMTokenIndex.EOC.value)[0][0]
    cad_tokens = sample_seq[cs + 1:ce]
    if geom_tokens is not None:
        cad_tokens = np.concatenate((geom_tokens, cad_tokens))

    try:
        pos_faces, code_faces, pos_edges, code_edges, face_edge_adj = model.decode(cad_tokens)
    except Exception as e:
        print(f"  ⚠️ {stem}: decode failed: {e}")
        return False

    with torch.no_grad():
        gz_f = surface_fsq.quantizer.indices_to_codes(
            torch.LongTensor(code_faces).to(device)
        ).permute(0, 2, 1)
        uv_f = surface_fsq.decode(gz_f.unflatten(-1, (2, 2))).sample
        uv_f = rearrange(uv_f, "b d ... -> b ... d").float().cpu().numpy()

        gz_e = edge_fsq.quantizer.indices_to_codes(
            torch.LongTensor(code_edges).to(device)
        ).permute(0, 2, 1)
        uv_e = edge_fsq.decode(gz_e).sample
        uv_e = rearrange(uv_e, "b d ... -> b ... d").float().cpu().numpy()

    batch_decoded = [(pos_faces, pos_edges, uv_e, uv_f, face_edge_adj)]
    batch_cad_data = AutoRegressiveSampler.convert_to_cad_data(batch_decoded)
    builders = [AutoBrepBuilder(
        device=device,
        z_threshold=args.z_threshold,
        vertex_threshold=args.vertex_threshold,
        sewing_tolerance=args.sewing_tolerance,
    )]

    cad_data = batch_cad_data[0]
    face_img = out_dir / f"{stem}_face.png"
    edge_img = out_dir / f"{stem}_edge.png"
    try:
        save_debug_images(cad_data, face_img, edge_img)
    except Exception as e:
        print(f"  ⚠️ {stem}: save_debug_images failed: {e}")

    try:
        result = reconstruct_compound(cad_data, builders)
        if result is not None:
            save_step_func([result], out_dir / f"{stem}.step")
            print(f"  ✅ {stem}: reconstructed and saved")
            return True
        print(f"  ⚠️ {stem}: reconstruction returned None")
    except Exception as e:
        print(f"  ⚠️ {stem}: reconstruct failed: {e}")
    return False




# ---------------------------------------------------------------------------
# Step 3 + 5: Download one HF sample, tokenize via ARDataModule
# ---------------------------------------------------------------------------

def iter_hf_samples(repo: str, split: str, n: int, skip: int, data_module: ARDataModule):
    """
    Yield up to `n` raw HF rows (after skipping the first `skip`) that pass
    ARDataModule.pre_filter. Logs progress.
    """
    print(f"\n📥 Streaming {repo} split={split} (need {n}, skip {skip}) ...")
    ds = load_dataset(repo, split=split, streaming=True)
    it = iter(ds)
    for _ in range(skip):
        next(it)
    seen, yielded = 0, 0
    for row in it:
        if yielded >= n:
            return
        seen += 1
        if not row.get("scaled_unique", False):
            continue
        nf = row.get("num_faces_after_splitting", 0)
        if nf < data_module.hparams.min_face or nf > data_module.hparams.max_face:
            continue
        try:
            if not data_module.pre_filter(row):
                continue
        except Exception as e:
            print(f"  pre_filter error: {e}")
            continue
        yielded += 1
        print(f"  ✅ [{split} {yielded}/{n}] after streaming {seen} rows: "
              f"stem={row.get('stem')!r}, num_faces={nf}")
        yield row
    if yielded < n:
        print(f"  ⚠️ stream exhausted with only {yielded}/{n} accepted samples")


def save_original_step_from_brep(row, out_path: Path) -> bool:
    """
    The HF row's `bytes` field stores the OCC ASCII BREP for the (pre-splitting)
    body. Write it to a temp .brep file, read with OCC, save STEP.
    """
    import tempfile
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepTools import breptools_Read
    from OCC.Core.TopoDS import TopoDS_Shape

    raw = row.get("bytes")
    if raw is None:
        print("  ⚠️ row has no 'bytes' field; cannot save original STEP")
        return False
    if isinstance(raw, str):
        raw = raw.encode()

    with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as f:
        f.write(raw)
        brep_path = f.name

    try:
        shape = TopoDS_Shape()
        builder = BRep_Builder()
        ok = breptools_Read(shape, brep_path, builder)
        if not ok or shape.IsNull():
            print(f"  ⚠️ BRepTools_Read failed for {out_path.name}")
            return False
        save_step_func([shape], out_path)
        print(f"  💾 saved original STEP → {out_path}")
        return True
    except Exception as e:
        print(f"  ⚠️ original STEP save failed: {e}")
        return False
    finally:
        try:
            os.unlink(brep_path)
        except OSError:
            pass


def tokenize_hf_sample(row, data_module: ARDataModule):
    """Run unpickle + map_func without augmentation. Returns dict with seq/face_ncs/edge_ncs."""
    row_unpickled = data_module.unpickle(row)
    if not data_module.post_filter(row_unpickled):
        raise RuntimeError("Sample failed post_filter")
    return data_module.map_func(row_unpickled, aug=False)


# ---------------------------------------------------------------------------
# Step 6: teacher-forced perplexity / loss
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenized, device):
    """Mirror train_lora_ar.validate() for a single sample. Returns (loss, perplexity)."""
    token = torch.from_numpy(np.asarray(tokenized["seq"])).long().unsqueeze(0).to(device)
    face_ncs = torch.from_numpy(np.asarray(tokenized["face_ncs"])).unsqueeze(0).to(device, dtype=torch.bfloat16)
    edge_ncs = torch.from_numpy(np.asarray(tokenized["edge_ncs"])).unsqueeze(0).to(device, dtype=torch.bfloat16)

    surf_id, edge_id = model.encode_fsq_code(face_ncs, edge_ncs)

    updated = []
    for _tok, _sid, _eid in zip(token, surf_id, edge_id):
        bd = model.copy_fsq_code(_tok, _sid, _eid)
        bd = torch.nn.functional.pad(bd, (0, model.pad_len - len(bd)), value=-1)
        updated.append(bd)
    updated_tokens = torch.stack(updated).detach()

    cond_mask = torch.zeros(updated_tokens.shape[0], model.pad_len,
                            dtype=torch.bool, device=device)
    cond_mask[:, :4] = True

    with torch.no_grad():
        loss = model.cad_gpt(updated_tokens, cond_mask=cond_mask)
    loss_val = loss.item()
    return loss_val, float(np.exp(loss_val)), updated_tokens


# ---------------------------------------------------------------------------
# Step 7: SEQUENCE COMPLETION from first prefix_ratio of the real sequence
#
# This is NOT the paper's "B-Rep autocomplete." That feature relies on
# `geom_tokens` (user-specified faces) being prepended to the token stream
# during training. In abc_data.py:811 the line `# + geom_tokens` is
# commented out, so the base model has only ever seen sequences of shape
# [BOS, BOM, complexity, EOM, BOC, ...cad_tokens..., EOC, EOS]. Here we
# just feed the first `prefix_ratio` of a real tokenized sample and let
# the AR transformer continue — no mechanism tells the model "preserve
# these faces exactly", so divergent (but plausible) completions are
# expected.
# ---------------------------------------------------------------------------

def cut_prefix_at_level_boundary(seq: np.ndarray, ratio: float):
    """
    Return (prefix_tokens, cut_position).

    The AR sequence is `[BOS, BOM, c, EOM, BOC, BOL ... EOL, BOL ... EOL, ..., EOC, EOS]`
    where each BOL/EOL pair wraps one BFS level (see abc_data.py:428-513).
    `model.decode()` walks the sequence level-by-level, so a token in the
    middle of a level is meaningless on its own. The safe cut is *after*
    a complete EOL — that gives the model a prompt ending on a fully formed
    BFS level, and lets us also decode the prefix into a valid partial body
    (after appending EOC) for visualization.
    """
    target = int(len(seq) * ratio)
    eol_positions = np.where(seq == MMTokenIndex.EOL.value)[0]
    if len(eol_positions) == 0:
        return seq[:target], target  # degenerate; downstream decode may fail
    valid = eol_positions[eol_positions <= target]
    cut = (valid[-1] + 1) if len(valid) > 0 else (eol_positions[0] + 1)
    return seq[:cut], int(cut)


def complete_sequence_from_prefix(model, surface_fsq, edge_fsq, updated_tokens,
                                  args, device, output_dir: Path):
    print("\n" + "=" * 80)
    print(f"🪡 SEQUENCE COMPLETION — prefix_ratio={args.prefix_ratio}, "
          f"num_completions={args.num_completions}")
    print("=" * 80)
    sub_dir = output_dir / "sequence_completion"
    sub_dir.mkdir(exist_ok=True, parents=True)

    # Trim padding (-1) — keep only the valid tokens of the real sequence.
    seq = updated_tokens[0]
    valid_len = int((seq >= 0).sum().item())
    valid_seq = seq[:valid_len].cpu().numpy()

    # Ground-truth reconstruction (the full real sequence).
    decode_and_reconstruct(valid_seq, model, surface_fsq, edge_fsq, device,
                           sub_dir, "ground_truth", args)

    # Cut the prompt at a BFS-level boundary. See cut_prefix_at_level_boundary.
    prefix_np, cut_pos = cut_prefix_at_level_boundary(valid_seq, args.prefix_ratio)
    actual_ratio = cut_pos / max(valid_len, 1)
    print(f"  valid_len={valid_len}, requested cut at {int(valid_len*args.prefix_ratio)}, "
          f"snapped to EOL at {cut_pos} (actual ratio {actual_ratio:.2%})")

    # Visualize the prompt itself as a partial body by appending EOC so
    # decode_and_reconstruct() can parse it like a complete CAD.
    prompt_for_viz = np.concatenate(
        [prefix_np, np.array([MMTokenIndex.EOC.value], dtype=prefix_np.dtype)]
    )
    decode_and_reconstruct(prompt_for_viz, model, surface_fsq, edge_fsq, device,
                           sub_dir, "prompt_partial", args)

    # Actual prompt for the model (no trailing EOC — we want the model to
    # generate the closing levels + EOC itself), batched into num_completions
    # independent samples in one .generate() call.
    prefix = torch.from_numpy(prefix_np).long().to(device)
    prompt = prefix.unsqueeze(0).expand(args.num_completions, -1).contiguous()
    with torch.no_grad():
        generated = model.generate(prompt, args.sample_temperature, args.sample_threshold)
    full_seqs = torch.concat([prompt, generated], dim=-1).cpu().numpy()

    pref_pct = int(args.prefix_ratio * 100)
    for i, full_seq in enumerate(full_seqs):
        decode_and_reconstruct(full_seq, model, surface_fsq, edge_fsq, device,
                               sub_dir, f"completion_p{pref_pct}_run{i:03d}", args)


# ---------------------------------------------------------------------------
# Visualize the original (ground-truth) HF sample using its decoded geometry
# ---------------------------------------------------------------------------

def visualize_real_sample(device, sample_dir: Path, args, real_row_unpickled):
    """
    Build CAD data directly from the real face_ncs / edge_ncs / bbox / adjacency
    (i.e. ground-truth POST-SPLIT geometry, no model involvement) and save STEP
    + PNGs as `real_*` inside `sample_dir`.
    """
    print("📷 Visualizing real HF sample (post-splitting reconstruction)")
    sample_dir.mkdir(exist_ok=True, parents=True)

    face_ncs = real_row_unpickled["face_points_normalized"]
    edge_ncs = real_row_unpickled["edge_points_normalized"]
    face_pos = real_row_unpickled["face_bbox_world"]
    edge_pos = real_row_unpickled["edge_bbox_world"]
    face_edge_adj = real_row_unpickled["face_edge_incidence"].astype(bool)

    batch_decoded = [(face_pos, edge_pos, edge_ncs, face_ncs, face_edge_adj)]
    batch_cad_data = AutoRegressiveSampler.convert_to_cad_data(batch_decoded)
    cad_data = batch_cad_data[0]

    try:
        save_debug_images(cad_data, sample_dir / "real_face.png", sample_dir / "real_edge.png")
    except Exception as e:
        print(f"  ⚠️ save_debug_images failed: {e}")

    builders = [AutoBrepBuilder(
        device=device,
        z_threshold=args.z_threshold,
        vertex_threshold=args.vertex_threshold,
        sewing_tolerance=args.sewing_tolerance,
    )]
    try:
        result = reconstruct_compound(cad_data, builders)
        if result is not None:
            save_step_func([result], sample_dir / "real_reconstructed.step")
            print("  ✅ Real sample reconstructed and saved")
    except Exception as e:
        print(f"  ⚠️ reconstruct failed: {e}")


def process_one_sample(row, model, surface_fsq, edge_fsq, data_module,
                       args, device, sample_dir: Path) -> dict:
    """
    For one raw HF row: save the original STEP from `bytes`, the post-split
    ground-truth reconstruction, tokenize, compute teacher-forced perplexity,
    and run --num_completions sequence completions. Returns a metrics dict.
    """
    sample_dir.mkdir(exist_ok=True, parents=True)
    stem = row.get("stem", sample_dir.name)
    print(f"\n{'-' * 80}\n📦 Sample {sample_dir.name} (stem={stem})\n{'-' * 80}")

    save_original_step_from_brep(row, sample_dir / "original.step")

    row_unpickled = data_module.unpickle(row)
    visualize_real_sample(device, sample_dir, args, row_unpickled)

    tokenized = data_module.map_func(row_unpickled, aug=False)
    valid_len = int((np.asarray(tokenized["seq"]) >= 0).sum())
    print(f"  seq length: valid={valid_len} (padded to {len(tokenized['seq'])})")

    loss, ppl, updated_tokens = compute_perplexity(model, tokenized, device)
    print(f"  loss = {loss:.4f}, perplexity = {ppl:.4f}")

    complete_sequence_from_prefix(model, surface_fsq, edge_fsq, updated_tokens,
                                  args, device, sample_dir)

    return {
        "stem": stem,
        "num_faces": row.get("num_faces"),
        "num_faces_after_splitting": row.get("num_faces_after_splitting"),
        "valid_seq_len": valid_len,
        "loss": loss,
        "perplexity": ppl,
    }


# ---------------------------------------------------------------------------
# Collages
# ---------------------------------------------------------------------------

def _render_via_occ_viewer(step_path: Path, out_path: Path, size: int) -> bool:
    """
    Primary path: OCC's own offscreen Viewer3d. Produces shaded renders with
    proper lighting/anti-aliasing/depth — matching the quality you get from
    desktop pythonocc viewers, but without an X display. Mirrors the demo at
    https://github.com/tpaviot/pythonocc-demos/blob/master/examples/core_offscreen_rendering.py
    """
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Display.OCCViewer import Viewer3d
    except Exception as e:
        print(f"  ⚠️ OCC viewer not available: {e}")
        return False

    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != 1:
            return False
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            return False

        v = Viewer3d()
        v.Create()                                  # create offscreen GL context
        v.SetModeShaded()
        v.SetSize(size, size)
        try:
            v.EnableAntiAliasing()
        except Exception:
            pass
        try:
            v.set_bg_gradient_color([255, 255, 255], [220, 230, 245])
        except Exception:
            pass

        v.DisplayShape(shape, update=False, color="BLUE")
        try:
            v.View.FitAll()
        except Exception:
            pass
        try:
            from OCC.Core.V3d import V3d_XposYposZpos
            v.View.SetProj(V3d_XposYposZpos)
            v.View.FitAll()
        except Exception:
            pass
        try:
            v.View.Redraw()
        except Exception:
            pass

        out_path.parent.mkdir(exist_ok=True, parents=True)
        # View.Dump infers file format from extension (.png/.jpeg/.bmp).
        v.View.Dump(str(out_path))
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        print(f"  ⚠️ OCC offscreen render failed for {step_path.name}: {e}")
        return False


def _render_via_matplotlib(step_path: Path, out_path: Path, size: int) -> bool:
    """
    Fallback path if OCC's offscreen viewer isn't usable (no Mesa/EGL on the
    node). Lambertian-shaded triangle mesh via matplotlib's Agg backend — same
    code as before, just isolated so it can serve as fallback only.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods

    fig = None
    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != 1:
            return False
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            return False

        meshed = False
        for linear, angular in [(0.02, 0.2), (0.05, 0.35), (0.1, 0.5), (0.25, 0.8)]:
            try:
                BRepMesh_IncrementalMesh(shape, linear, False, angular, True)
                meshed = True
                break
            except Exception:
                continue
        if not meshed:
            return False

        all_verts, all_tris, offset = [], [], 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            try:
                face = topods.Face(explorer.Current())
                loc = TopLoc_Location()
                tri = BRep_Tool.Triangulation(face, loc)
                if tri is not None and tri.NbNodes() > 0 and tri.NbTriangles() > 0:
                    trsf = loc.Transformation()
                    verts = np.array([
                        [n.X(), n.Y(), n.Z()]
                        for n in (tri.Node(i).Transformed(trsf)
                                  for i in range(1, tri.NbNodes() + 1))
                    ])
                    tris = np.array([
                        list(tri.Triangle(i).Get())
                        for i in range(1, tri.NbTriangles() + 1)
                    ]) - 1 + offset
                    if len(verts) > 0 and len(tris) > 0:
                        all_verts.append(verts)
                        all_tris.append(tris)
                        offset += len(verts)
            except Exception:
                pass
            explorer.Next()

        if not all_verts:
            return False
        verts = np.concatenate(all_verts, axis=0)
        tris = np.concatenate(all_tris, axis=0)
        if len(tris) == 0:
            return False

        v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
        norm_len[norm_len == 0] = 1
        normals = normals / norm_len
        light = np.array([0.6, -0.4, 0.7]) / np.linalg.norm([0.6, -0.4, 0.7])
        intensity = np.abs(normals @ light)
        shade = 0.35 + 0.65 * intensity
        base = np.array([0.36, 0.55, 0.78])
        face_colors = np.clip(shade[:, None] * base[None, :], 0, 1)
        face_colors = np.concatenate([face_colors, np.ones((len(face_colors), 1))], axis=1)
        depth_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        depths = ((v0 + v1 + v2) / 3.0) @ depth_axis
        order = np.argsort(depths)
        tris_sorted = tris[order]
        face_colors = face_colors[order]

        fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("white")
        poly = Poly3DCollection(
            verts[tris_sorted], facecolors=face_colors, edgecolor="none",
            linewidth=0, antialiased=True, shade=False,
        )
        ax.add_collection3d(poly)
        mins, maxs = verts.min(0), verts.max(0)
        ctr = (mins + maxs) / 2.0
        r = float((maxs - mins).max()) / 2.0 * 1.1 or 1.0
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        ax.view_init(elev=25, azim=45)
        ax.set_axis_off()
        out_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        return True
    except Exception as e:
        print(f"  ⚠️ matplotlib render failed for {step_path.name}: {e}")
        return False
    finally:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass


# Whether the OCC offscreen path has worked this process. None = untested,
# True/False = remember the result so we don't keep retrying after a failure.
_OCC_OFFSCREEN_OK = None


def render_step_to_png(step_path: Path, out_path: Path, size: int = 512) -> bool:
    """
    Try OCC's native offscreen renderer (better lighting/shading) first. If
    that path isn't usable on the node (no Mesa/EGL), remember that and use
    the matplotlib Lambertian fallback for every subsequent render.
    """
    global _OCC_OFFSCREEN_OK
    if not step_path.exists():
        return False

    if _OCC_OFFSCREEN_OK is not False:  # untested or known good → try OCC
        if _render_via_occ_viewer(step_path, out_path, size):
            _OCC_OFFSCREEN_OK = True
            return True
        if _OCC_OFFSCREEN_OK is None:
            print("  ↪️  falling back to matplotlib renderer for the rest of this run")
            _OCC_OFFSCREEN_OK = False

    return _render_via_matplotlib(step_path, out_path, size)



def _placeholder_image(size: tuple) -> Image.Image:
    img = Image.new("RGB", size, color=(230, 230, 230))
    return img


def build_collage_for_split(split_dir: Path, num_completions: int,
                            mode: str, out_path: Path,
                            prefix_ratio: float, cell_size: int = 400) -> bool:
    """
    Build a grid collage with one row per sample under `split_dir`. Columns
    are: ground_truth | prompt_partial | completion_0 .. completion_{K-1}.

    mode="faces": load existing <stem>_face.png from sequence_completion/.
    mode="steps": render <stem>.step via render_step_to_png() into temp PNGs.
    """
    assert mode in ("faces", "steps")
    sample_dirs = sorted([d for d in split_dir.iterdir()
                          if d.is_dir() and d.name.startswith("sample_")])
    if not sample_dirs:
        print(f"  ⚠️ no samples under {split_dir}, skipping collage")
        return False

    pref_pct = int(prefix_ratio * 100)
    col_stems = ["ground_truth", "prompt_partial"] + [
        f"completion_p{pref_pct}_run{i:03d}" for i in range(num_completions)
    ]
    n_rows, n_cols = len(sample_dirs), len(col_stems)
    canvas = Image.new("RGB", (cell_size * n_cols, cell_size * n_rows), "white")

    for row_idx, sd in enumerate(sample_dirs):
        sc_dir = sd / "sequence_completion"
        for col_idx, stem in enumerate(col_stems):
            if mode == "faces":
                src = sc_dir / f"{stem}_face.png"
                img = Image.open(src).convert("RGB") if src.exists() else _placeholder_image((cell_size, cell_size))
            else:
                step = sc_dir / f"{stem}.step"
                tmp_png = sc_dir / f"{stem}_step_render.png"
                if not tmp_png.exists():
                    render_step_to_png(step, tmp_png, size=cell_size)
                img = Image.open(tmp_png).convert("RGB") if tmp_png.exists() else _placeholder_image((cell_size, cell_size))
            img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            canvas.paste(img, (col_idx * cell_size, row_idx * cell_size))

    out_path.parent.mkdir(exist_ok=True, parents=True)
    canvas.save(out_path, format="JPEG", quality=85, optimize=True)
    print(f"  🖼  collage saved → {out_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}  (rerun with --seed {args.seed} to reproduce this exact sample set)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{job_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs → {output_dir}")

    # ------------------------------------------------------------------
    # 1. Load base AutoBrepModel + VAEs (inference_mode=False so we keep
    #    the encoder VAEs for encode_fsq_code on the real sample).
    # ------------------------------------------------------------------
    print("\nLoading base AutoBrepModel...")
    model = AutoBrepModel.load_from_checkpoint(
        f"{args.ckpt_dir}/ar.ckpt",
        inference_mode=False,
        surf_fsq_ckpt=f"{args.ckpt_dir}/surf-fsq.ckpt",
        edge_fsq_ckpt=f"{args.ckpt_dir}/edge-fsq.ckpt",
        strict=False,
        map_location=device,
    )
    model.to(device).eval()
    # bfloat16 cast for cad_gpt forward — matches train_lora_ar.train()
    model.to(dtype=torch.bfloat16)

    print("Loading decoder-only VAEs for reconstruction...")
    surface_fsq = (SurfaceFSQVAE.load_from_checkpoint(f"{args.ckpt_dir}/surf-fsq.ckpt")
                   .drop_encoder().to(device).eval())
    edge_fsq = (EdgeFSQVAE.load_from_checkpoint(f"{args.ckpt_dir}/edge-fsq.ckpt")
                .drop_encoder().to(device).eval())

    # ------------------------------------------------------------------
    # 3. Build a (no-op) ARDataModule purely for its filter/tokenize logic
    # ------------------------------------------------------------------
    data_module = ARDataModule(
        data_root="/tmp/_unused",  # we won't call setup()
        max_seq=model.hparams.max_seq,
        bit=model.hparams.bit,
        max_face=model.hparams.max_face,
        max_edge=1000,
        load_geom=False,
        load_meta=True,
        uv_invariant=True,
        batch_size=1,
        buffer_size=0,
        num_workers=0,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # 4-8. For each split, stream N samples and run the full per-sample
    #      pipeline (original STEP, real viz, tokenize, perplexity, completions).
    # ------------------------------------------------------------------
    all_metrics: list = []
    for split, n in [("train", args.num_train_samples),
                     ("validation", args.num_val_samples)]:
        if n <= 0:
            continue
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True, parents=True)
        for i, row in enumerate(iter_hf_samples(args.hf_repo, split, n,
                                                args.hf_sample_skip, data_module)):
            sample_dir = split_dir / f"sample_{i:03d}"
            try:
                m = process_one_sample(row, model, surface_fsq, edge_fsq,
                                       data_module, args, device, sample_dir)
                m["split"] = split
                m["index"] = i
                all_metrics.append(m)
            except Exception as e:
                print(f"  ❌ {split}/{i} failed: {e}")

        # Per-split collages: one from face debug PNGs, one rendered from STEPs.
        for mode in ("faces", "steps"):
            try:
                build_collage_for_split(
                    split_dir, args.num_completions, mode,
                    output_dir / f"{split}_collage_{mode}.jpg",
                    args.prefix_ratio,
                )
            except Exception as e:
                print(f"  ⚠️ {split} {mode} collage failed: {e}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    lines = [f"DONE - outputs at {output_dir}", ""]
    for m in all_metrics:
        lines.append(
            f"  [{m['split']:>10s} #{m['index']}] stem={m['stem']} "
            f"faces={m.get('num_faces_after_splitting')} "
            f"valid_seq_len={m['valid_seq_len']} "
            f"loss={m['loss']:.4f} ppl={m['perplexity']:.4f}"
        )
    summary = "\n" + "=" * 80 + "\n" + "\n".join(lines) + "\n" + "=" * 80 + "\n"
    print(summary)
    (output_dir / "summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
