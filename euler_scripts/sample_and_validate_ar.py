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
    p.add_argument("--hf_split", type=str, default="validation")
    p.add_argument("--hf_sample_skip", type=int, default=0,
                   help="Skip N samples in the stream before taking one")

    # Unconditional sampling
    p.add_argument("--sample_batch_size", type=int, default=4)
    p.add_argument("--sample_temperature", type=float, default=0.7)
    p.add_argument("--sample_threshold", type=float, default=0.9)
    p.add_argument("--sample_complexity", type=int, default=17,
                   help="14=easy, 15=medium, 16=hard, 17=random")

    # Autocomplete
    p.add_argument("--prefix_ratio", type=float, default=0.5,
                   help="Fraction of the real sequence to feed as prompt")

    # Reconstruction tolerances
    p.add_argument("--z_threshold", type=float, default=0.002)
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
# Step 2: unconditional sampling
# ---------------------------------------------------------------------------

def run_unconditional_batch(model, surface_fsq, edge_fsq, args, device, output_dir: Path):
    print("\n" + "=" * 80)
    print(f"🎨 UNCONDITIONAL SAMPLING — batch_size={args.sample_batch_size}")
    print("=" * 80)
    sub_dir = output_dir / "uncond"
    sub_dir.mkdir(exist_ok=True, parents=True)

    prompt = (
        torch.LongTensor([
            MMTokenIndex.BOS.value,
            MMTokenIndex.BOM.value,
            args.sample_complexity,
            MMTokenIndex.EOM.value,
            MMTokenIndex.BOC.value,
        ] * args.sample_batch_size)
        .reshape(args.sample_batch_size, 5)
        .to(device)
    )

    with torch.no_grad():
        samples = model.generate(prompt, args.sample_temperature, args.sample_threshold)
    full_seqs = torch.concat([prompt, samples], dim=-1)

    success = 0
    for i, seq in enumerate(full_seqs):
        seq_np = seq.cpu().numpy()
        if decode_and_reconstruct(seq_np, model, surface_fsq, edge_fsq, device,
                                  sub_dir, f"uncond_{i:03d}", args):
            success += 1
    print(f"  📊 Unconditional success: {success}/{args.sample_batch_size}")


# ---------------------------------------------------------------------------
# Step 3 + 5: Download one HF sample, tokenize via ARDataModule
# ---------------------------------------------------------------------------

def fetch_hf_sample(args, data_module: ARDataModule):
    """
    Stream from HF until one sample passes ARDataModule.pre_filter, return the
    raw row dict (with bytes fields ready for unpickle()).
    """
    print(f"\n📥 Streaming {args.hf_repo} split={args.hf_split} ...")
    ds = load_dataset(args.hf_repo, split=args.hf_split, streaming=True)
    it = iter(ds)
    # Skip if requested
    for _ in range(args.hf_sample_skip):
        next(it)
    seen = 0
    for row in it:
        seen += 1
        # Match pre_filter expectations: scaled_unique + num_faces range.
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
        print(f"  ✅ Found sample after streaming {seen} rows: "
              f"stem={row.get('stem')!r}, num_faces={nf}")
        return row
    raise RuntimeError("No HF sample passed pre_filter")


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
# Step 7: autocomplete from first 50% of the real sequence
# ---------------------------------------------------------------------------

def autocomplete_from_prefix(model, surface_fsq, edge_fsq, updated_tokens,
                             args, device, output_dir: Path):
    print("\n" + "=" * 80)
    print(f"🪡 AUTOCOMPLETE — prefix_ratio={args.prefix_ratio}")
    print("=" * 80)
    sub_dir = output_dir / "autocomplete"
    sub_dir.mkdir(exist_ok=True, parents=True)

    # Trim padding (-1) — keep only the valid tokens of the real sequence
    seq = updated_tokens[0]
    valid_len = int((seq >= 0).sum().item())
    valid_seq = seq[:valid_len]

    # Save the full ground-truth reconstruction for reference
    gt_np = valid_seq.cpu().numpy()
    decode_and_reconstruct(gt_np, model, surface_fsq, edge_fsq, device,
                           sub_dir, "ground_truth", args)

    # Take the first prefix_ratio of valid tokens as the prompt
    prefix_len = max(5, int(valid_len * args.prefix_ratio))
    prompt = valid_seq[:prefix_len].unsqueeze(0).to(device)
    print(f"  Valid sequence length: {valid_len}, prefix: {prefix_len}")

    with torch.no_grad():
        generated = model.generate(prompt, args.sample_temperature, args.sample_threshold)
    full_seq = torch.concat([prompt, generated], dim=-1)[0].cpu().numpy()

    decode_and_reconstruct(full_seq, model, surface_fsq, edge_fsq, device,
                           sub_dir, f"autocomplete_p{int(args.prefix_ratio*100)}", args)


# ---------------------------------------------------------------------------
# Visualize the original (ground-truth) HF sample using its decoded geometry
# ---------------------------------------------------------------------------

def visualize_real_sample(tokenized_with_real_features, model, surface_fsq, edge_fsq,
                          device, output_dir: Path, args, real_row_unpickled):
    """
    Build CAD data directly from the real face_ncs / edge_ncs / bbox / adjacency
    (i.e. ground-truth geometry, no model involvement) and save STEP + PNGs.
    """
    print("\n📷 Visualizing real HF sample (ground-truth geometry, no model)")
    sub_dir = output_dir / "real_sample"
    sub_dir.mkdir(exist_ok=True, parents=True)

    face_ncs = real_row_unpickled["face_points_normalized"]
    edge_ncs = real_row_unpickled["edge_points_normalized"]
    face_pos = real_row_unpickled["face_bbox_world"]
    edge_pos = real_row_unpickled["edge_bbox_world"]
    face_edge_adj = real_row_unpickled["face_edge_incidence"].astype(bool)

    batch_decoded = [(face_pos, edge_pos, edge_ncs, face_ncs, face_edge_adj)]
    batch_cad_data = AutoRegressiveSampler.convert_to_cad_data(batch_decoded)
    cad_data = batch_cad_data[0]

    face_img = sub_dir / "real_face.png"
    edge_img = sub_dir / "real_edge.png"
    try:
        save_debug_images(cad_data, face_img, edge_img)
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
            save_step_func([result], sub_dir / "real.step")
            print("  ✅ Real sample reconstructed and saved")
    except Exception as e:
        print(f"  ⚠️ reconstruct failed: {e}")


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
    # 2. Unconditional sampling
    # ------------------------------------------------------------------
    run_unconditional_batch(model, surface_fsq, edge_fsq, args, device, output_dir)

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
    # 4. Download one HF sample
    # ------------------------------------------------------------------
    row = fetch_hf_sample(args, data_module)
    row_unpickled = data_module.unpickle(row)

    # ------------------------------------------------------------------
    # 5. Visualize the real sample (no model involvement)
    # ------------------------------------------------------------------
    visualize_real_sample(None, model, surface_fsq, edge_fsq, device,
                          output_dir, args, row_unpickled)

    # ------------------------------------------------------------------
    # 6. Tokenize and compute teacher-forced perplexity
    # ------------------------------------------------------------------
    print("\n🔢 Tokenizing sample via ARDataModule.map_func ...")
    tokenized = data_module.map_func(row_unpickled, aug=False)
    print(f"  seq length: {len(tokenized['seq'])} (incl. padding)")
    print(f"  face_ncs shape: {np.asarray(tokenized['face_ncs']).shape}")
    print(f"  edge_ncs shape: {np.asarray(tokenized['edge_ncs']).shape}")

    print("\n📊 Computing teacher-forced loss / perplexity ...")
    loss, ppl, updated_tokens = compute_perplexity(model, tokenized, device)
    print(f"  loss = {loss:.4f}")
    print(f"  perplexity = {ppl:.4f}")

    # ------------------------------------------------------------------
    # 7 + 8. Autocomplete from prefix and visualize
    # ------------------------------------------------------------------
    autocomplete_from_prefix(model, surface_fsq, edge_fsq, updated_tokens,
                             args, device, output_dir)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    summary = (
        f"\n{'=' * 80}\n"
        f"DONE — outputs at {output_dir}\n"
        f"  loss = {loss:.4f}\n"
        f"  perplexity = {ppl:.4f}\n"
        f"{'=' * 80}\n"
    )
    print(summary)
    (output_dir / "summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
