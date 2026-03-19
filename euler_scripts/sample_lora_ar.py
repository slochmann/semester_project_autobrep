import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from peft import PeftModel
from einops import rearrange
import time
from datetime import datetime

# Import AutoBrep core (assuming run from core directory)
from autobrep.models.autoregressive import AutoBrepModel, AutoRegressiveSampler
from autobrep.models.vaes import EdgeFSQVAE, SurfaceFSQVAE
from autobrep.data.token_mapping import MMTokenIndex
from occwl.io import save_step as save_step_func
from autobrep.inference.brepgen_brep_builder import AutoBrepBuilder
from autobrep.inference.inference_common import reconstruct_compound

def parse_args():
    parser = argparse.ArgumentParser(description="Run AutoBrep LoRA Inference on Euler")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing base AutoBrep checkpoints")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated STEP files")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of CAD samples to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--threshold", type=float, default=0.95, help="Top-p sampling threshold")
    parser.add_argument("--complexity", type=int, default=17, help="Complexity token (14=easy, 15=medium, 16=hard, 17=random)")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of samples to generate concurrently (e.g. 10 for 24GB GPUs)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Load Base Model and VAEs
    print(f"Loading base AutoBrepModel from {args.ckpt_dir}...")
    model = AutoBrepModel.load_from_checkpoint(
        f"{args.ckpt_dir}/ar.ckpt",
        inference_mode=True,
        surf_fsq_ckpt=f"{args.ckpt_dir}/surf-fsq.ckpt",
        edge_fsq_ckpt=f"{args.ckpt_dir}/edge-fsq.ckpt",
        strict=False,
        map_location=device
    )
    model.to(device).eval()
    
    print("Loading VAEs for decoding...")
    surface_fsq = SurfaceFSQVAE.load_from_checkpoint(f"{args.ckpt_dir}/surf-fsq.ckpt").drop_encoder().to(device).eval()
    edge_fsq = EdgeFSQVAE.load_from_checkpoint(f"{args.ckpt_dir}/edge-fsq.ckpt").drop_encoder().to(device).eval()
    
    # 2. Attach LoRA Adapter
    print(f"Attaching LoRA adapter from {args.lora_adapter_path}...")
    model.cad_gpt = PeftModel.from_pretrained(model.cad_gpt, args.lora_adapter_path)
    model.cad_gpt.eval()
    
    # 3. Generate CAD Sequences
    generated_samples = []
    print(f"\n=== Generating {args.num_samples} samples (batch_size={args.batch_size}) ===")
    
    with torch.no_grad():
        num_batches = int(np.ceil(args.num_samples / args.batch_size))
        for batch_idx in range(num_batches):
            current_batch_size = min(args.batch_size, args.num_samples - batch_idx * args.batch_size)
            print(f"Generating batch {batch_idx+1}/{num_batches} ({current_batch_size} samples)...")
            
            prompt = (
                torch.LongTensor([
                    MMTokenIndex.BOS.value,
                    MMTokenIndex.BOM.value,
                    args.complexity,
                    MMTokenIndex.EOM.value,
                    MMTokenIndex.BOC.value,
                ] * current_batch_size)
                .reshape(current_batch_size, 5)
                .to(device)
            )
            
            start_time = time.time()
            try:
                samples = model.generate(prompt, args.temperature, args.threshold)
                full_seqs = torch.concat([prompt, samples], dim=-1)
                for seq in full_seqs:
                    generated_samples.append(seq.unsqueeze(0))
                print(f"  ✅ Batch complete in {time.time() - start_time:.2f}s")
            except Exception as e:
                print(f"  ❌ Error generating batch: {e}")
                
    # 4. Decode Tokens
    print(f"\n=== Decoding {len(generated_samples)} generated sequences ===")
    batch_decoded = []
    
    for idx, generated_seq in enumerate(generated_samples):
        sample = generated_seq[0].cpu().numpy()
        
        try:
            geom_tokens = None
            if MMTokenIndex.BOGEOM.value in sample:
                geom_s = np.where(sample==MMTokenIndex.BOGEOM.value)[0][0]
                geom_e = np.where(sample==MMTokenIndex.EOGEOM.value)[0][0]
                geom_tokens = sample[geom_s+1:geom_e]

            if MMTokenIndex.BOC.value in sample:
                cad_s = np.where(sample==MMTokenIndex.BOC.value)[0][0]
                cad_e = np.where(sample==MMTokenIndex.EOC.value)[0][0]
                cad_tokens = sample[cad_s+1:cad_e]

                if geom_tokens is not None:
                    cad_tokens = np.concatenate((geom_tokens, cad_tokens))

            pos_faces, code_faces, pos_edges, code_edges, face_edge_adj = model.decode(cad_tokens)
        except Exception as err:
            print(f"  ❌ Error extracting CAD tokens for sample {idx}: {err}")
            continue

        with torch.no_grad():
            geomZ_faces = surface_fsq.quantizer.indices_to_codes(
                torch.LongTensor(code_faces).to(device)
            ).permute(0, 2, 1)
            uv_ncs_faces = surface_fsq.decode(geomZ_faces.unflatten(-1, (2, 2))).sample
            uv_ncs_faces = rearrange(uv_ncs_faces, "b d ... -> b ... d").float().cpu().numpy()
            
            geomZ_edges = edge_fsq.quantizer.indices_to_codes(
                torch.LongTensor(code_edges).to(device)
            ).permute(0, 2, 1)
            uv_ncs_edges = edge_fsq.decode(geomZ_edges).sample
            uv_ncs_edges = rearrange(uv_ncs_edges, "b d ... -> b ... d").float().cpu().numpy()

        batch_decoded.append((pos_faces, pos_edges, uv_ncs_edges, uv_ncs_faces, face_edge_adj))

    print(f"  ✅ Successfully decoded {len(batch_decoded)} sequences.")
    
    # 5. Convert to Topologies and Export STEP
    print(f"\n=== Building Solid Topologies and Exporting to STEP ===")
    batch_cad_data = AutoRegressiveSampler.convert_to_cad_data(batch_decoded)
    
    builders = [AutoBrepBuilder(device=device, z_threshold=0.5, vertex_threshold=0.02, sewing_tolerance=1e-4)]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for sample_idx, cad_data in enumerate(batch_cad_data):
        sample_stem = f"lora_sample_{timestamp}_{str(sample_idx).zfill(3)}"
        try:
            result = reconstruct_compound(cad_data, builders)
            if result is not None:
                out_path = output_dir / f"{sample_stem}.step"
                save_step_func([result], out_path)
                print(f"  ✅ Saved: {out_path}")
            else:
                print(f"  ❌ Reconstruct returned None for {sample_stem}")
        except Exception as e:
            print(f"  ❌ Error reconstructing {sample_stem}: {e}")
            
    print(f"\n🎉 Inference and Export Complete. Files saved to {output_dir}")

if __name__ == "__main__":
    main()
