import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import wandb
from autobrep.data.abc_data import ARDataModule
from autobrep.data.token_mapping import MMTokenIndex
from autobrep.inference.brepgen_brep_builder import AutoBrepBuilder
from autobrep.inference.inference_common import (
    reconstruct_compound,
    save_debug_images,
)
from autobrep.models.autoregressive import AutoBrepModel
from autobrep.models.vaes import EdgeFSQVAE, SurfaceFSQVAE
from einops import rearrange
from occwl.io import save_step as save_step_func
from peft import LoraConfig, get_peft_model
from PIL import Image

# Get SLURM job name (or None if not running under SLURM)
job_name = os.environ.get("SLURM_JOB_NAME", "local_run")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA on AutoBrep CAD_GPT")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory containing base checkpoints",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to parquet dataset (containing train/ and val/)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to save the LoRA adapter"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for AdamW"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers (0=disabled for memory efficiency)",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=1,
        help="Prefetch factor for dataloader (reduce if OOM)",
    )

    # LoRA configuration
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank (r parameter)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (scaling factor)",
    )

    # Weights & Biases tracking
    parser.add_argument(
        "--track", type=bool, default=True, help="Track the experiment with wandb"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="lora-autobrep", help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Wandb entity name"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    # Inference sampling during training
    parser.add_argument(
        "--sample_interval_epochs",
        type=int,
        default=1,
        help="Interval (in epochs) to run inference sampling",
    )
    parser.add_argument(
        "--sample_interval_batches",
        type=int,
        default=0,
        help="Interval (in batches) to run inference sampling within epochs (0=disabled)",
    )
    parser.add_argument(
        "--num_samples_to_generate",
        type=int,
        default=24,
        help="Number of samples to generate during inference sampling",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=5,
        help="Batch size for inference sampling (generate in smaller batches to manage memory)",
    )
    parser.add_argument(
        "--sample_temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--sample_threshold",
        type=float,
        default=0.95,
        help="Top-p sampling threshold for generation",
    )
    parser.add_argument(
        "--sample_complexity",
        type=int,
        default=14,
        help="Complexity token (14=easy, 15=medium, 16=hard, 17=random)",
    )

    # Validation frequency
    parser.add_argument(
        "--val_interval_epochs",
        type=int,
        default=0,
        help="Interval (in epochs) to run validation",
    )
    parser.add_argument(
        "--val_interval_batches",
        type=int,
        default=30,
        help="Interval (in batches) to run validation within epochs (0=disabled)",
    )

    return parser.parse_args()


def sample_during_training(
    model,
    device,
    surface_fsq,
    edge_fsq,
    args,
    step_num,
    epoch,
    output_dir,
    wandb_run=None,
):
    """
    Run inference sampling during training for visualization.

    Args:
        model: AutoBrepModel with LoRA adapters attached
        device: torch device
        surface_fsq: Surface FSQ VAE for decoding
        edge_fsq: Edge FSQ VAE for decoding
        args: parsed arguments
        step_num: current training step
        epoch: current epoch
        output_dir: directory to save samples
        wandb_run: wandb run object (optional)
    """
    print(f"\n{'=' * 80}")
    print(f"🎨 INFERENCE SAMPLING at Step {step_num}, Epoch {epoch}")
    print(f"{'=' * 80}")

    sample_dir = Path(output_dir) / f"samples_step_{step_num:06d}"
    sample_dir.mkdir(exist_ok=True, parents=True)

    model.eval()
    generated_samples = []

    # Generate samples in batches
    num_batches = int(np.ceil(args.num_samples_to_generate / args.sample_batch_size))
    with torch.no_grad():
        for batch_idx in range(num_batches):
            current_batch_size = min(
                args.sample_batch_size,
                args.num_samples_to_generate - batch_idx * args.sample_batch_size,
            )

            # Create prompt for sampling
            prompt = (
                torch.LongTensor(
                    [
                        MMTokenIndex.BOS.value,
                        MMTokenIndex.BOM.value,
                        args.sample_complexity,
                        MMTokenIndex.EOM.value,
                        MMTokenIndex.BOC.value,
                    ]
                    * current_batch_size
                )
                .reshape(current_batch_size, 5)
                .to(device)
            )

            try:
                # Generate tokens
                samples = model.generate(
                    prompt, args.sample_temperature, args.sample_threshold
                )
                full_seqs = torch.concat([prompt, samples], dim=-1)
                for seq in full_seqs:
                    generated_samples.append(seq.unsqueeze(0).detach())
                print(f"  ✅ Batch {batch_idx + 1}/{num_batches} generated")
            except Exception as e:
                print(f"  ❌ Error generating batch {batch_idx + 1}: {e}")

    # Decode tokens and convert to CAD data
    print(f"\nDecoding {len(generated_samples)} sequences...")
    batch_decoded = []

    for idx, generated_seq in enumerate(generated_samples):
        sample = generated_seq[0].cpu().detach().numpy()

        try:
            cad_tokens = None
            geom_tokens = None
            if MMTokenIndex.BOGEOM.value in sample:
                geom_s = np.where(sample == MMTokenIndex.BOGEOM.value)[0][0]
                geom_e = np.where(sample == MMTokenIndex.EOGEOM.value)[0][0]
                geom_tokens = sample[geom_s + 1 : geom_e]

            if MMTokenIndex.BOC.value in sample:
                cad_s = np.where(sample == MMTokenIndex.BOC.value)[0][0]
                cad_e = np.where(sample == MMTokenIndex.EOC.value)[0][0]
                cad_tokens = sample[cad_s + 1 : cad_e]

                if geom_tokens is not None:
                    cad_tokens = np.concatenate((geom_tokens, cad_tokens))

            if cad_tokens is None:
                raise ValueError("No CAD tokens found in sample")

            pos_faces, code_faces, pos_edges, code_edges, face_edge_adj = model.decode(
                cad_tokens
            )

            # Decode surfaces and edges
            geomZ_faces = surface_fsq.quantizer.indices_to_codes(
                torch.LongTensor(code_faces).to(device)
            ).permute(0, 2, 1)
            uv_ncs_faces = surface_fsq.decode(geomZ_faces.unflatten(-1, (2, 2))).sample
            uv_ncs_faces = (
                rearrange(uv_ncs_faces, "b d ... -> b ... d")
                .float()
                .cpu()
                .detach()
                .numpy()
            )

            geomZ_edges = edge_fsq.quantizer.indices_to_codes(
                torch.LongTensor(code_edges).to(device)
            ).permute(0, 2, 1)
            uv_ncs_edges = edge_fsq.decode(geomZ_edges).sample
            uv_ncs_edges = (
                rearrange(uv_ncs_edges, "b d ... -> b ... d")
                .float()
                .cpu()
                .detach()
                .numpy()
            )

            batch_decoded.append(
                (pos_faces, pos_edges, uv_ncs_edges, uv_ncs_faces, face_edge_adj)
            )
        except Exception as e:
            print(f"  ⚠️ Error decoding sample {idx}: {e}")
            continue

    print(f"  ✅ Successfully decoded {len(batch_decoded)} sequences")

    # Convert to CAD data and reconstruct
    print("\nReconstructing geometries...")
    from autobrep.models.autoregressive import AutoRegressiveSampler

    batch_cad_data = AutoRegressiveSampler.convert_to_cad_data(batch_decoded)
    builders = [
        AutoBrepBuilder(
            device=device, z_threshold=0.5, vertex_threshold=0.02, sewing_tolerance=1e-4
        )
    ]

    success_count = 0
    sample_images_list = []  # List to store concatenated face+edge images
    for sample_idx, cad_data in enumerate(batch_cad_data):
        sample_stem = f"step_{step_num:06d}_sample_{str(sample_idx).zfill(3)}"
        try:
            # Save debug images
            face_img_path = sample_dir / f"{sample_stem}_face.png"
            edge_img_path = sample_dir / f"{sample_stem}_edge.png"

            save_debug_images(
                cad_data,
                face_img_path,
                edge_img_path,
            )

            # Reconstruct and save STEP
            result = reconstruct_compound(cad_data, builders)
            if result is not None:
                step_path = sample_dir / f"{sample_stem}.step"
                save_step_func([result], step_path)
                success_count += 1

                # Load and concatenate face + edge images
                try:
                    face_img = Image.open(face_img_path)
                    edge_img = Image.open(edge_img_path)

                    # Downscale images to reduce memory/file size (60% of original)
                    scale_factor = 0.6
                    face_img = face_img.resize(
                        (
                            int(face_img.width * scale_factor),
                            int(face_img.height * scale_factor),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                    edge_img = edge_img.resize(
                        (
                            int(edge_img.width * scale_factor),
                            int(edge_img.height * scale_factor),
                        ),
                        Image.Resampling.LANCZOS,
                    )

                    # Concatenate horizontally
                    concat_img = Image.new(
                        "RGB",
                        (face_img.width + edge_img.width, face_img.height),
                    )
                    concat_img.paste(face_img, (0, 0))
                    concat_img.paste(edge_img, (face_img.width, 0))
                    sample_images_list.append(concat_img)
                except Exception as e:
                    print(f"  ⚠️ Sample {sample_idx}: error concatenating images: {e}")

                print(f"  ✅ Sample {sample_idx}: reconstructed and saved")
            else:
                print(f"  ⚠️ Sample {sample_idx}: reconstruction returned None")
        except Exception as e:
            print(f"  ⚠️ Sample {sample_idx}: error during reconstruction: {e}")

    # Create a 6x4 collage from the 24 samples
    collage = None
    if len(sample_images_list) >= 1:
        # Use only the first 24 samples (or fewer if less generated)
        sample_images_list = sample_images_list[:24]

        # Determine collage grid size based on successful samples
        num_samples = len(sample_images_list)
        grid_cols = 4  # 4 columns
        grid_rows = 6  # 6 rows for 24 samples

        if num_samples > 0:
            # Resize images to 50% of original size
            resize_scale = 0.25
            resized_images = []
            for img in sample_images_list:
                new_width = int(img.width * resize_scale)
                new_height = int(img.height * resize_scale)
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized_images.append(resized_img)

            # Get image dimensions from first resized sample
            img_width = resized_images[0].width
            img_height = resized_images[0].height

            # Create collage canvas
            collage_width = img_width * grid_cols
            collage_height = img_height * grid_rows
            collage = Image.new("RGB", (collage_width, collage_height))

            # Paste resized images into grid
            for idx, img in enumerate(resized_images):
                row = idx // grid_cols
                col = idx % grid_cols
                x = col * img_width
                y = row * img_height
                collage.paste(img, (x, y))

    # Log to wandb
    if wandb_run is not None:
        log_dict = {
            "sampling/success_rate": success_count / max(len(batch_cad_data), 1),
            "sampling/num_valid": success_count,
        }

        # Log collage if available
        if collage is not None:
            collage_path = sample_dir / "collage.jpg"
            collage.save(collage_path, format="JPEG", quality=70, optimize=True)
            log_dict["sampling/collage"] = wandb.Image(str(collage_path))

        wandb.log(log_dict, step=step_num)
        print("  📊 Logged sampling metrics to wandb")

    print(f"  📁 Samples saved to {sample_dir}")
    print(f"{'=' * 80}\n")

    model.train()  # Resume training mode


def validate(model, model_lora, val_dataloader, device):
    """
    Run validation on the full validation set.
    Returns average validation loss.
    """
    model_lora.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            token = batch["seq"].to(device)
            face_ncs = batch["face_ncs"].to(device).to(dtype=torch.bfloat16)
            edge_ncs = batch["edge_ncs"].to(device).to(dtype=torch.bfloat16)

            # Encode FSQ codes (same as training) - use base model
            surf_id, edge_id = model.encode_fsq_code(face_ncs, edge_ncs)

            updated_tokens = []
            for _token, _surf_id, _edge_id in zip(token, surf_id, edge_id):
                batch_data = model.copy_fsq_code(_token, _surf_id, _edge_id)
                batch_data = torch.nn.functional.pad(
                    batch_data,
                    (0, model.pad_len - len(batch_data)),
                    value=-1,
                )
                updated_tokens.append(batch_data)

            updated_tokens = torch.stack(updated_tokens).detach()

            # Build cond_mask: ignore loss on first 4 tokens (BOS, BOM, complexity, EOM)
            cond_mask = torch.zeros(
                updated_tokens.shape[0], model.pad_len, dtype=torch.bool, device=device
            )
            cond_mask[:, :4] = True

            # Forward pass using model_lora
            loss = model_lora(updated_tokens, cond_mask=cond_mask)

            val_loss += loss.item()
            num_batches += 1

    return val_loss / max(num_batches, 1)


def train_lora(
    model,
    model_lora,
    dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    device,
    num_epochs=5,
    wandb_run=None,
    surface_fsq=None,
    edge_fsq=None,
    args=None,
    output_dir=None,
):
    """
    Fine-tune LoRA adapters on BREP generation task.
    Scheduler runs continuously across epochs - no recreation mid-training.
    """
    # Cast to bfloat16 for training
    model.to(device).to(dtype=torch.bfloat16)
    model_lora.train()
    model.cad_gpt.train()

    loss_history = []
    global_step = 0

    print("\n" + "=" * 80)
    print("STARTING LORA FINE-TUNING")
    print("=" * 80)

    # Run initial validation before training
    print("\n📊 Running initial validation...")
    initial_val_loss = validate(model, model_lora, val_dataloader, device)
    print(f"   Initial Validation Loss: {initial_val_loss:.4f}\n")
    if wandb_run is not None:
        wandb.log({"val_loss": initial_val_loss}, step=0)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            token = batch["seq"].to(device)
            face_ncs = batch["face_ncs"].to(device).to(dtype=torch.bfloat16)
            edge_ncs = batch["edge_ncs"].to(device).to(dtype=torch.bfloat16)

            with torch.no_grad():
                surf_id, edge_id = model.encode_fsq_code(face_ncs, edge_ncs)

            updated_tokens = []
            for _token, _surf_id, _edge_id in zip(token, surf_id, edge_id):
                batch_data = model.copy_fsq_code(_token, _surf_id, _edge_id)
                batch_data = torch.nn.functional.pad(
                    batch_data,
                    (0, model.pad_len - len(batch_data)),
                    value=-1,
                )
                updated_tokens.append(batch_data)

            updated_tokens = torch.stack(updated_tokens).detach()

            # Build cond_mask: ignore loss on first 4 tokens (BOS, BOM, complexity, EOM)
            cond_mask = torch.zeros(
                updated_tokens.shape[0], model.pad_len, dtype=torch.bool, device=device
            )
            cond_mask[:, :4] = True

            # Forward pass - model computes loss internally using autoregressive wrapper
            # Pass full token sequence; ar_decoder handles shift internally
            loss = model_lora(updated_tokens, cond_mask=cond_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
            )  # Note: this should clip model_lora or model, depending on where grad requires are. model is fine since only LoRA requires grad.
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            loss_history.append(loss.item())

            # Capture step before incrementing for accurate logging
            current_step = global_step
            global_step += 1

            # Clean up GPU cache to avoid memory fragmentation
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # Periodic batch-level validation during epoch
            if (
                args is not None
                and args.val_interval_batches > 0
                and (batch_idx + 1) % args.val_interval_batches == 0
            ):
                print(f"\n📊 Running validation at step {global_step}...")
                val_loss = validate(model, model_lora, val_dataloader, device)
                print(f"   Validation Loss: {val_loss:.4f}\n")
                if wandb_run is not None:
                    wandb.log({"val_loss": val_loss}, step=global_step)
                # Resume training mode after validation
                model_lora.train()
                model.cad_gpt.train()

            # Periodic batch-level sampling during epoch
            if (
                args is not None
                and args.sample_interval_batches > 0
                and (batch_idx + 1) % args.sample_interval_batches == 0
            ):
                sample_during_training(
                    model=model,
                    device=device,
                    surface_fsq=surface_fsq,
                    edge_fsq=edge_fsq,
                    args=args,
                    step_num=current_step,
                    epoch=epoch + 1,
                    output_dir=output_dir,
                    wandb_run=wandb_run,
                )
                # Resume training mode after sampling
                model_lora.train()
                model.cad_gpt.train()

            if (batch_idx + 1) % 5 == 0:
                avg_loss = epoch_loss / num_batches
                lr = optimizer.param_groups[0]["lr"]

                print(
                    f"Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx + 1} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )

                # Log to wandb using captured step to avoid conflicts with async rendering
                if wandb_run is not None:
                    wandb.log(
                        {
                            "loss": avg_loss,
                            "learning_rate": lr,
                        },
                        step=current_step,
                    )

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(
            f"\n✅ Epoch {epoch + 1}/{num_epochs} completed | Avg Loss: {avg_epoch_loss:.4f} | Batches: {num_batches}\n"
        )

        # Run validation at epoch intervals
        if args.val_interval_epochs > 0 and (epoch + 1) % args.val_interval_epochs == 0:
            print("📊 Running validation...")
            val_loss_avg = validate(model, model_lora, val_dataloader, device)
            print(f"   Validation Loss: {val_loss_avg:.4f}\n")

            # Log to wandb
            if wandb_run is not None:
                wandb.log(
                    {
                        "epoch_loss": avg_epoch_loss,
                        "val_loss": val_loss_avg,
                    },
                    step=global_step,
                )
        else:
            # Still log epoch loss if not using wandb
            if wandb_run is not None:
                wandb.log(
                    {
                        "epoch_loss": avg_epoch_loss,
                    },
                    step=global_step,
                )

        # Resume training mode for next epoch
        model_lora.train()
        model.cad_gpt.train()

        # Periodic inference sampling at epoch intervals
        if args is not None and (epoch + 1) % args.sample_interval_epochs == 0:
            sample_during_training(
                model=model,
                device=device,
                surface_fsq=surface_fsq,
                edge_fsq=edge_fsq,
                args=args,
                step_num=global_step,
                epoch=epoch + 1,
                output_dir=output_dir,
                wandb_run=wandb_run,
            )

            # Save LoRA checkpoint at sampling milestone
            if output_dir is not None:
                ckpt_dir = Path(output_dir) / f"checkpoint_epoch_{epoch + 1:03d}"
                ckpt_dir.mkdir(exist_ok=True, parents=True)
                model_lora.save_pretrained(ckpt_dir)
                print(f"💾 LoRA checkpoint saved to {ckpt_dir}")

            # Resume training mode
            model_lora.train()
            model.cad_gpt.train()

    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

    return loss_history


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(args.seed)

    # Initialize wandb
    timestamp = datetime.now().strftime("%d.%m.%H:%M:%S")
    # Create a descriptive run name using hyperparameters
    run_name = f"{job_name}_T{args.sample_temperature}_comp{args.sample_complexity}_lr{args.learning_rate}_e{args.num_epochs}_r{args.lora_r}_a{args.lora_alpha}_{timestamp}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name,
        )

    # 1. Load Model
    print("Loading base AutoBrepModel...")
    model = AutoBrepModel.load_from_checkpoint(
        f"{args.ckpt_dir}/ar.ckpt",
        inference_mode=False,
        surf_fsq_ckpt=f"{args.ckpt_dir}/surf-fsq.ckpt",
        edge_fsq_ckpt=f"{args.ckpt_dir}/edge-fsq.ckpt",
        strict=False,
        map_location=device,
    )
    model.to(device).eval()

    # 1b. Load VAEs for inference sampling
    print("Loading VAEs for inference sampling...")
    surface_fsq = (
        SurfaceFSQVAE.load_from_checkpoint(f"{args.ckpt_dir}/surf-fsq.ckpt")
        .drop_encoder()
        .to(device)
        .eval()
    )
    edge_fsq = (
        EdgeFSQVAE.load_from_checkpoint(f"{args.ckpt_dir}/edge-fsq.ckpt")
        .drop_encoder()
        .to(device)
        .eval()
    )

    # 2. Configure LoRA
    print(f"Configuring LoRA... (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_v", "proj"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=[],
    )
    # Apply to cad_gpt only
    model_lora = get_peft_model(model.cad_gpt, lora_config)

    # 3. Setup ARDataModule
    print("Setting up ARDataModule...")
    data_module = ARDataModule(
        data_root=args.dataset_path,
        max_seq=model.hparams.max_seq,
        bit=model.hparams.bit,
        max_face=model.hparams.max_face,
        max_edge=1000,
        load_geom=False,
        load_meta=True,
        uv_invariant=True,
        batch_size=args.batch_size,
        buffer_size=1000,
        num_workers=args.num_workers,  # 0 for single-process loading (memory safe)
        drop_last=True,
        pin_memory=(args.num_workers > 0),  # Only pin if using workers
        persistent_workers=(args.num_workers > 0),  # Only persist if using workers
    )
    data_module.setup(stage="fit")
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # 4. Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model_lora.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.05,
        eps=1e-5,
    )

    # Count actual batches in dataset for accurate scheduler initialization
    print("Counting batches in training dataset...")
    num_batches_counted = 0
    for _ in train_dataloader:
        num_batches_counted += 1

    total_steps = args.num_epochs * num_batches_counted
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    print(f"✅ Dataset has {num_batches_counted} batches per epoch")
    print(
        f"Scheduler initialized with T_max={total_steps} ({num_batches_counted} steps/epoch × {args.num_epochs} epochs)"
    )

    # 5. Train and Save
    loss_history = train_lora(
        model=model,
        model_lora=model_lora,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        wandb_run=wandb.run if args.track else None,
        surface_fsq=surface_fsq,
        edge_fsq=edge_fsq,
        args=args,
        output_dir=args.output_dir,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_lora.save_pretrained(args.output_dir)
    print(f"\n💾 LoRA adapter saved to {args.output_dir}")

    # Log final artifacts to wandb
    if args.track:
        artifact = wandb.Artifact("lora-adapter", type="model")
        artifact.add_dir(args.output_dir)
        wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    # Print LoRA adapter specs
    print("\n" + "=" * 80)
    print("LORA ADAPTER SPECIFICATIONS")
    print("=" * 80)
    print(f"Config: {lora_config}")
    print(f"\nAdapter Output Directory: {args.output_dir}")
    print("\nTrainable Parameters:")
    trainable_params = sum(
        p.numel() for p in model_lora.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model_lora.parameters())
    print(f"  - Trainable: {trainable_params:,}")
    print(f"  - Total: {total_params:,}")
    print(f"  - Percentage: {100 * trainable_params / total_params:.2f}%")
    print("\nBase Model Parameters:")
    base_trainable = sum(
        p.numel() for p in model.cad_gpt.parameters() if p.requires_grad
    )
    base_total = sum(p.numel() for p in model.cad_gpt.parameters())
    print(f"  - Trainable: {base_trainable:,}")
    print(f"  - Total: {base_total:,}")
    print("=" * 80)

    # Finish wandb run
    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
