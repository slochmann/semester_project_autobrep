import argparse
import os
from pathlib import Path

import torch
import torch.optim as optim
import wandb
from autobrep.data.abc_data import ARDataModule
from autobrep.models.autoregressive import AutoBrepModel
from peft import LoraConfig, get_peft_model

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

    return parser.parse_args()


def train_lora(
    model,
    model_lora,
    dataloader,
    optimizer,
    scheduler,
    device,
    num_epochs=5,
    wandb_run=None,
):
    """
    Fine-tune LoRA adapters on BREP generation task.
    """
    # Cast to bfloat16 for training
    model.to(device).to(dtype=torch.bfloat16)
    model_lora.train()
    model.cad_gpt.train()

    loss_history = []

    print("\n" + "=" * 80)
    print("STARTING LORA FINE-TUNING")
    print("=" * 80)

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

            # Forward pass - model computes loss internally using autoregressive wrapper
            # Pass full token sequence; ar_decoder handles shift internally
            loss = model_lora(updated_tokens)

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

            # Clean up GPU cache to avoid memory fragmentation
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            if (batch_idx + 1) % 5 == 0:
                avg_loss = epoch_loss / num_batches
                lr = optimizer.param_groups[0]["lr"]

                print(
                    f"Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx + 1} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )

                # Log to wandb
                if wandb_run is not None:
                    wandb.log(
                        {
                            "loss": avg_loss,
                            "learning_rate": lr,
                        }
                    )

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(
            f"\n✅ Epoch {epoch + 1}/{num_epochs} completed | Avg Loss: {avg_epoch_loss:.4f}\n"
        )

        # Log epoch summary to wandb
        if wandb_run is not None:
            wandb.log(
                {
                    "epoch_loss": avg_epoch_loss,
                }
            )

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
    run_name = f"{job_name}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name,
        )
    else:
        wandb_run = None

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

    # 2. Configure LoRA
    print(f"Configuring LoRA... (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_v"],
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
        num_workers=args.num_workers,  # 0 for single-process loading (memory safe)
        drop_last=True,
        pin_memory=(args.num_workers > 0),  # Only pin if using workers
        persistent_workers=(args.num_workers > 0),  # Only persist if using workers
    )
    data_module.setup(stage="fit")
    train_dataloader = data_module.train_dataloader()

    # 4. Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model_lora.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.05,
        eps=1e-5,
    )

    # Calculate steps per epoch
    try:
        steps_per_epoch = max(1, len(train_dataloader))
    except TypeError:
        # Fallback if ARDataModule uses IterableDataset with no len()
        steps_per_epoch = 25

    total_steps = args.num_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # 5. Train and Save
    loss_history = train_lora(
        model=model,
        model_lora=model_lora,
        dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        wandb_run=wandb.run if args.track else None,
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
