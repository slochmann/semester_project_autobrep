import argparse
import sys
import torch
import torch.optim as optim
from pathlib import Path

from peft import LoraConfig, get_peft_model
from autobrep.models.autoregressive import AutoBrepModel
from autobrep.data.abc_data import ARDataModule

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA on AutoBrep CAD_GPT")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing base checkpoints")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to parquet dataset (containing train/ and val/)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the LoRA adapter")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for AdamW")
    return parser.parse_args()

def train_lora(model, model_lora, dataloader, optimizer, scheduler, device, num_epochs=5):
    """
    Fine-tune LoRA adapters on BREP generation task.
    """
    model.to(device)
    model_lora.train()
    model.cad_gpt.train()
    
    loss_history = []
    
    print("\n" + "="*80)
    print("STARTING LORA FINE-TUNING")
    print("="*80)
    
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
            
            input_ids = updated_tokens[:, :-1]
            target_ids = updated_tokens[:, 1:]
            
            logits = model_lora(input_ids)
            
            batch_size, seq_len, vocab_size = logits.shape
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                target_ids.view(-1),
                reduction='mean',
                ignore_index=-1
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0
            ) # Note: this should clip model_lora or model, depending on where grad requires are. model is fine since only LoRA requires grad.
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            loss_history.append(loss.item())
            
            if (batch_idx + 1) % 5 == 0:
                avg_loss = epoch_loss / num_batches
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\n✅ Epoch {epoch+1}/{num_epochs} completed | Avg Loss: {avg_epoch_loss:.4f}\n")
        
    print("="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    return loss_history

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    print("Loading base AutoBrepModel...")
    model = AutoBrepModel.load_from_checkpoint(
        f"{args.ckpt_dir}/ar.ckpt",
        inference_mode=False,
        surf_fsq_ckpt=f"{args.ckpt_dir}/surf-fsq.ckpt",
        edge_fsq_ckpt=f"{args.ckpt_dir}/edge-fsq.ckpt",
        strict=False,
        map_location=device
    )
    model.to(device).eval()
    
    # 2. Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["to_q", "to_v"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=[]
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
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )
    data_module.setup(stage="fit")
    train_dataloader = data_module.train_dataloader()
    
    # 4. Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model_lora.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.05,
        eps=1e-5
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
        num_epochs=args.num_epochs
    )
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_lora.save_pretrained(args.output_dir)
    print(f"\n💾 LoRA adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()
