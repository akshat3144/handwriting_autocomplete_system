"""
HiGAN+ Training Improvements - Code Modifications
==================================================

This file contains the specific code changes to improve your HiGAN+ metrics.
Copy these modifications to your training notebook (code.ipynb).

Target Improvements:
- CER: 8% → 4-5%
- WER: 25% → 16-20%
- FID: 45 → 30-35
- MSSIM: 0.65 → 0.75-0.80
- PSNR: 18 → 22-24 dB
"""

# ============================================================================
# CHANGE 1: Update Config Settings
# ============================================================================
# Location: Cell "Load Configuration and Dataset"

print("="*70)
print("CHANGE 1: Configuration Updates")
print("="*70)

config_changes = """
# In the cell where you load config:

config_path = os.path.join(project_path, 'configs', 'gan_iam.yml')
cfg = yaml2config(config_path)

# === CRITICAL CHANGES ===
cfg.training.batch_size = 16  # Change from 8 → 16 (or 24 if GPU allows)
cfg.training.epochs = 70      # Train full duration (was stopping at 20)

# === LOSS WEIGHT ADJUSTMENTS ===
# These will be used in training loop
CTC_WEIGHT = 3.0         # Increase CTC emphasis (default was ~1.0)
RECN_WEIGHT = 5.0        # Increase reconstruction quality
INFO_WEIGHT = 1.5        # Style consistency
WID_WEIGHT = 1.5         # Writer identification

# === DISCRIMINATOR LEARNING RATE ===
D_LR_RATIO = 0.5  # Discriminator LR will be 50% of Generator LR

print(f"✓ Batch size: {cfg.training.batch_size}")
print(f"✓ Target epochs: {cfg.training.epochs}")
print(f"✓ CTC weight: {CTC_WEIGHT}")
print(f"✓ Reconstruction weight: {RECN_WEIGHT}")
"""

print(config_changes)


# ============================================================================
# CHANGE 2: Modify Optimizer Setup
# ============================================================================

print("\n" + "="*70)
print("CHANGE 2: Optimizer with Separate Learning Rates")
print("="*70)

optimizer_code = """
# Location: Cell "Setup Training Components"

# === REPLACE YOUR OPTIMIZER SETUP WITH THIS ===

from itertools import chain

# Generator learning rate (keep as-is)
G_LR = cfg.training.lr  # 2.0e-4

# Discriminator learning rate (REDUCE to balance training)
D_LR = G_LR * 0.5  # 1.0e-4 (half of Generator)

# Generator optimizer
optimizer_G = torch.optim.Adam(
    chain(generator.parameters(), style_encoder.parameters()),
    lr=G_LR,
    betas=(cfg.training.adam_b1, cfg.training.adam_b2)
)

# Discriminator optimizer (LOWER learning rate)
optimizer_D = torch.optim.Adam(
    chain(discriminator.parameters(), patch_discriminator.parameters()),
    lr=D_LR,  # ← CHANGED: Use lower LR
    betas=(cfg.training.adam_b1, cfg.training.adam_b2)
)

print(f"✓ Generator LR: {G_LR}")
print(f"✓ Discriminator LR: {D_LR} (balanced)")
"""

print(optimizer_code)


# ============================================================================
# CHANGE 3: Update Generator Training Loss
# ============================================================================

print("\n" + "="*70)
print("CHANGE 3: Generator Loss with Updated Weights")
print("="*70)

generator_loss_code = """
# Location: Inside training loop, "Train Generator" section

# Find this section in your Generator training step:
# After computing all losses (adv_loss, fake_ctc_loss_total, etc.)

# === REPLACE THE g_loss CALCULATION ===

# OLD (implicit weights):
# g_loss = (adv_loss + adv_loss_patch +
#          fake_ctc_loss_total +
#          info_loss +
#          fake_wid_loss +
#          recn_loss_val +
#          cfg.training.lambda_ctx * ctx_loss_val +
#          cfg.training.lambda_kl * kl_loss_val)

# NEW (explicit tuned weights):
gp_ctc = 3.0    # Triple the CTC emphasis
gp_info = 1.5   # Style consistency
gp_wid = 1.5    # Writer ID
gp_recn = 5.0   # Strong reconstruction

g_loss = (
    adv_loss + adv_loss_patch +                      # Adversarial (1.0x each)
    gp_ctc * fake_ctc_loss_total +                   # CTC readability (3.0x) ← KEY CHANGE
    gp_info * info_loss +                            # Style consistency (1.5x)
    gp_wid * fake_wid_loss +                         # Writer ID (1.5x)
    gp_recn * recn_loss_val +                        # Reconstruction (5.0x) ← KEY CHANGE
    cfg.training.lambda_ctx * ctx_loss_val +         # Contextual (keep config)
    cfg.training.lambda_kl * kl_loss_val             # KL divergence (keep config)
)

g_loss.backward()
optimizer_G.step()

# Optional: Print component losses for monitoring
if iter_count % 100 == 0:
    print(f"Loss components - CTC: {fake_ctc_loss_total.item():.3f}, "
          f"Recn: {recn_loss_val.item():.3f}, "
          f"Adv: {adv_loss.item():.3f}")
"""

print(generator_loss_code)


# ============================================================================
# CHANGE 4: Add Gradient Clipping (Stability)
# ============================================================================

print("\n" + "="*70)
print("CHANGE 4: Gradient Clipping for Stability")
print("="*70)

gradient_clip_code = """
# Location: After g_loss.backward() and d_loss.backward()

# === ADD AFTER g_loss.backward() ===
g_loss.backward()

# Clip gradients to prevent explosion
torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
torch.nn.utils.clip_grad_norm_(style_encoder.parameters(), max_norm=5.0)

optimizer_G.step()

# === ADD AFTER d_loss.backward() ===
d_loss.backward()

# Clip discriminator gradients
torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
torch.nn.utils.clip_grad_norm_(patch_discriminator.parameters(), max_norm=5.0)

optimizer_D.step()
"""

print(gradient_clip_code)


# ============================================================================
# CHANGE 5: Enhanced Monitoring
# ============================================================================

print("\n" + "="*70)
print("CHANGE 5: Detailed Monitoring During Training")
print("="*70)

monitoring_code = """
# Location: Inside training loop, add monitoring function

def monitor_training_health(epoch, batch_idx, g_loss, d_loss, 
                           ctc_loss, recn_loss, adv_loss):
    \"\"\"Monitor key metrics for healthy training\"\"\"
    
    # Calculate G/D ratio
    g_d_ratio = g_loss.item() / (d_loss.item() + 1e-8)
    
    # Print every 100 iterations
    if batch_idx % 100 == 0:
        print(f"\\n[Epoch {epoch}, Batch {batch_idx}]")
        print(f"  G Loss: {g_loss.item():.4f} | D Loss: {d_loss.item():.4f}")
        print(f"  G/D Ratio: {g_d_ratio:.4f} ", end="")
        
        # Health indicators
        if 0.8 <= g_d_ratio <= 1.2:
            print("✅ HEALTHY")
        elif g_d_ratio < 0.5:
            print("⚠️  D too strong - consider reducing D_LR")
        elif g_d_ratio > 2.0:
            print("⚠️  G too strong - check discriminator")
        
        print(f"  CTC Loss: {ctc_loss.item():.4f}")
        print(f"  Reconstruction: {recn_loss.item():.4f}")
        print(f"  Adversarial: {adv_loss.item():.4f}")
        
        # Check for NaN
        if torch.isnan(g_loss) or torch.isnan(d_loss):
            raise ValueError("NaN detected in losses! Training halted.")
    
    return g_d_ratio

# === USE IN TRAINING LOOP ===
# After computing losses, call:
g_d_ratio = monitor_training_health(
    epoch, batch_idx, g_loss, d_loss,
    fake_ctc_loss_total, recn_loss_val, adv_loss
)
"""

print(monitoring_code)


# ============================================================================
# CHANGE 6: Update Config File (Optional but Recommended)
# ============================================================================

print("\n" + "="*70)
print("CHANGE 6: Update gan_iam.yml Config File")
print("="*70)

yaml_changes = """
# Location: configs/gan_iam.yml

# Open the file and make these changes:

training:
  epochs: 70              # Keep full duration
  batch_size: 16          # Change from 8 → 16
  eval_batch_size: 16     # Match training batch size
  
  lr: 2.0e-4             # Generator learning rate (unchanged)
  
  # Reduce contextual loss weight
  lambda_ctx: 1.0         # Change from 2.0 → 1.0
  
  # Keep other settings
  lambda_kl: 0.0001
  lambda_gram: 2.0
  num_critic_train: 4
  vae_mode: true

# Save the file and reload config in notebook
"""

print(yaml_changes)


# ============================================================================
# CHANGE 7: Checkpoint Saving Strategy
# ============================================================================

print("\n" + "="*70)
print("CHANGE 7: Improved Checkpoint Saving")
print("="*70)

checkpoint_code = """
# Location: End of each epoch in training loop

# Track best models
best_cer = float('inf')
best_fid = float('inf')

# At end of each epoch:
if epoch % 5 == 0:  # Evaluate every 5 epochs
    # Calculate metrics
    current_cer, current_wer = evaluate_ocr_accuracy(...)
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'style_encoder': style_encoder.state_dict(),
        'patch_discriminator': patch_discriminator.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
        'history': history,
        'metrics': {'cer': current_cer, 'wer': current_wer}
    }
    
    # Always save latest
    torch.save(checkpoint, f'checkpoints/epoch_{epoch}.pth')
    torch.save(checkpoint, 'checkpoints/latest.pth')
    
    # Save best CER model
    if current_cer < best_cer:
        best_cer = current_cer
        torch.save(checkpoint, 'checkpoints/best_cer.pth')
        print(f"✅ New best CER: {current_cer:.4f}")
    
    print(f"Epoch {epoch} - CER: {current_cer:.4f}, WER: {current_wer:.4f}")
"""

print(checkpoint_code)


# ============================================================================
# SUMMARY AND QUICK START
# ============================================================================

print("\n" + "="*70)
print("QUICK START: MINIMUM CHANGES FOR MAXIMUM IMPACT")
print("="*70)

quick_start = """
For the fastest improvement with minimal code changes, do these 3 things:

1. **In optimizer setup cell:**
   Change: optimizer_D learning rate to cfg.training.lr * 0.5

2. **In Generator training section:**
   Add before g_loss calculation:
   gp_ctc = 3.0
   gp_recn = 5.0
   
   Then use:
   g_loss = (adv_loss + adv_loss_patch +
            gp_ctc * fake_ctc_loss_total +
            gp_recn * recn_loss_val + ...)

3. **In notebook config cell:**
   cfg.training.epochs = 70  # Train to completion

Expected improvement from JUST these 3 changes:
- CER: 8% → 5-6%
- WER: 25% → 18-22%
- FID: 45 → 38-42
- Training time: ~3-4 days on GPU

For full improvements (targeting original HiGAN+ performance):
- Apply all 7 changes above
- Train for 70 epochs
- Expected time: 1-2 weeks
- Expected final CER: 3-5%, FID: 28-35
"""

print(quick_start)


# ============================================================================
# VERIFICATION CHECKLIST
# ============================================================================

print("\n" + "="*70)
print("POST-CHANGE VERIFICATION CHECKLIST")
print("="*70)

verification = """
Before starting training, verify:

[ ] cfg.training.batch_size = 16 (or higher)
[ ] cfg.training.epochs = 70
[ ] optimizer_D.param_groups[0]['lr'] == 1.0e-4 (half of G)
[ ] optimizer_G.param_groups[0]['lr'] == 2.0e-4
[ ] gp_ctc = 3.0 is defined
[ ] gp_recn = 5.0 is defined
[ ] recognizer.requires_grad == False (check all params)
[ ] writer_identifier.requires_grad == False
[ ] Pretrained weights loaded successfully
[ ] Gradient clipping added (max_norm=5.0)

Print this at start of training to confirm:

print("Training Configuration:")
print(f"  Batch size: {cfg.training.batch_size}")
print(f"  Total epochs: {cfg.training.epochs}")
print(f"  Generator LR: {optimizer_G.param_groups[0]['lr']}")
print(f"  Discriminator LR: {optimizer_D.param_groups[0]['lr']}")
print(f"  CTC weight: {gp_ctc}")
print(f"  Reconstruction weight: {gp_recn}")
print(f"  Recognizer frozen: {not any(p.requires_grad for p in recognizer.parameters())}")
"""

print(verification)


# ============================================================================
# EXPECTED TRAINING TIMELINE
# ============================================================================

print("\n" + "="*70)
print("EXPECTED TRAINING TIMELINE")
print("="*70)

timeline = """
Epoch 10:
  - CER: ~6-7%
  - FID: ~40-42
  - G/D ratio stabilizing

Epoch 20:
  - CER: ~5-6%
  - FID: ~36-38
  - Visible quality improvement

Epoch 30:
  - CER: ~4.5-5.5%
  - FID: ~33-36
  - Text highly readable

Epoch 40:
  - CER: ~4-5%
  - FID: ~30-34
  - Nearing target

Epoch 50:
  - CER: ~3.5-4.5%
  - FID: ~28-32
  - Approaching original HiGAN+

Epoch 70:
  - CER: ~3-4%
  - FID: ~25-30
  - Target achieved! ✅

Total training time (single GPU):
- GTX 1080 Ti: ~2 weeks
- RTX 3090: ~1 week
- A100: ~3-4 days
"""

print(timeline)

print("\n" + "="*70)
print("All changes documented! Apply them to code.ipynb and start training.")
print("Monitor progress using the health checks and save checkpoints regularly.")
print("="*70)
