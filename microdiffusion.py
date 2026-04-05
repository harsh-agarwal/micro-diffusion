"""
microdiffusion.py
=================
A minimal, self-contained implementation of Denoising Diffusion Probabilistic Models
(DDPMs) — the algorithm behind Stable Diffusion, DALL-E, and Imagen — in one file
with zero "magic". Every line of math is explained in plain English.

──────────────────────────────────────────────────────────────────────────────────
  THE BIG IDEA
──────────────────────────────────────────────────────────────────────────────────
  A diffusion model is a generative model that learns to REVERSE a noise process.

  1. FORWARD  (easy, fixed, no learning):
     Take a real image x_0. Add a tiny amount of Gaussian noise. Repeat T times.
     After T steps you have x_T, which is indistinguishable from N(0, I).
     The image has been completely destroyed.

  2. REVERSE  (hard, learned):
     Train a neural network to undo one denoising step at a time.
     At generation time, start from pure noise x_T ~ N(0,I) and run the
     network T times, each time removing a bit of noise, until you reach x_0.

  The key mathematical insight: we can write down the EXACT distribution of the
  noisy image x_t at ANY timestep t directly from x_0, without simulating all
  the steps in between. This lets training be O(1) per batch regardless of T.

──────────────────────────────────────────────────────────────────────────────────
  WHAT YOU WILL LEARN
──────────────────────────────────────────────────────────────────────────────────
  Ch 1  Noise schedule    — β_t, α_t, ᾱ_t: the math of gradual destruction
  Ch 2  Synthetic data    — generating circles & squares in pure numpy
  Ch 3  The model         — a 3-layer MLP that predicts the noise at each step
  Ch 4  Forward process   — q(x_t|x_0): the reparameterization trick, derived
  Ch 5  Training          — the simplified ELBO objective; why it works
  Ch 6  Reverse sampling  — p_θ(x_{t-1}|x_t): deriving the reverse formula
  Ch 7  Visualization     — forward process, loss curve, real vs generated

  Reference: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
             https://arxiv.org/abs/2006.11239

  Usage:        python microdiffusion.py
  Dependencies: torch, matplotlib, numpy  (no torchvision, no huggingface)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# CHAPTER 0 — SETUP & HYPERPARAMETERS
# ==============================================================================
# All "magic numbers" live here so you can experiment by changing one line.

IMG_SIZE = 16          # 16×16 grayscale images
IMG_DIM  = IMG_SIZE * IMG_SIZE   # 256 pixels, flattened into a vector

# ── Diffusion timesteps ───────────────────────────────────────────────────────
# The original DDPM paper uses T=1000. We use T=20 for speed while keeping the
# math identical. We compensate with a more aggressive noise schedule (see Ch 1)
# so the image is fully destroyed in 20 steps instead of 1000.
# Rule of thumb: you need ᾱ_T ≈ 0 (nearly all signal destroyed by the last step).
T = 20

# ── Training ──────────────────────────────────────────────────────────────────
N_SAMPLES  = 1024    # total training images (512 circles + 512 squares)
BATCH_SIZE = 128
EPOCHS     = 500
LR         = 3e-4    # Adam default; works well for small models

# ── Hardware ──────────────────────────────────────────────────────────────────
# MPS = Metal Performance Shaders (Apple Silicon GPU). ~3x faster than CPU.
# Falls back to CPU on non-Apple machines. CUDA would be: 'cuda' if available.
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)


# ==============================================================================
# CHAPTER 1 — THE NOISE SCHEDULE
# ==============================================================================
#
# ── What is a noise schedule? ─────────────────────────────────────────────────
# We add noise in T small steps rather than all at once. The "schedule" controls
# how much noise is added at each step. We define:
#
#   β_t  ∈ (0,1)   "beta"  — the VARIANCE of the noise added at step t
#                            (β_t = 0.5 means we inject noise with std = √0.5 ≈ 0.7)
#
# ── Derived quantities ────────────────────────────────────────────────────────
#
#   α_t  = 1 - β_t          "alpha"  — fraction of signal KEPT at step t
#                                       (α_t = 0.5 → half the signal survives)
#
#   ᾱ_t  = ∏_{s=1}^{t} α_s  "alpha-bar" — CUMULATIVE fraction of signal kept
#                                           from step 1 all the way to step t
#                                           This is the key quantity for training.
#
# ── Why cumprod? The Markov chain multiplication ──────────────────────────────
# The forward process is a Markov chain. At each step:
#   x_t = √α_t · x_{t-1} + √β_t · ε_t     (ε_t ~ N(0,I) i.i.d.)
#
# After 2 steps:
#   x_2 = √α_2 · x_1 + √β_2 · ε_2
#       = √α_2 · (√α_1 · x_0 + √β_1 · ε_1) + √β_2 · ε_2
#       = √(α_1·α_2) · x_0  +  √(α_2·β_1)·ε_1 + √β_2·ε_2
#
# The noise terms combine as a single Gaussian (sum of independent Gaussians):
#   variance = α_2·β_1 + β_2 = α_2(1-α_1) + (1-α_2) = 1 - α_1·α_2 = 1 - ᾱ_2
#
# So after 2 steps:  x_2 = √ᾱ_2 · x_0 + √(1-ᾱ_2) · ε,   ε ~ N(0,I)
# After t steps:     x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε     ← THE KEY FORMULA
#
# The coefficient √ᾱ_t is the square root of the cumulative product of all alphas.
# That is exactly what torch.cumprod computes.
#
# ── The signal-to-noise ratio (SNR) ──────────────────────────────────────────
# At timestep t, the noisy image x_t has:
#   signal power  = ᾱ_t       (the original image contributes ᾱ_t of the variance)
#   noise power   = 1 - ᾱ_t   (the injected noise contributes the rest)
#   SNR(t) = ᾱ_t / (1 - ᾱ_t)
#
# We want SNR(T) ≈ 0 so x_T is pure noise, indistinguishable from N(0,I).
#
# ── Our schedule: LINEAR from β_1=0.02 to β_T=0.50 ───────────────────────────
# With T=20, a mild schedule (like the paper's β ∈ [1e-4, 0.02]) gives ᾱ_T≈0.98
# — the image is barely touched. We need an aggressive schedule:
#   β_1 = 0.02  (small noise early, preserves structure)
#   β_T = 0.50  (large noise late, ensures full destruction)
# This gives ᾱ_20 ≈ 0.0016 → only 0.16% of the original signal survives at t=20.
#
# (Production: DDPM uses β ∈ [1e-4, 0.02] with T=1000; Improved DDPM uses a
#  cosine schedule. Both achieve ᾱ_T ≈ 0 via different paths.)

betas     = torch.linspace(0.02, 0.50, T)          # (T,)  β_1, β_2, ..., β_T
alphas    = 1.0 - betas                             # (T,)  α_t = 1 - β_t
alpha_bar = torch.cumprod(alphas, dim=0)            # (T,)  ᾱ_t = α_1·α_2·...·α_t

# ── Precompute square-roots used in forward and reverse formulas ───────────────
# We compute these once here (not inside the training loop) for efficiency.
# Each is a (T,) tensor; indexing with a batch of t values gives the coefficients.
sqrt_alpha_bar           = torch.sqrt(alpha_bar)            # √ᾱ_t   (signal coeff in forward)
sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)      # √(1-ᾱ_t) (noise coeff in forward)
sqrt_alphas              = torch.sqrt(alphas)               # √α_t   (denominator in reverse)
sqrt_betas               = torch.sqrt(betas)                # √β_t   (std dev in reverse step)

# Move all schedule constants to device once (avoids repeated .to(device) calls)
betas                    = betas.to(device)
alpha_bar                = alpha_bar.to(device)
sqrt_alpha_bar           = sqrt_alpha_bar.to(device)
sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.to(device)
sqrt_alphas              = sqrt_alphas.to(device)
sqrt_betas               = sqrt_betas.to(device)

# Print the full schedule — reading this table builds intuition for all the math.
print("\nNoise schedule (T=20 steps):")
print(f"{'t':>4} | {'β_t':>8} | {'ᾱ_t':>12} | {'signal%':>9} | {'SNR':>8}")
print("-" * 54)
for t_idx in range(T):
    ab  = alpha_bar[t_idx].item()
    snr = ab / (1.0 - ab + 1e-8)
    print(f"{t_idx+1:>4} | {betas[t_idx].item():>8.4f} | {ab:>12.6f} | {100*ab:>8.2f}% | {snr:>8.4f}")
print()


# ==============================================================================
# CHAPTER 2 — SYNTHETIC DATASET
# ==============================================================================
#
# ── Why synthetic data? ───────────────────────────────────────────────────────
# We want zero dependencies. Circles and squares are simple enough that the
# model can learn the distribution in minutes, but structured enough to verify
# the generated samples actually look like shapes (not random noise blobs).
#
# ── Normalization to [-1, +1] — WHY this matters ─────────────────────────────
# Look at the forward process formula: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
# At t=T (pure noise):  x_T ≈ ε ~ N(0, I)
#   → values centered at 0, standard deviation 1
#
# If our clean images x_0 are in [0,1]:
#   At t=T: x_T ≈ √(0.0016)·x_0 + √(0.9984)·ε ≈ 0.04·x_0 + 0.9992·ε
#   The tiny x_0 term biases x_T toward [0,1] instead of centering on 0.
#   Result: the noise prior N(0,I) and the data distribution live on different
#           scales, making the reverse process harder to learn.
#
# If we normalize x_0 to [-1,+1]:
#   x_0 is symmetric around 0, same as N(0,I). The two distributions align,
#   and the model has a much easier job at high noise levels.
#   This is standard in all diffusion model implementations.

def make_circle(size=IMG_SIZE):
    """
    Generate a random filled circle on a black background.
    The circle center and radius are randomized to give diversity.
    Returns float32 array of shape (size, size) with values in [-1, +1].
    """
    img = np.zeros((size, size), dtype=np.float32)
    cx  = np.random.uniform(size * 0.3, size * 0.7)   # center x
    cy  = np.random.uniform(size * 0.3, size * 0.7)   # center y
    r   = np.random.uniform(size * 0.2, size * 0.35)  # radius
    # np.mgrid creates a 2D grid of pixel coordinates; we use the distance formula
    # (x-cx)^2 + (y-cy)^2 <= r^2  to fill pixels inside the circle with 1.0
    gy, gx = np.mgrid[0:size, 0:size]
    mask = (gx - cx) ** 2 + (gy - cy) ** 2 <= r ** 2
    img[mask] = 1.0
    return img * 2.0 - 1.0    # remap [0,1] → [-1,+1]

def make_square(size=IMG_SIZE):
    """
    Generate a random filled square on a black background.
    The position and size are randomized.
    Returns float32 array of shape (size, size) with values in [-1, +1].
    """
    s  = np.random.randint(int(size * 0.25), int(size * 0.55))  # side length
    x0 = np.random.randint(1, size - s - 1)                     # top-left x
    y0 = np.random.randint(1, size - s - 1)                     # top-left y
    img = np.zeros((size, size), dtype=np.float32)
    img[y0:y0 + s, x0:x0 + s] = 1.0
    return img * 2.0 - 1.0

def generate_dataset(n=N_SAMPLES):
    """
    Build N training images: n/2 circles + n/2 squares, shuffled.
    Pre-flatten each image to a 256-dim vector so the MLP can consume it directly.
    Output shape: (N, 256) float32 tensor.
    """
    half = n // 2
    data = [make_circle() for _ in range(half)] + [make_square() for _ in range(half)]
    np.random.shuffle(data)
    return torch.tensor(np.stack([d.ravel() for d in data]), dtype=torch.float32)

dataset    = generate_dataset()   # (1024, 256) — each row is one flattened image
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    # drop_last=True: discard the final partial batch so all batches have size BATCH_SIZE.
    # This simplifies shape reasoning and avoids edge cases in the training loop.
)
print(f"Dataset: {dataset.shape[0]} images of {IMG_SIZE}×{IMG_SIZE} pixels\n")

# ── Figure A helper ───────────────────────────────────────────────────────────
# Defined here, called in Chapter 4 (after q_sample is defined).
# We want to show the forward process BEFORE training, so the viewer understands
# what the model will learn to undo — but q_sample is needed to draw it.
def _show_forward_process():
    """
    Visualize one image at 6 noise levels: clean, light noise, ..., pure noise.
    Each column applies q(x_t|x_0) = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε with a different t.
    """
    sample     = dataset[0].to(device)
    show_steps = [0, 4, 8, 12, 16, 19]    # t indices to display (0-indexed → t=1..20)

    fig, axes = plt.subplots(1, len(show_steps), figsize=(14, 2.8))
    fig.suptitle(
        "Figure A — Forward Diffusion Process:  x_0 (clean)  →  x_T (pure noise)\n"
        "Formula:  x_t = √ᾱ_t · x_0  +  √(1-ᾱ_t) · ε      ε ~ N(0,I)\n"
        "As ᾱ_t → 0, the signal vanishes and x_t becomes indistinguishable from N(0,I)",
        fontsize=8.5
    )
    for ax, t_idx in zip(axes, show_steps):
        noise    = torch.randn_like(sample)
        t_tensor = torch.tensor([t_idx], device=device)
        x_t      = q_sample(sample.unsqueeze(0), t_tensor, noise.unsqueeze(0))
        img      = x_t.squeeze().cpu().reshape(IMG_SIZE, IMG_SIZE).numpy()
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        ab  = alpha_bar[t_idx].item()
        snr = ab / (1.0 - ab + 1e-8)
        ax.set_title(f"t={t_idx+1}\nᾱ={ab:.3f}\nSNR={snr:.2f}", fontsize=7)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("forward_process.png", dpi=120, bbox_inches='tight')
    plt.show()
    plt.close('all')
    print("Saved: forward_process.png\n")


# ==============================================================================
# CHAPTER 3 — THE MODEL: ε_θ(x_t, t)
# ==============================================================================
#
# ── What does the model need to do? ──────────────────────────────────────────
# Given a noisy image x_t and the timestep t, predict the noise ε that was
# mixed into x_t. In other words, learn the function:
#
#   ε_θ(x_t, t)  ≈  ε   where x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
#
# If we can predict ε, we can reconstruct x_0:
#   x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t
# And then compute the reverse step mean (see Chapter 6).
#
# ── Architecture: 3-layer MLP ────────────────────────────────────────────────
#   Input  : [x_t_flat | t_onehot]   shape (B, 256 + 20) = (B, 276)
#   Layer 1: Linear(276, 256) → GELU
#   Layer 2: Linear(256, 256) → GELU
#   Layer 3: Linear(256, 256)          ← output: predicted ε, shape (B, 256)
#
# Why 256 hidden dim?  Same as IMG_DIM — the model has enough capacity to
# represent spatial noise patterns without over-parameterizing for this task.
#
# Why 3 layers (not 2, not 4)?
#   2 layers: too little capacity to capture the noise-vs-structure separation.
#   4 layers: risks overfitting on only 1024 training images.
#   3 is the sweet spot for this dataset size.
#
# Why GELU (not ReLU)?
#   ReLU's hard zero-cutoff creates gradient discontinuities that slow learning
#   when outputs are near zero (common in noise prediction). GELU is smooth
#   everywhere — f(x) = x·Φ(x) where Φ is the standard normal CDF — and
#   empirically converges faster on diffusion tasks.
#
# Why no BatchNorm or LayerNorm?
#   BatchNorm makes outputs depend on OTHER samples in the batch, which means
#   the model's behavior changes between training (batch stats) and eval (running
#   stats). For a simple model this size, skip it — training is stable without it.
#
# ── Timestep embedding: ONE-HOT (why not sinusoidal?) ────────────────────────
# The model MUST know which timestep it's at. Different t values mean different
# noise levels, so the optimal denoising function is different for each t.
#
# With T=20, we use a 20-dim one-hot vector: t=5 → [0,0,0,0,1,0,...,0]
# This is transparent: a human reading the vector immediately knows the timestep.
#
# Production models (T=1000) use SINUSOIDAL embeddings (as in Transformers):
#   PE(t, 2i)   = sin(t / 10000^(2i/d))
#   PE(t, 2i+1) = cos(t / 10000^(2i/d))
# because a 1000-dim one-hot would be wasteful and sparse. With T=20, one-hot
# is both efficient and pedagogically clearer.
#
# Why concatenate instead of add?
#   Adding t_emb to x_t would require them to have the same dimension (256),
#   padding/projecting one of them. Concatenation is simpler and preserves both
#   signals independently — the first 256 dimensions are always "image", the
#   last T dimensions are always "time", with no mixing by construction.

class NoisePredictorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG_DIM + T, 256),  # 256 image pixels + 20 time dims → 256 hidden
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, IMG_DIM),      # output: predicted noise, same shape as image
        )

    def forward(self, x_t, t):
        """
        Forward pass: concatenate noisy image with one-hot timestep, run through MLP.

        x_t  : (B, 256)  — flattened noisy images at timestep t
        t    : (B,)      — integer timestep indices in {0, ..., T-1}

        Returns ε_pred : (B, 256) — predicted noise that was added to get x_t
        """
        # One-hot encode t: creates a (B, T) matrix with a single 1 per row.
        # e.g. t=3, T=20 → [0,0,0,1,0,0,...,0]  (index 3 is 1, rest are 0)
        t_emb = F.one_hot(t, num_classes=T).float()   # (B, 20)
        inp   = torch.cat([x_t, t_emb], dim=1)        # (B, 276)  ← concat along feature dim
        return self.net(inp)                           # (B, 256)

total_params = sum(p.numel() for p in NoisePredictorMLP().parameters())
print(f"Model: 3-layer MLP  |  {total_params:,} parameters\n")


# ==============================================================================
# CHAPTER 4 — THE FORWARD PROCESS: q(x_t | x_0)
# ==============================================================================
#
# ── The Markov chain ──────────────────────────────────────────────────────────
# Each forward step adds a little Gaussian noise:
#
#   q(x_t | x_{t-1}) = N(x_t;  √(1-β_t)·x_{t-1},  β_t·I)
#
# In sampled form:  x_t = √(1-β_t)·x_{t-1} + √β_t·ε,   ε ~ N(0,I)
#                       = √α_t · x_{t-1}   + √(1-α_t)·ε  (since α_t = 1-β_t)
#
# ── THE REPARAMETERIZATION TRICK: jumping directly to any t ──────────────────
# Naively, to get x_t we'd simulate all t steps. But there's a closed form.
#
# Derivation (2 steps, then generalize):
#
#   x_1 = √α_1·x_0 + √(1-α_1)·ε_1                           ε_1 ~ N(0,I)
#
#   x_2 = √α_2·x_1 + √(1-α_2)·ε_2
#       = √α_2·[√α_1·x_0 + √(1-α_1)·ε_1] + √(1-α_2)·ε_2
#       = √(α_1α_2)·x_0  +  √α_2·√(1-α_1)·ε_1  +  √(1-α_2)·ε_2
#
# The two noise terms are independent Gaussians. Their SUM is also Gaussian
# (this is the key property of Gaussians: N(0,a²) + N(0,b²) = N(0, a²+b²)):
#
#   variance of noise = α_2·(1-α_1) + (1-α_2)
#                     = α_2 - α_1·α_2 + 1 - α_2
#                     = 1 - α_1·α_2
#                     = 1 - ᾱ_2          (since ᾱ_2 = α_1·α_2)
#
# So:  x_2 = √ᾱ_2·x_0 + √(1-ᾱ_2)·ε   where ε ~ N(0,I)
#
# By induction, this holds for all t:
#
#   ┌─────────────────────────────────────────────────────────────────┐
#   │  q(x_t | x_0) = N(x_t;  √ᾱ_t · x_0,  (1-ᾱ_t)·I)            │
#   │                                                                 │
#   │  Sampled form:  x_t = √ᾱ_t · x_0  +  √(1-ᾱ_t) · ε           │
#   │                        ↑                    ↑                  │
#   │                   signal term          noise term               │
#   │                   shrinks to 0       grows to 1                 │
#   │                   as t → T           as t → T                  │
#   └─────────────────────────────────────────────────────────────────┘
#
# ── Why this is crucial for training ─────────────────────────────────────────
# During training we sample a RANDOM t for each batch element. With the closed
# form, we compute x_t for that t in ONE operation — no sequential simulation.
# Training is O(1) per step regardless of T. This is what makes DDPM efficient.

def q_sample(x0, t, noise):
    """
    Apply the closed-form forward process: corrupt x_0 to noise level t.

    Implements:  x_t = √ᾱ_t · x_0  +  √(1-ᾱ_t) · ε

    x0    : (B, 256)  — clean images, values in [-1, +1]
    t     : (B,)      — integer timestep indices in {0,...,T-1} (0-indexed)
    noise : (B, 256)  — pre-sampled Gaussian noise ε ~ N(0,I)

    Returns x_t : (B, 256)  — noisy images at timestep t
    """
    # Index the precomputed schedule tensors with the batch of t values.
    # sqrt_alpha_bar[t] gives a (B,) tensor; .unsqueeze(1) → (B,1) for broadcasting.
    sa  = sqrt_alpha_bar[t].unsqueeze(1)            # (B, 1)  — √ᾱ_t
    soa = sqrt_one_minus_alpha_bar[t].unsqueeze(1)  # (B, 1)  — √(1-ᾱ_t)
    return sa * x0 + soa * noise                    # (B, 256) — broadcast over pixel dim

# ── Show the forward process NOW, before training ─────────────────────────────
# This is the first thing a new reader should see: *what are we learning to undo?*
print("Plotting Figure A: Forward Process (before training)...")
_show_forward_process()


# ==============================================================================
# CHAPTER 5 — TRAINING: the simplified ELBO objective
# ==============================================================================
#
# ── The objective: what are we maximizing? ───────────────────────────────────
# We want to maximize the log-likelihood log p_θ(x_0) — the probability our
# model assigns to the real data. This is intractable directly, so we maximize
# the Evidence Lower BOund (ELBO) instead:
#
#   ELBO = E_q[ log p_θ(x_0|x_1) ]
#          - Σ_{t=2}^{T} KL[ q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t) ]
#          - KL[ q(x_T|x_0) || p(x_T) ]
#
# The last term (KL with the prior p(x_T)=N(0,I)) is near-zero by design of our
# noise schedule, so we ignore it.
#
# Each KL term compares the TRUE reverse posterior q(x_{t-1}|x_t,x_0) (which is
# tractable because it conditions on x_0) to our model p_θ(x_{t-1}|x_t).
# Both are Gaussian; for two Gaussians with the same variance, KL reduces to
# a squared distance between their means.
#
# ── The simplified objective (Ho et al. Eq. 14) ──────────────────────────────
# After substituting the posterior mean formula (derived in Ch 6) and
# parameterizing p_θ via a noise predictor ε_θ(x_t, t), the ELBO simplifies to:
#
#   L_simple = E_{t, x_0, ε} [ || ε  -  ε_θ(x_t, t) ||² ]
#
# In words: for a random timestep t and a random noise ε, corrupt x_0 to get x_t,
# then ask the network to predict ε. Minimize the MSE. That's it.
#
# Ho et al. show empirically that this simplified form (which drops the 1/T
# weighting from the full ELBO) outperforms the full objective in practice.
#
# ── Why sample t randomly (not sequentially)? ────────────────────────────────
# We want the model to be good at ALL noise levels equally — both heavy noise
# (large t, near pure Gaussian) and light noise (small t, almost clean image).
# Sampling t ~ Uniform{0,...,T-1} ensures every level gets equal training time.
# Sequential t would bias training toward whichever levels appear most recently.
#
# ── Training Algorithm (Ho et al. Algorithm 1) ───────────────────────────────
# Repeat until convergence:
#   1.  x_0  ~ data distribution               (sample a batch of real images)
#   2.  t    ~ Uniform{0,...,T-1}              (random noise level per sample)
#   3.  ε    ~ N(0, I)                         (sample the noise to be predicted)
#   4.  x_t   = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε        (corrupt: closed-form forward jump)
#   5.  ε̂    = ε_θ(x_t, t)                    (predict: run the MLP)
#   6.  loss  = MSE(ε̂, ε)                      (compare predicted vs actual noise)
#   7.  backprop + Adam step

model     = NoisePredictorMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# Adam with lr=3e-4: the "Karpathy constant" — works reliably for small transformer-
# and MLP-scale models. We don't need a scheduler for 500 epochs at this model size.

CHECKPOINT = "microdiffusion_checkpoint.pt"

# ── Checkpoint: skip training if we already trained ───────────────────────────
# Delete microdiffusion_checkpoint.pt to force a full retrain.
if os.path.exists(CHECKPOINT):
    ckpt         = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model'])
    loss_history = ckpt['loss_history']
    print(f"Loaded checkpoint '{CHECKPOINT}'  (final loss = {loss_history[-1]:.4f})\n"
          f"Skipping training. Delete '{CHECKPOINT}' to retrain from scratch.\n")
else:
    loss_history = []
    print(f"Training for {EPOCHS} epochs on {device}...")
    print(f"(~{EPOCHS * (N_SAMPLES // BATCH_SIZE)} gradient steps total)\n")

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for x0 in dataloader:
            x0 = x0.to(device)               # (B, 256)  — batch of clean images
            B  = x0.shape[0]

            # ── Step 2: random timestep per sample ──────────────────────────
            # Each sample in the batch gets an INDEPENDENT random t.
            # This is important: a single batch simultaneously trains on images
            # corrupted at many different noise levels, giving diverse gradients.
            t = torch.randint(0, T, (B,), device=device)   # (B,)  integers in [0,T-1]

            # ── Step 3: sample the noise ε ~ N(0, I) ────────────────────────
            noise = torch.randn_like(x0)   # (B, 256)  — this is what the model must predict

            # ── Step 4: corrupt the images (forward process, one-shot) ──────
            x_t = q_sample(x0, t, noise)   # (B, 256)  — noisy images

            # ── Step 5: predict the noise ───────────────────────────────────
            eps_pred = model(x_t, t)        # (B, 256)  — predicted noise ε̂

            # ── Step 6: MSE loss ─────────────────────────────────────────────
            # loss = (1/N) Σ_i ||ε̂_i - ε_i||²
            # If loss=1.0: the model is predicting no better than random.
            # If loss≈0.05: the model has learned the noise distribution well.
            # (MSE=0 would mean perfect memorization — not the goal.)
            loss = F.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:>4}/{EPOCHS}  |  loss = {avg_loss:.4f}")

    torch.save({'model': model.state_dict(), 'loss_history': loss_history}, CHECKPOINT)
    print(f"\nCheckpoint saved → '{CHECKPOINT}'")

print("\nTraining complete!\n")


# ==============================================================================
# CHAPTER 6 — SAMPLING: the reverse process p_θ(x_{t-1} | x_t)
# ==============================================================================
#
# ── Where does the reverse formula come from? ────────────────────────────────
# We want p_θ(x_{t-1}|x_t): the distribution over "less noisy" x_{t-1} given x_t.
# We model this as a Gaussian. But what mean and variance should it have?
#
# The answer comes from Bayes' theorem applied to the FORWARD process.
# The TRUE posterior (conditioned on x_0) is tractable:
#
#   q(x_{t-1} | x_t, x_0)  =  N(x_{t-1};  μ̃_t(x_t, x_0),  β̃_t · I)
#
# where:
#   β̃_t   = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)       (posterior variance)
#
#   μ̃_t   = [ √ᾱ_{t-1}·β_t / (1-ᾱ_t) ]·x_0
#           + [ √α_t·(1-ᾱ_{t-1}) / (1-ᾱ_t) ]·x_t   (posterior mean)
#
# ── Substituting our noise prediction ────────────────────────────────────────
# We don't know x_0 during sampling. But we can estimate it from the forward
# process formula: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε  →  x̂_0 = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t
#
# Plugging x̂_0 into μ̃_t and simplifying (expanding, collecting terms):
#
#   μ_θ(x_t, t) = (1/√α_t) · [ x_t  -  β_t/√(1-ᾱ_t) · ε_θ(x_t,t) ]
#
# ── The reverse step ─────────────────────────────────────────────────────────
#
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │  x_{t-1} = (1/√α_t) · [ x_t  -  (β_t / √(1-ᾱ_t)) · ε_θ(x_t,t) ] │
#   │           +  √β_t · z                                               │
#   │                                                                     │
#   │  where z ~ N(0,I)  if t > 0                                         │
#   │        z = 0       if t = 0  (no noise at the final step)           │
#   └─────────────────────────────────────────────────────────────────────┘
#
# ── Why add noise z at every step except the last? ───────────────────────────
# The reverse process is stochastic by design. Adding z ~ N(0,I) scaled by √β_t
# means that two runs starting from different x_T samples (or the same x_T with
# different random seeds) produce different x_0 outputs — DIVERSE generations.
#
# Without z: all samples would follow a deterministic path and collapse toward
# the same output (the posterior mean). The model would lose its generative power.
#
# At t=0 (final step): we want the CLEAN image estimate, not another noisy version.
# Mathematically, the ELBO reconstruction term log p(x_0|x_1) is a deterministic
# mapping — there is no stochastic component in the final step.
#
# ── Variance choice: σ_t = √β_t ──────────────────────────────────────────────
# Ho et al. also consider σ_t = √β̃_t (the true posterior std). Both give
# similar FID scores. We use √β_t because it's one less computation — β̃_t
# requires α_bar_{t-1} and a ratio — and the difference is negligible at T=20.

@torch.no_grad()
def p_sample(model, x_t, t_scalar):
    """
    One reverse diffusion step: x_t → x_{t-1}.

    Implements:
        μ_θ(x_t,t) = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t,t))
        x_{t-1}    = μ_θ(x_t,t) + √β_t · z     (z=0 at t=0)

    model    : trained NoisePredictorMLP
    x_t      : (B, 256)  — current noisy images
    t_scalar : int        — current timestep in {0,...,T-1}

    Returns  x_{t-1} : (B, 256)  — slightly less noisy images
    """
    B       = x_t.shape[0]
    # Broadcast the scalar t to all batch elements so the model's one-hot
    # embedding works correctly for the whole batch.
    t_batch = torch.full((B,), t_scalar, device=device, dtype=torch.long)

    # ── Step A: predict the noise ε̂ = ε_θ(x_t, t) ────────────────────────────
    eps_pred = model(x_t, t_batch)   # (B, 256)

    # ── Step B: compute the reverse mean μ_θ ──────────────────────────────────
    # coeff1 = 1/√α_t   — rescales the prediction back to the clean image scale
    # coeff2 = β_t/√(1-ᾱ_t) — weights how much noise to subtract
    coeff1 = 1.0 / sqrt_alphas[t_scalar]
    coeff2 = betas[t_scalar] / sqrt_one_minus_alpha_bar[t_scalar]
    mean   = coeff1 * (x_t - coeff2 * eps_pred)   # (B, 256)

    # ── Step C: add stochasticity (except at the final step) ──────────────────
    if t_scalar == 0:
        return mean                              # deterministic final step
    else:
        z = torch.randn_like(x_t)               # z ~ N(0, I)
        return mean + sqrt_betas[t_scalar] * z   # σ_t = √β_t


@torch.no_grad()
def p_sample_loop(model, n_samples=8):
    """
    Full reverse diffusion: generate images from pure Gaussian noise.

    Starts from x_T ~ N(0,I) and applies p_sample() T times:
        x_T → x_{T-1} → ... → x_1 → x_0

    Each call to p_sample() uses the model to predict and subtract one
    "layer" of noise. After T steps, we have a (hopefully) clean image.

    Returns:
        final_images : (n_samples, 256)
        trajectory   : list of (n_samples, 256) tensors, length T+1
                       trajectory[0]  = x_T  (pure noise)
                       trajectory[T]  = x_0  (final generated image)
    """
    # Start from pure Gaussian noise — the model's "blank canvas"
    x          = torch.randn(n_samples, IMG_DIM, device=device)
    trajectory = [x.cpu().clone()]   # record every step for visualization

    # Denoise from t=T-1 down to t=0  (reversed(range(T)) = T-1, T-2, ..., 1, 0)
    for t_scalar in reversed(range(T)):
        x = p_sample(model, x, t_scalar)
        trajectory.append(x.cpu().clone())

    return x.cpu(), trajectory   # (n_samples,256),  list of T+1 tensors


# ==============================================================================
# CHAPTER 7 — VISUALIZATION
# ==============================================================================
# Three figures:
#   A  (already shown before training)  — forward noising process
#   B  — training loss curve (how well the model converged)
#   C  — real images vs generated images + one sample's denoising trajectory

model.eval()   # disable dropout/batchnorm in eval mode (not used here, but good habit)

# ── Figure B: Training Loss ────────────────────────────────────────────────────
# Log scale reveals both the fast early drop and the slow late improvement.
# A converging run typically drops from ~1.0 to ~0.1-0.3 over 500 epochs.
# If the loss plateaus above 0.5, the model hasn't learned much.
print("Plotting Figure B: Training Loss...")
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(loss_history, color='steelblue', linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss  ||ε̂ - ε||²  (log scale)")
ax.set_title(
    "Figure B — Training Loss\n"
    "Objective: L = E[||ε - ε_θ(√ᾱ_t·x_0 + √(1-ᾱ_t)·ε, t)||²]   (Ho et al. Eq.14)\n"
    "Lower loss = more accurate noise prediction = better image generation"
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=120, bbox_inches='tight')
plt.show()
plt.close('all')

# ── Figure C: Real vs Generated vs Denoising Trajectory ───────────────────────
# Row 0: real training images
# Row 1: fresh model-generated images (starting from pure noise)
# Row 2: the full denoising trajectory of ONE sample, from x_T (noise) to x_0 (clean)
#
# The trajectory row is the most instructive: you can literally watch the model
# "imagine" a shape out of noise, step by step.
print("Plotting Figure C: Real vs Generated Samples...")
n_show = 8

real_imgs = dataset[:n_show].reshape(n_show, IMG_SIZE, IMG_SIZE).numpy()

generated, trajectory = p_sample_loop(model, n_samples=n_show)
gen_imgs = generated.reshape(n_show, IMG_SIZE, IMG_SIZE).numpy()

# trajectory[0] = x_T (pure noise), trajectory[T] = x_0 (clean final image)
# Pick n_show evenly-spaced snapshots so the last column is always the final result.
traj_indices = [int(round(i * T / (n_show - 1))) for i in range(n_show)]
traj_imgs    = [trajectory[i][0].reshape(IMG_SIZE, IMG_SIZE).numpy() for i in traj_indices]
traj_labels  = [T - i for i in traj_indices]   # trajectory index → timestep label

fig, axes = plt.subplots(3, n_show, figsize=(14, 6.5))  # extra height for arrow
fig.suptitle(
    "Figure C — Real vs Generated Images\n"
    "Row 3 (trajectory): ONE sample denoising from x_T (pure noise) → x_0 (clean)\n"
    "Each column is one application of: x_{t-1} = (1/√α_t)·(x_t - β_t/√(1-ᾱ_t)·ε_θ) + √β_t·z",
    fontsize=8.5
)
for col in range(n_show):
    axes[0, col].imshow(real_imgs[col],  cmap='gray', vmin=-1, vmax=1)
    axes[0, col].axis('off')

    axes[1, col].imshow(gen_imgs[col],   cmap='gray', vmin=-1, vmax=1)
    axes[1, col].axis('off')

    axes[2, col].imshow(traj_imgs[col],  cmap='gray', vmin=-1, vmax=1)
    axes[2, col].text(0.5, -0.08, f"t={traj_labels[col]}",
                      transform=axes[2, col].transAxes,
                      ha='center', va='top', fontsize=7, color='#333333')
    axes[2, col].axis('off')

axes[0, 0].set_ylabel("Real",       fontsize=9, rotation=0, labelpad=35)
axes[1, 0].set_ylabel("Generated",  fontsize=9, rotation=0, labelpad=50)
axes[2, 0].set_ylabel("Trajectory\n(noise→clean)", fontsize=9, rotation=0, labelpad=60)

# Reserve bottom margin for the direction arrow, top margin for suptitle
plt.tight_layout(rect=[0, 0.08, 1, 0.97])

# ── Divider between "Generated" row and "Trajectory" row ─────────────────────
# A dashed line makes it visually clear that the trajectory row is a separate
# concept (one sample denoising over time) vs the generated samples above it.
# get_position() returns axes bounds in figure-fraction coordinates (0→1).
row1_y0 = axes[1, 0].get_position().y0   # bottom edge of the "Generated" row
row2_y1 = axes[2, 0].get_position().y1   # top edge of the "Trajectory" row
div_y   = (row1_y0 + row2_y1) / 2.0     # midpoint between the two rows
fig.add_artist(
    plt.Line2D([0.03, 0.97], [div_y, div_y],
               transform=fig.transFigure,
               color='#888888', linewidth=1.2, linestyle='--', zorder=10)
)

# ── Direction arrow below the "Trajectory" row ────────────────────────────────
# Shows the denoising direction: x_T (pure noise) ──────→ x_0 (clean image)
# This makes clear that time flows LEFT to RIGHT in the trajectory row,
# i.e. we START from noise on the left and END at the clean image on the right.
traj_y0 = axes[2, 0].get_position().y0   # bottom edge of trajectory row
arrow_y = traj_y0 - 0.04                 # sit just below the row

axes[2, 0].annotate('',
    xy     =(0.96, arrow_y), xycoords='figure fraction',
    xytext =(0.04, arrow_y), textcoords='figure fraction',
    arrowprops=dict(arrowstyle='->', color='#444444', lw=1.8)
)
fig.text(0.04, arrow_y - 0.01, 'x_T  — pure noise  (t=20)',
         ha='left',   va='top', fontsize=7.5, color='#444444')
fig.text(0.96, arrow_y - 0.01, 'x_0  — clean image  (t=0)',
         ha='right',  va='top', fontsize=7.5, color='#444444')
fig.text(0.50, arrow_y - 0.01, 'denoising direction',
         ha='center', va='top', fontsize=7.5, color='#666666', style='italic')

plt.savefig("generated_samples.png", dpi=120, bbox_inches='tight')
plt.show()
plt.close('all')

print("\nDone!")
print("Saved: forward_process.png, training_loss.png, generated_samples.png")
print()
print("=" * 62)
print("SUMMARY — The Full Diffusion Algorithm")
print("=" * 62)
print(f"""
FORWARD PROCESS  q(x_t|x_0):
  x_t = √ᾱ_t · x_0  +  √(1-ᾱ_t) · ε       ε ~ N(0,I)
  At t={T}: ᾱ_{T} ≈ 0.0016 → image is 99.84% noise.

TRAINING  L_simple (Ho et al. Eq.14):
  L = E[ ||ε - ε_θ(x_t, t)||² ]
  Predict the noise ε that was added.  MSE loss.
  Final loss: {loss_history[-1]:.4f}  (trained {len(loss_history)} epochs)

REVERSE PROCESS  p_θ(x_{{t-1}}|x_t):
  μ_θ  = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t,t))
  x_{{t-1}} = μ_θ + √β_t · z     (z~N(0,I) for t>0, z=0 for t=0)
  Start from x_T~N(0,I), apply {T} steps, arrive at a generated image.

MODEL:
  ε_θ(x_t, t) = MLP(x_t_flat ⊕ one_hot(t))
  Input: (B, {IMG_DIM + T}) → Hidden: (B, 256) × 2 → Output: (B, {IMG_DIM})
  {total_params:,} parameters total.

FURTHER READING:
  → Replace MLP with U-Net for spatial structure (add skip connections)
  → Use sinusoidal time embedding (needed when T=1000)
  → Try DDIM: deterministic sampling, fewer steps at inference
  → Classifier-free guidance: condition generation on a class label
  → Score-based generative models: the continuous-time generalization
""")
