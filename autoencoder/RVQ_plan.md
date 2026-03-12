# Residual Vector Quantization (RVQ) — Extending VQ-VAE

## What is RVQ?

Standard VQ uses a **single codebook** to quantize the encoder output. RVQ uses **multiple codebooks in sequence**, where each codebook quantizes the **residual** (error) left by the previous one.

This is the core technique behind SoundStream, EnCodec, and other modern neural codecs.

## Algorithm

```
residual = encoder_output
quantized_total = 0

for each codebook_i in [codebook_0, codebook_1, ..., codebook_N]:
    quantized_i = codebook_i.quantize(residual)     # nearest neighbor lookup
    quantized_total += quantized_i
    residual = encoder_output - quantized_total      # remaining error
```

Each layer captures progressively finer details:
- **Layer 0**: coarse structure (dominant features)
- **Layer 1**: medium details missed by layer 0
- **Layer 2+**: fine-grained corrections

## Why RVQ over single VQ?

| Aspect | Single VQ | RVQ (N layers) |
|--------|-----------|-----------------|
| Total codebook capacity | M entries | M entries x N layers |
| Representable vectors | M | M^N (combinatorial) |
| Bitrate | log2(M) bits | N x log2(M) bits |
| Quality | Limited by single codebook | Progressive refinement |

With M=512 and N=4 layers, RVQ can represent 512^4 ≈ 68 billion unique vectors, vs just 512 with single VQ.

## Changes Needed to Current Notebook

### 1. New hyperparameter (cell 1)

```python
n_residual_layers = 4   # number of RVQ layers
n_embeddings = 128      # can reduce per-codebook size since we have multiple
```

### 2. New `ResidualVQ` class (after `VQEmbeddingEMA`)

```python
class ResidualVQ(nn.Module):
    def __init__(self, n_layers, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999):
        super().__init__()
        self.n_layers = n_layers
        self.codebooks = nn.ModuleList([
            VQEmbeddingEMA(n_embeddings, embedding_dim, commitment_cost, decay)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        residual = x
        quantized_total = torch.zeros_like(x)
        total_commitment_loss = 0
        total_codebook_loss = 0
        total_perplexity = 0

        for codebook in self.codebooks:
            quantized, commitment_loss, codebook_loss, perplexity = codebook(residual)
            quantized_total = quantized_total + quantized
            residual = x - quantized_total  # recompute from original to avoid error drift
            total_commitment_loss += commitment_loss
            total_codebook_loss += codebook_loss
            total_perplexity += perplexity

        # Average losses across layers
        n = self.n_layers
        return quantized_total, total_commitment_loss / n, total_codebook_loss / n, total_perplexity / n

    def encode(self, x):
        residual = x
        quantized_total = torch.zeros_like(x)
        all_indices = []

        for codebook in self.codebooks:
            quantized, indices = codebook.encode(residual)
            quantized_total = quantized_total + quantized
            residual = x - quantized_total
            all_indices.append(indices)

        return quantized_total, all_indices  # list of indices per layer

    def retrieve_random_codebook(self, random_indices_list):
        """random_indices_list: list of index tensors, one per layer"""
        quantized_total = None
        for codebook, random_indices in zip(self.codebooks, random_indices_list):
            quantized = codebook.retrieve_random_codebook(random_indices)
            if quantized_total is None:
                quantized_total = quantized
            else:
                quantized_total = quantized_total + quantized
        return quantized_total
```

### 3. Model instantiation (cell 10)

```python
# Replace:
codebook = VQEmbeddingEMA(n_embeddings=n_embeddings, embedding_dim=latent_dim)

# With:
codebook = ResidualVQ(n_layers=n_residual_layers, n_embeddings=n_embeddings, embedding_dim=latent_dim)
```

The `Model` class needs **no changes** — it already calls `self.codebook(z)` generically.

### 4. Random sample generation (cell 22)

```python
def draw_random_sample_image(codebook, decoder, indices_shape):
    random_indices_list = [
        torch.floor(torch.rand(indices_shape) * n_embeddings).long().to(DEVICE)
        for _ in range(codebook.n_layers)
    ]
    codes = codebook.retrieve_random_codebook(random_indices_list)
    x_hat = decoder(codes.to(DEVICE))

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Visualization of Random Codes (RVQ)")
    plt.imshow(np.transpose(make_grid(x_hat.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
```

## Key Design Decisions

- **Recompute residual from original** (`residual = x - quantized_total`) rather than subtracting layer-by-layer — avoids floating point error accumulation
- **Average losses** across layers so the total loss magnitude stays similar to single VQ
- **Each codebook is independent** — they don't share parameters, and each has its own EMA state
- The straight-through estimator works the same way in each layer
