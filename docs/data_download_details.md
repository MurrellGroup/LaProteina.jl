# Data Download Details

Detailed notes on how training data was sourced, filtered, and organized.

## Data Source

All training data comes from the **AlphaFold Protein Structure Database (AFDB)** — synthetic protein structures predicted by AlphaFold2. NVIDIA's La-Proteina release provides two curated ID lists (downloadable from NGC):

| File | IDs | Purpose |
|------|----:|---------|
| `AFDB_IDs-512.txt` | 344,507 | Short/medium proteins. Used for LD1, LD2 (up to 512 residues), AE1. |
| `AFDB_IDs-896.txt` | 46,942,694 | Full dataset including long proteins. Used for LD3 (300-800 residues), AE2. |

Both lists are at `/home/claudey/BFlaproteina/data/`. The IDs have the format `AF-{UniProtID}-F1-model_v4` but must be downloaded from AFDB as **v6** (v4 has been retired). Around 9% of IDs return 404 on v6, presumably removed from AFDB between versions.

## Clustering

FoldSeek structural clustering files are at `/home/claudey/BFlaproteina/data/afdb_clusters/`:

| File | Rows | Description |
|------|-----:|-------------|
| `cluster_reps.tsv` | 2,302,908 | Cluster representatives with metadata |
| `cluster_membership.tsv` | 30,045,247 | Maps every member → its cluster representative |
| `filtered_cluster_reps.tsv` | 910,510 | Subset after quality filtering |

### cluster_reps.tsv columns

```
UniProtID  flag  n_members  rep_length  mean_length  rep_pLDDT  mean_pLDDT  taxid
```

Example: `A0A6A4IZ81  0  4  386  387.25  89.69  91.1875  131567`

This metadata is critical for building smart subsets: length and pLDDT let us filter without downloading every structure.

### cluster_membership.tsv columns

```
member_id  representative_id  taxid
```

**Important**: Most member IDs are UniProt accessions that do NOT have AFDB predictions. Only IDs also present in `AFDB_IDs-896.txt` are actually downloadable from AlphaFold DB.

## Download 1: Short/Medium Proteins (512 list)

**When**: Early February 2026
**Script**: `BFlaproteina/scripts/download_afdb_laproteina.py`
**Source**: `AFDB_IDs-512.txt` (344,507 IDs)
**Result**: 289,317 structures downloaded to `~/shared_data/afdb_laproteina/raw/`
**Missing**: ~55k IDs returned 404 (AFDB v4→v6 attrition)
**Format**: `{UniProtID}.cif` (mmCIF)

Length distribution of the 289,317 downloaded files:
- Min: 33, Max: 1,026, Mean: 179, Median: 152
- 80.2% are ≤256 residues
- Only 22 are >512 (the list was curated for ≤512 but a few longer ones slipped in)

### Precomputed shards (AE1)

**Script**: `scripts/precompute_all_training_data.jl`
**Encoder**: AE1_ucond_512 (trained on ≤512 residues)
**Filter**: 30-256 residues → 231,971 proteins
**Output**: `~/shared_data/afdb_laproteina/precomputed_shards/train_shard_{01..10}.jld2`
**Total**: 2.5 GB, ~23k proteins per shard

## Download 2: Long Proteins (for LD3 fine-tuning)

**When**: March 2026
**Script**: `temporaries/download_long_proteins.py`
**Strategy**: Instead of downloading all 46.9M IDs from the 896 list (would need ~47 TB and months), we selected a diverse subset using FoldSeek clusters:

1. From `cluster_reps.tsv`, identified **288,642 cluster representatives** with:
   - Length 300-896 residues
   - pLDDT ≥ 70

2. From `cluster_membership.tsv`, for each of these clusters, selected up to **3 members** that are also present in `AFDB_IDs-896.txt` (i.e., actually downloadable from AFDB).

3. This yielded **437,698 candidate IDs** to download.

**Download stats**:
- ~90.5% hit rate (remaining 9.5% are 404 on AFDB v6)
- Expected yield: ~396k structures
- Stored in same raw dir: `~/shared_data/afdb_laproteina/raw/`
- Progress tracked in: `~/shared_data/afdb_laproteina/download_long_progress.json`

**Why 3 per cluster?** Just cluster reps alone gives ~133k long proteins. Adding 2 more per cluster (where available) gives ~400k — enough diversity for fine-tuning without the 47TB full download. The clusters have a median of 5 members, so 3 is a reasonable sample.

**Why not just download cluster reps?** Many cluster rep IDs from `cluster_reps.tsv` are not in the AFDB at all (the clustering was done on a broader UniProt set). Only IDs cross-referenced with `AFDB_IDs-896.txt` are guaranteed downloadable.

### Precomputed shards (AE2)

**Script**: `temporaries/precompute_ae2_long_shards.jl`
**Encoder**: AE2_ucond_800 (trained on ≤896 residues, same architecture as AE1, different weights)
**Filter**: 256-896 residues
**Output**: `~/shared_data/afdb_laproteina/precomputed_shards_ae2_long/train_shard_{01..20}.jld2`

AE2 encoder verified working on proteins up to L=896. GPU memory usage for pair features:
- L=400: ~164 MB
- L=700: ~502 MB
- L=896: ~822 MB (fits within 64 GB GPU limit)

## Autoencoder Variants

All three autoencoders share identical architecture (12-layer transformer, 768 token dim, 256 pair dim, 12 heads, 8D latent). Only training data differs:

| Variant | Checkpoint | Max training length | Used with |
|---------|-----------|-------------------:|-----------|
| AE1 | `AE1_ucond_512.safetensors` | 512 | LD1, LD2 (unconditional, ≤500 res) |
| AE2 | `AE2_ucond_800.safetensors` | 896 | LD3 (long proteins, 300-800 res) |
| AE3 | `AE3_motif.safetensors` | 256 | LD4-LD7 (motif scaffolding) |

Encoder constructor (same for all):
```julia
encoder = EncoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
load_encoder_weights_st!(encoder, "AE2_ucond_800.safetensors")
```

## Precomputed Shard Format

Each shard stores a `Vector{PrecomputedProteinNT}` under the JLD2 key `"proteins"`. Each element is a NamedTuple:

```julia
(ca_coords = [3, L],        # Zero-centered CA positions (nm)
 z_mean = [8, L],           # Encoder posterior mean
 z_log_scale = [8, L],      # Encoder posterior log std
 mask = [L])                # All 1.0 (filtered to valid only)
```

At training time, latents are sampled via reparameterization: `z = z_mean + randn() * exp(z_log_scale)`.

## File Locations Summary

```
~/shared_data/afdb_laproteina/
├── raw/                            # All downloaded CIF files (~300k+)
├── download_progress.json          # 512-list download tracking
├── download_long_progress.json     # 896-list download tracking
├── cluster_mapping.json            # FoldSeek cluster map for 512-list
├── precomputed_shards/             # AE1 shards (30-256 res, 231k proteins)
│   └── train_shard_{01..10}.jld2
└── precomputed_shards_ae2_long/    # AE2 shards (256-896 res)
    └── train_shard_{01..20}.jld2

~/BFlaproteina/data/
├── AFDB_IDs-512.txt                # 344k ID list (short/medium)
├── AFDB_IDs-896.txt                # 46.9M ID list (full, including long)
├── La-Proteina_AFDB_IDs.zip        # Original zip from NGC
└── afdb_clusters/
    ├── cluster_reps.tsv            # 2.3M cluster reps with metadata
    ├── cluster_membership.tsv      # 30M member→rep mappings
    └── filtered_cluster_reps.tsv   # 910k quality-filtered reps
```
