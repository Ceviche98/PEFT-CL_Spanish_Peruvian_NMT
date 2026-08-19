# Multilingual NMT with CaLoRA for Peruvian Indigenous Languages
## Thesis Code Repository — Oscar Moreno, QMUL MSc Data Science 2025-26

This repository contains all scripts required to reproduce the experiments described in the thesis:
*"Continual Learning for Low-Resource Neural Machine Translation: Benchmarking PEFT Methods and CaLoRA on Peruvian Indigenous Languages"*

The project fine-tunes **NLLB-200-1.3B** (Meta AI) on five Peruvian low-resource languages:
| Code | Language | Family |
|------|----------|--------|
| `quy` | Southern Quechua | Andean |
| `ayr` | Central Aymara | Andean |
| `shp` | Shipibo-Konibo | Amazonian |
| `agr` | Awajún | Amazonian |
| `cni` | Ashaninka | Amazonian |

---

## Repository Structure

```
repo/
├── README.md                            ← This file
├── requirements.txt                     ← Python dependencies
│
├── scripts/
│   ├── download_raw_data.py             # Download corpora from AmericasNLP
│   ├── preprocess_data.py               # Normalize & split train/dev pairs
│   ├── modify_tokenizer.py              # Add shp/agr/cni tokens to NLLB vocab
│   ├── download_nllb_model.py           # (Optional) Download base model locally
│   ├── evaluate.py                      # BLEU + chrF++ score calculator
│   ├── evaluate_lora.py                 # LoRA model inference + evaluation
│   │
│   ├── experiment_A/
│   │   ├── filter_and_count.py          # Length-filter to ≤256 tokens
│   │   ├── token_fertilizer.py          # Morphological fertility analysis
│   │   ├── train_benchmarks.py          # LoRA / DoRA / QLoRA / GaLORE training
│   │   ├── train_benchmarks_pissa.py    # PiSSA variant training
│   │   ├── train_fft_unfrozen_clean.py  # Full Fine-Tuning (FFT) training
│   │   ├── train_abridge_qkvo_v3.py     # A-Bridge joint QKVO LoRA (bridge to Exp B)
│   │   ├── evaluate_all_experiment_a_beam4.py   # Beam-4 final evaluation
│   │   ├── evaluate_flores200_forgetting_spa.py # FLORES-200 forgetting benchmark
│   │   └── check_final_embeddings.py    # Embedding integrity verification
│   │
│   └── experiment_B/
│       ├── nllb_prompt.py               # CaLoRALinear + CustomM2M100Attention modules
│       ├── calora_utils.py              # CaLoRA math: PaCA, CaGA, SVD helpers
│       ├── calora_x_utils.py            # CaLoRA-X gradient memory & soft projection
│       ├── train_lora_sequential.py     # Sequential LoRA baseline (continual learning)
│       ├── train_calora_sequential.py   # Sequential CaLoRA (main Experiment B)
│       ├── train_calora_x_sequential.py # Sequential CaLoRA-X (extended variant)
│       └── calculate_bwt_from_matrix.py # BWT / FM / AA from triangular eval matrix
│
├── run_setup.sh                         # Phase 0: data + tokenizer setup
├── run_experiment_A_train.sh            # Phase 1: all Exp A training
├── run_experiment_A_eval.sh             # Phase 2: Exp A evaluation
├── run_experiment_B_calora.sh           # Phase 3: LoRA + CaLoRA sequential
├── run_experiment_B_calorax.sh          # Phase 3b: CaLoRA-X sequential
│
├── data_in/                             # Populated by run_setup.sh
│   ├── raw/                             # Downloaded corpora
│   ├── train/                           # Preprocessed training pairs
│   └── dev/                             # Preprocessed dev pairs
│
├── data_out/                            # Evaluation outputs
│   ├── experiment_A_forgetting/
│   └── experiment_A_forgetting_spa/
│
├── flores200/                           # FLORES-200 benchmark (download separately)
│   ├── dev/
│   └── devtest/
│
└── models/                              # All model checkpoints (not included)
    ├── v0/tokenizer/                    # Modified tokenizer (from modify_tokenizer.py)
    ├── nllb-200-1.3B/                   # Base model (download from HuggingFace)
    ├── experiment_A/final/              # Experiment A trained checkpoints
    └── experiment_B/                    # Experiment B sequential adapters
```

---

## Prerequisites

### Hardware
- **Minimum:** NVIDIA GPU with 24 GB VRAM (RTX 3090 / A10G) for LoRA/PiSSA
- **Recommended:** 48 GB+ VRAM (A100 80GB / H100) for Full Fine-Tuning (FFT)
- The experiments were run on Vast.ai GPU instances and a local RTX 5090

### Software
- Python 3.10–3.12
- CUDA 12.1+
- Git (for corpus download)
- `bash` shell (Linux/macOS; on Windows use WSL2 or Git Bash)

### HuggingFace Token
A HuggingFace account token is needed to download the NLLB-200-1.3B model.
Get yours from: <https://huggingface.co/settings/tokens>

Replace `YOUR_HF_TOKEN_HERE` in any shell script you run with your actual token.

### FLORES-200 Benchmark (for Experiment A forgetting evaluation only)
Download from the official repository and place under `flores200/`:
```bash
git clone https://github.com/facebookresearch/flores.git /tmp/flores
cp -r /tmp/flores/data/flores200_dataset/dev      flores200/dev
cp -r /tmp/flores/data/flores200_dataset/devtest  flores200/devtest
```

---

## Step-by-Step Execution

### Phase 0 — Setup (data, tokenizer, filtering)

> Run once before any training.

```bash
# Set your HuggingFace token inside the script first
nano run_setup.sh   # replace YOUR_HF_TOKEN_HERE

bash run_setup.sh
```

This script performs the following steps in sequence:

| Step | Script | Output |
|------|--------|--------|
| 0A | `download_raw_data.py` | `data_in/raw/` — four corpus repos cloned |
| 0B | `preprocess_data.py` | `data_in/train/<lang>/`, `data_in/dev/<lang>/` |
| 1 | `modify_tokenizer.py` | `models/v0/tokenizer/` — extended vocabulary |
| 2 | `filter_and_count.py` | `data_in/train/*.filtered.*`, `data_in/summary_counts.txt` |
| 3 | `token_fertilizer.py` | `scripts/experiment_A/token_fertility.txt` |

**Optional — download the base model locally** (useful if the HuggingFace hub is slow):
```bash
HF_TOKEN=<your_token> python scripts/download_nllb_model.py
# Downloads to: models/nllb-200-1.3B/
```

---

### Phase 1 — Experiment A: PEFT Benchmark Training

> Trains LoRA (r=32, r=128, r=128 QKVO), PiSSA (r=128), FFT unfrozen, and A-Bridge QKVO v3.

```bash
bash run_experiment_A_train.sh
```

**Key CLI flags you can pass via environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE` | `8` | Per-device micro-batch size |
| `GRAD_ACCUM` | `2` | Gradient accumulation steps (eff. batch = 16) |
| `MAX_STEPS` | `611300` | Total optimizer steps (~20 epochs) |
| `PATIENCE` | `5` | Early-stopping patience (epochs without chrF++ improvement) |
| `SKIP_FFT` | `0` | Set to `1` to skip the FFT run (saves ~24 GB VRAM) |

**Example — run only with smaller batch on a 24 GB GPU:**
```bash
BATCH_SIZE=4 GRAD_ACCUM=4 bash run_experiment_A_train.sh
```

**Resuming from a checkpoint:**
Each training script automatically calls `get_last_checkpoint()` on startup and resumes
from the latest rolling `hf/checkpoint-NNNNN/` folder. Simply re-run the same command
after a crash — no manual intervention needed.

**Where are checkpoints saved?**
```
models/experiment_A/
    lora_r32/lr1e-4/
        best_chrf_checkpoint/    ← Best model by avg chrF++ (used for evaluation)
        hf/checkpoint-NNNNN/     ← Latest rolling checkpoint (for crash recovery)
        training_log.csv
    lora_r128/lr1e-4/ ...
    pissa_r128/lr1e-4/ ...
    fft_unfrozen/adafactor_lr1e-5_wd1e-3_t0.7_b12x10_5ep/ ...
    experiment_A_bridge_qkvo_v3/ ...
```

---

### Phase 2 — Experiment A: Evaluation

> Requires Phase 1 to be complete. FLORES-200 also requires the flores200/ directory.

```bash
bash run_experiment_A_eval.sh
```

This runs three evaluation stages:

| Stage | Script | What it measures |
|-------|--------|-----------------|
| 1 | `evaluate_all_experiment_a_beam4.py` | chrF++ + BLEU on 5-language AmericasNLP dev sets (beam=4) |
| 2 | `evaluate_flores200_forgetting_spa.py` | Catastrophic forgetting on ~200 FLORES languages (Spanish→X) |
| 3 | `check_final_embeddings.py` | Verifies embedding drift for new language tokens |

**Reducing batch size if OOM:**
```bash
BATCH_SIZE=8 bash run_experiment_A_eval.sh
```

**Running a single stage manually:**
```bash
# Beam-4 only, custom models directory
python scripts/experiment_A/evaluate_all_experiment_a_beam4.py \
    --models_dir models/ --data_dir data_in/ \
    --output_dir models/experiment_A/final/beam4_eval \
    --batch_size 16 --num_beams 4 \
    --models lora_r32 lora_r128 pissa_r128

# FLORES-200 forgetting only
python scripts/experiment_A/evaluate_flores200_forgetting_spa.py \
    --batch_size 64 --num_beams 4
```

---

### Phase 3 — Experiment B: Sequential Continual Learning (CaLoRA)

> Trains the plain LoRA baseline and the CaLoRA system sequentially across all 5 languages.

```bash
bash run_experiment_B_calora.sh
```

By default, this runs **both order1 and order2** for both LoRA and CaLoRA.

**Language orderings:**
- `order1` — High-resource → Low-resource: `quy → ayr → shp → agr → cni`
- `order2` — Low-resource → High-resource: `cni → agr → shp → ayr → quy`

**Key environment overrides:**

| Variable | Default | Description |
|----------|---------|-------------|
| `ORDER` | `order1 order2` | Which ordering(s) to run |
| `BATCH_SIZE` | `10` | Per-device batch size |
| `GRAD_ACCUM` | `12` | Accumulation steps (eff. batch = 120) |
| `LR` | `1e-4` | Learning rate |
| `LORA_R` | `128` | LoRA rank |
| `PACA_OFF` | `0` | Set to `1` to disable CaGA/PaCA correction (raw LoRA baseline) |
| `START_TASK` | `1` | Resume from this task index (1–5) |
| `END_TASK` | `5` | Stop after this task index |
| `RUN_TAG` | (auto) | Appended to output folder name to prevent overwriting |
| `SEED` | `42` | Random seed |

**Resuming from a checkpoint after a crash:**
```bash
# Example: resume CaLoRA from task 3 (shp) onwards for order1
START_TASK=3 ORDER=order1 bash run_experiment_B_calora.sh
```
The script loads the task-2 adapter (`ayr`) as its starting point and continues from task 3.

**Running a single ordering only:**
```bash
ORDER=order1 bash run_experiment_B_calora.sh
```

**After training — compute BWT / FM / AA manually on any run:**
```bash
python scripts/experiment_B/calculate_bwt_from_matrix.py \
    models/experiment_B/<run_tag>/triangular_eval_matrix.json \
    --order order1
```
This prints Average Accuracy (AA), Forgetting Measure (FM), and Backward Transfer (BWT)
for each language and the aggregate, matching the metrics reported in the thesis.

**Output structure:**
```
models/experiment_B/<run_tag>/
    task_1_quy/best_checkpoint/        ← Task 1 adapter
    task_2_ayr/best_checkpoint/        ← Task 2 adapter (loaded when resuming task 3)
    ...
    task_5_cni/best_checkpoint/
    triangular_eval_matrix.json        ← Full evaluation matrix (all tasks × all langs)
    training_log.csv
```

---

### Phase 3b — Experiment B: CaLoRA-X (Extended Variant)

> CaLoRA-X replaces single boundary-gradient CaGA with compact representative gradient
> subspaces and soft conflict-gated projection. Evaluated primarily on order2.

```bash
bash run_experiment_B_calorax.sh
```

**Key additional parameters:**

| Variable | Default | Description |
|----------|---------|-------------|
| `ORDER` | `order2` | Language ordering |
| `X_MEMORY_SAMPLES` | `4` | Number of gradient snapshots per task |
| `X_MEMORY_START_FRACTION` | `0.4` | Fraction of task steps before first snapshot |
| `X_MEMORY_RANK` | `8` | SVD rank for compressed gradient basis |
| `X_ATTENUATION` | `0.5` | Soft projection attenuation factor λ |
| `X_MIN_SCALE` | `0.1` | Minimum gradient scale (floor) |

**Resuming CaLoRA-X from a checkpoint:**
```bash
START_TASK=3 RUN_TAG=calorax_order2_v1 bash run_experiment_B_calorax.sh
```

---

## Individual Script Reference

### Data Pipeline

| Script | Purpose | Example |
|--------|---------|---------|
| `scripts/download_raw_data.py` | Clone AmericasNLP 2021/2025 corpora and NLLB-Seed data into `data_in/raw/` | `python scripts/download_raw_data.py` |
| `scripts/preprocess_data.py` | NFC normalization, MosesPunctNorm, blank-pair removal, train/dev split | `python scripts/preprocess_data.py --languages specific --lang_list quy,ayr,shp,agr,cni` |
| `scripts/modify_tokenizer.py` | Add `shp_Latn`, `agr_Latn`, `cni_Latn` tokens to NLLB-200 vocabulary | `python scripts/modify_tokenizer.py` |
| `scripts/download_nllb_model.py` | Download `facebook/nllb-200-1.3B` to `models/nllb-200-1.3B/` | `python scripts/download_nllb_model.py` |
| `scripts/experiment_A/filter_and_count.py` | Remove pairs exceeding 256 tokens; save `summary_counts.txt` | `python scripts/experiment_A/filter_and_count.py` |
| `scripts/experiment_A/token_fertilizer.py` | Compute Fertility = Subwords / Words per language | `python scripts/experiment_A/token_fertilizer.py` |

### Evaluation Utilities

| Script | Purpose | Example |
|--------|---------|---------|
| `scripts/evaluate.py` | Compute BLEU + chrF++ from hypothesis/reference files | `python scripts/evaluate.py --sys output.txt --ref reference.txt` |
| `scripts/evaluate_lora.py` | Load a LoRA adapter, run inference, report scores | `python scripts/evaluate_lora.py --model_path models/... --source_file data_in/dev/agr/dev.es --reference_file data_in/dev/agr/dev.agr --src_lang spa_Latn --tgt_lang agr_Latn` |

### Experiment A

| Script | Purpose | Key CLI arguments |
|--------|---------|------------------|
| `train_benchmarks.py` | Joint multilingual LoRA/DoRA/QLoRA/GaLORE training | `--method lora --lora_r 128 --lr 1e-4 --max_steps 611300 --patience 5` |
| `train_benchmarks_pissa.py` | PiSSA (Principal Singular Values) training | `--lora_r 128 --lr 1e-4 --max_steps 611300` |
| `train_fft_unfrozen_clean.py` | Full Fine-Tuning with unfrozen embeddings | `--lr 1e-5 --optimizer adafactor --max_steps 20375` *(requires `transformers==4.40.2`)* |
| `train_abridge_qkvo_v3.py` | A-Bridge: standalone QKVO LoRA joint multilingual control | `--lora_r 128 --lr 1e-4 --max_steps 611280 --patience 5` |
| `evaluate_all_experiment_a_beam4.py` | Beam-4 chrF++ + BLEU on all Exp A models | `--models lora_r32 lora_r128 lora_r128_qkvo pissa_r128 --num_beams 4` |
| `evaluate_flores200_forgetting_spa.py` | Forgetting benchmark across ~200 FLORES languages | `--batch_size 64 --num_beams 4` |
| `check_final_embeddings.py` | Verify embedding integrity of trained adapters vs base model | `python scripts/experiment_A/check_final_embeddings.py` |

### Experiment B

| Script | Purpose | Key CLI arguments |
|--------|---------|------------------|
| `nllb_prompt.py` | Custom `CaLoRALinear` and `CustomM2M100Attention` modules (imported, not run directly) | — |
| `calora_utils.py` | CaLoRA math helpers: `compute_Et`, `lora_project_svd`, cosine similarity (imported, not run directly) | — |
| `calora_x_utils.py` | CaLoRA-X gradient memory and soft projection utilities (imported, not run directly) | — |
| `train_lora_sequential.py` | Sequential LoRA continual learning baseline | `--order order1 --lora_r 128 --lr 1e-4 --batch_size 10 --grad_accum 12` |
| `train_calora_sequential.py` | Sequential CaLoRA with PaCA + CaGA | `--order order1 --lora_r 128 --lr 1e-4 --batch_size 10 --grad_accum 12` |
| `train_calora_x_sequential.py` | Sequential CaLoRA-X with representative memory | `--order order2 --lora_r 128 --lr 1e-4 --batch_size 6 --grad_accum 20 --x_memory_samples 4` |
| `calculate_bwt_from_matrix.py` | Read triangular evaluation matrix → print AA, FM, BWT | `python calculate_bwt_from_matrix.py path/to/triangular_eval_matrix.json --order order1` |

---

## Key Architecture Notes

### M2M100 Transformer Compatibility Patch
NLLB-200 uses the M2M100 architecture. Transformers ≥4.50 has a regression where
`M2M100Model.forward` passes both `decoder_input_ids` AND `decoder_inputs_embeds`
positionally to the decoder, causing a collision. All training scripts contain an inline
monkey-patch at the top that fixes this by converting `decoder_input_ids` to embeddings
and nulling the ids before dispatch.

### Embedding Masking
Three language tokens (`shp_Latn`, `agr_Latn`, `cni_Latn`) were added to the NLLB
vocabulary. During training, a gradient hook freezes all 256 000 original token rows
and only allows gradient flow for the 5 active language tag rows. This prevents
AdamW weight decay from eroding the base vocabulary.

### Checkpoint Structure for LoRA Models
PEFT `save_pretrained` only saves the adapter weights. Our custom callback additionally
saves `embeddings.pt` (the modified embedding matrix). Evaluation scripts load
`embeddings.pt` into the base model BEFORE wrapping with `PeftModel.from_pretrained`.

### FFT Requires `transformers==4.40.2`
`train_fft_unfrozen_clean.py` uses the native `Seq2SeqTrainer` from Transformers 4.40.2
which does NOT have the M2M100 regression. The training script in `run_experiment_A_train.sh`
automatically installs this pinned version for the FFT run and then restores the full
requirements. If running the FFT script manually, install the pinned version first:
```bash
pip install "transformers==4.40.2"
python scripts/experiment_A/train_fft_unfrozen_clean.py ...
```

---

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{moreno2026calora,
  author  = {Oscar Moreno},
  title   = {Continual Learning for Low-Resource Neural Machine Translation:
             Benchmarking PEFT Methods and CaLoRA on Peruvian Indigenous Languages},
  school  = {Queen Mary University of London},
  year    = {2026},
  month   = {August}
}
```

---

## Data Sources

| Dataset | URL |
|---------|-----|
| AmericasNLP 2025 | <https://github.com/AmericasNLP/americasnlp2025> |
| AmericasNLP 2021 Shared Task | <https://github.com/Helsinki-NLP/americasnlp2021-st> |
| REPUcs AmericasNLP 2021 | <https://github.com/Ceviche98/REPUcs-AmericasNLP2021> |
| NLLB Seed Data | <https://tinyurl.com/NLLBSeed> |
| FLORES-200 | <https://github.com/facebookresearch/flores> |
| NLLB-200-1.3B base model | <https://huggingface.co/facebook/nllb-200-1.3B> |
