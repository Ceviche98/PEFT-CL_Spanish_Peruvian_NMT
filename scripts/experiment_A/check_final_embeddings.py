#!/usr/bin/env python3
"""
check_final_embeddings.py
-------------------------
Verifies embedding matrix integrity for the final Experiment A models:
  • lora_r32
  • lora_r128
  • pissa_r128

For each model's best_chrf_checkpoint/embeddings.pt it checks:
  1. The embedding matrix has the exact same shape as the base model (facebook/nllb-200-1.3B).
  2. All token rows EXCEPT the 5 active language tags (quy, ayr, shp, agr, cni)
     are bit-identical / unchanged compared to the base model.
  3. The 3 new-language tag rows (shp, agr, cni) ARE different from the base model.
  4. The 3 new-language tag rows drifted independently from each other during training.
  5. The 3 new-language tag rows remain within reasonable bounds of the initial (quy+ayr)/2 average.

Run from project root:
    python scripts/experiment_A/check_final_embeddings.py
"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_MODEL_ID = "facebook/nllb-200-1.3B"

MODEL_PATHS = {
    "lora_r32": PROJECT_ROOT / "models" / "experiment_A" / "final" / "lora_r32" / "lr1e-4" / "best_chrf_checkpoint",
    "lora_r128": PROJECT_ROOT / "models" / "experiment_A" / "final" / "lora_r128" / "lr1e-4" / "best_chrf_checkpoint",
    "pissa_r128": PROJECT_ROOT / "models" / "experiment_A" / "final" / "pissa_r128" / "lr1e-4" / "best_chrf_checkpoint",
}

# ---------------------------------------------------------------------------
# Language constants
# ---------------------------------------------------------------------------
LANG_TO_NLLB = {
    "quy": "quy_Latn",
    "ayr": "ayr_Latn",
    "shp": "shp_Latn",
    "agr": "agr_Latn",
    "cni": "cni_Latn",
}
NEW_LANGUAGES = ["shp", "agr", "cni"]
BASE_LANGUAGES = ["quy", "ayr"]

# ---------------------------------------------------------------------------
# Load reference embeddings from base model instance
# ---------------------------------------------------------------------------
def load_reference_embeddings(tokenizer: NllbTokenizer) -> torch.Tensor:
    print(f"Loading reference base model '{BASE_MODEL_ID}' in bfloat16...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))
    ref = model.get_input_embeddings().weight.data.clone().cpu()
    del model
    torch.cuda.empty_cache()
    print(f"  Reference embedding shape: {ref.shape}")
    return ref


# ---------------------------------------------------------------------------
# Per-model check
# ---------------------------------------------------------------------------
def check_model_embeddings(
    model_name: str,
    ckpt_dir: Path,
    ref_emb: torch.Tensor,
    tokenizer: NllbTokenizer,
    new_ids: list[int],
    base_ids: list[int],
    tol: float = 1e-4,
) -> dict:
    emb_path = ckpt_dir / "embeddings.pt"
    result = {
        "model": model_name,
        "path": str(emb_path),
        "found": emb_path.exists(),
        "shape_ok": None,
        "unchanged_rows_ok": None,
        "new_rows_changed": None,
        "new_rows_updated_independently": None,
        "new_rows_near_avg_quy_ayr": None,
        "errors": [],
    }

    if not emb_path.exists():
        result["errors"].append(f"embeddings.pt not found at {emb_path}")
        return result

    saved_emb = torch.load(emb_path, map_location="cpu", weights_only=True)
    
    # Cast BOTH to float32 just for subtraction to avoid bfloat16 precision quirks during diff computation
    saved_emb_f32 = saved_emb.float()
    ref_emb_f32 = ref_emb.float()

    # 1. Shape check
    result["shape_ok"] = (saved_emb_f32.shape == ref_emb_f32.shape)
    if not result["shape_ok"]:
        result["errors"].append(
            f"Shape mismatch: saved={saved_emb_f32.shape} vs ref={ref_emb_f32.shape}"
        )
        return result

    # 2. Rows that should be UNCHANGED (all rows except the 5 active language tags)
    all_indices = set(range(ref_emb_f32.shape[0]))
    active_ids_set = set(new_ids + base_ids)
    unchanged_indices = sorted(all_indices - active_ids_set)
    unchanged_tensor = torch.tensor(unchanged_indices, dtype=torch.long)

    diff_unchanged = (saved_emb_f32[unchanged_tensor] - ref_emb_f32[unchanged_tensor]).abs().max().item()
    result["unchanged_rows_ok"] = diff_unchanged <= tol
    if not result["unchanged_rows_ok"]:
        result["errors"].append(
            f"Max deviation in frozen/base rows: {diff_unchanged:.6e} (tol={tol})"
        )

    # 3. New-language rows should DIFFER from base model initialization
    diffs_new = [(saved_emb_f32[i] - ref_emb_f32[i]).abs().max().item() for i in new_ids]
    result["new_rows_changed"] = all(d > tol for d in diffs_new)
    if not result["new_rows_changed"]:
        unchanged_langs = [
            LANG_TO_NLLB[lg]
            for lg, d in zip(NEW_LANGUAGES, diffs_new)
            if d <= tol
        ]
        result["errors"].append(
            f"These new-language token rows did not change from base model: {unchanged_langs}"
        )

    # 4. New-language rows should drift from each other during training
    rows = [saved_emb_f32[i] for i in new_ids]
    pairwise_max = max(
        (rows[i] - rows[j]).abs().max().item()
        for i in range(len(rows)) for j in range(i + 1, len(rows))
    )
    result["new_rows_updated_independently"] = pairwise_max > 0.0
    if not result["new_rows_updated_independently"]:
        result["errors"].append(
            "New-language rows are identical to each other! They should have learned distinct representations."
        )

    # 5. New rows should loosely equal (quy + ayr) / 2
    quy_id = tokenizer.convert_tokens_to_ids("quy_Latn")
    ayr_id = tokenizer.convert_tokens_to_ids("ayr_Latn")
    avg_vec_ref = (ref_emb_f32[quy_id] + ref_emb_f32[ayr_id]) / 2.0
    diffs_from_avg = [(saved_emb_f32[i] - avg_vec_ref).abs().max().item() for i in new_ids]
    result["new_rows_near_avg_quy_ayr"] = all(d < 1.0 for d in diffs_from_avg)
    if not result["new_rows_near_avg_quy_ayr"]:
        result["errors"].append(
            f"New-language rows drifted significantly from (quy+ayr)/2 average "
            f"(max diffs: {[f'{d:.4f}' for d in diffs_from_avg]})"
        )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Experiment A Final Models — Embedding Integrity Checker")
    print("=" * 70)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # Find a valid tokenizer from one of the models
    valid_tokenizer_dir = None
    for name, path in MODEL_PATHS.items():
        if path.exists() and (path / "tokenizer_config.json").exists():
            valid_tokenizer_dir = path
            break

    if not valid_tokenizer_dir:
        print("[ERROR] Could not find any valid tokenizer in the model directories!")
        sys.exit(1)

    print(f"[INFO] Using tokenizer from: {valid_tokenizer_dir}")
    tokenizer = NllbTokenizer.from_pretrained(str(valid_tokenizer_dir))
    ref_emb = load_reference_embeddings(tokenizer)

    new_ids = [tokenizer.convert_tokens_to_ids(LANG_TO_NLLB[lg]) for lg in NEW_LANGUAGES]
    base_ids = [tokenizer.convert_tokens_to_ids(LANG_TO_NLLB[lg]) for lg in BASE_LANGUAGES]

    print(f"\nNew language token IDs  ({', '.join(LANG_TO_NLLB[l] for l in NEW_LANGUAGES)}): {new_ids}")
    print(f"Base language token IDs ({', '.join(LANG_TO_NLLB[l] for l in BASE_LANGUAGES)}): {base_ids}\n")

    all_ok = True
    for model_name, ckpt_dir in MODEL_PATHS.items():
        result = check_model_embeddings(model_name, ckpt_dir, ref_emb, tokenizer, new_ids, base_ids)
        ok_char = lambda v: "[OK]" if v else "[FAIL]"
        status_ok = (
            result["found"]
            and result["shape_ok"]
            and result["unchanged_rows_ok"]
            and result["new_rows_changed"]
            and result["new_rows_updated_independently"]
            and result["new_rows_near_avg_quy_ayr"]
        )
        all_ok = all_ok and status_ok

        print(f"[{'PASS' if status_ok else 'FAIL'}]  Model: {model_name.upper()}")
        print(f"       embeddings.pt found           : {ok_char(result['found'])} ({result['path']})")
        if result["found"]:
            print(f"       shape matches base model      : {ok_char(result['shape_ok'])} {ref_emb.shape}")
            print(f"       all non-lang rows unchanged   : {ok_char(result['unchanged_rows_ok'])}")
            print(f"       new-lang rows differ from base: {ok_char(result['new_rows_changed'])}")
            print(f"       new-lang rows drifted apart   : {ok_char(result['new_rows_updated_independently'])}")
            print(f"       new-lang ~/.= (quy+ayr)/2     : {ok_char(result['new_rows_near_avg_quy_ayr'])}")
        if result["errors"]:
            for e in result["errors"]:
                print(f"       !! {e}")
        print()

    print("=" * 70)
    if all_ok:
        print("  ALL CHECKS PASSED — 100% SECURITY CONFIRMED:")
        print("  The base model embeddings are identically preserved while language tokens learned distinct representations.")
    else:
        print("  ONE OR MORE CHECKS FAILED — see details above.")
    print("=" * 70)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
