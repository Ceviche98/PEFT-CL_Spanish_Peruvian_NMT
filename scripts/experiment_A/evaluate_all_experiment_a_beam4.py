#!/usr/bin/env python3
"""Final beam-4 evaluation for Experiment-A joint multilingual models.

Expected Vast.ai layout (all paths can be overridden from the CLI):

  <project>/
    data_in/dev/{quy,ayr,shp,agr,cni}/dev.{es,<language>}
    models/
      nllb-200-1.3B/          # local frozen NLLB-1.3B backbone
      v0/tokenizer/           # tokenizer with the five target tags
      lora_r32/               # adapter checkpoint or a folder containing it
      lora_r128/
      lora_r128_qkvo/
      pissa_r128/

Each model folder may either be the extracted ``best_chrf_checkpoint`` itself
or contain one.  LoRA/PiSSA adapters are recognised by adapter_config.json;
standard full-fine-tuned models are recognised by config.json plus model
weights.  Results use corpus chrF++ and beam width 4, matching Experiment B.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import sacrebleu
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Decoder, M2M100Model


LANG_TO_NLLB = {
    "quy": "quy_Latn",
    "ayr": "ayr_Latn",
    "shp": "shp_Latn",
    "agr": "agr_Latn",
    "cni": "cni_Latn",
}
ALL_LANGS = list(LANG_TO_NLLB)
DEFAULT_MODELS = ("lora_r32", "lora_r128", "lora_r128_qkvo", "pissa_r128")
# These checkpoints were trained by the original benchmark/FFT scripts, which
# applied an extra sqrt(d_model) on top of the already scaled M2M100 embedding
# in teacher forcing and generate().  They must be evaluated with that same
# legacy decoder representation.  New ``*_clean`` and Q/K/V/O A-Bridge runs
# intentionally do not appear here.
LEGACY_DOUBLE_SCALE_MODELS = frozenset((
    "lora_r32", "lora_r128", "pissa_r128", "dora_r128", "qlora_r128",
    "fft", "fft_frozen", "fft_unfrozen", "galore",
))


# Correct the Transformers M2M100 dual decoder-input regression while keeping
# NLLB's scaled embedding exactly once.  This is the same generation fix used
# by the final Experiment-B scripts: M2M100ScaledWordEmbedding already applies
# embed_scale, so multiplying by sqrt(d_model) a second time is incorrect.
_ORIGINAL_MODEL_FORWARD = M2M100Model.forward
_ORIGINAL_DECODER_FORWARD = M2M100Decoder.forward


def _safe_m2m_model_forward(
    self, input_ids=None, attention_mask=None,
    decoder_input_ids=None, decoder_attention_mask=None,
    head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None,
    encoder_outputs=None, past_key_values=None,
    inputs_embeds=None, decoder_inputs_embeds=None,
    use_cache=None, output_attentions=None, output_hidden_states=None,
    return_dict=None, **kwargs,
):
    if decoder_input_ids is not None:
        if decoder_inputs_embeds is None:
            decoder_inputs_embeds = self.shared(decoder_input_ids)
        decoder_input_ids = None
    return _ORIGINAL_MODEL_FORWARD(
        self,
        input_ids=input_ids, attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        head_mask=head_mask, decoder_head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        encoder_outputs=encoder_outputs, past_key_values=past_key_values,
        inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        **kwargs,
    )


def _safe_m2m_decoder_forward(
    self, input_ids=None, attention_mask=None,
    encoder_hidden_states=None, encoder_attention_mask=None,
    head_mask=None, cross_attn_head_mask=None,
    past_key_values=None, inputs_embeds=None,
    use_cache=None, output_attentions=None,
    output_hidden_states=None, return_dict=None,
    **kwargs,
):
    if input_ids is not None:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        input_ids = None
    return _ORIGINAL_DECODER_FORWARD(
        self,
        input_ids=input_ids, attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        head_mask=head_mask, cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        **kwargs,
    )


def _legacy_m2m_model_forward(
    self, input_ids=None, attention_mask=None,
    decoder_input_ids=None, decoder_attention_mask=None,
    head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None,
    encoder_outputs=None, past_key_values=None,
    inputs_embeds=None, decoder_inputs_embeds=None,
    use_cache=None, output_attentions=None, output_hidden_states=None,
    return_dict=None, **kwargs,
):
    """Exact generation-time patch used by legacy train_benchmarks.py."""
    if decoder_input_ids is not None:
        if decoder_inputs_embeds is None:
            scale = math.sqrt(self.config.d_model) if getattr(self.config, "scale_embedding", False) else 1.0
            decoder_inputs_embeds = self.shared(decoder_input_ids) * scale
        decoder_input_ids = None
    return _ORIGINAL_MODEL_FORWARD(
        self,
        input_ids=input_ids, attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        head_mask=head_mask, decoder_head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        encoder_outputs=encoder_outputs, past_key_values=past_key_values,
        inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        **kwargs,
    )


def _legacy_m2m_decoder_forward(
    self, input_ids=None, attention_mask=None,
    encoder_hidden_states=None, encoder_attention_mask=None,
    head_mask=None, cross_attn_head_mask=None,
    past_key_values=None, inputs_embeds=None,
    use_cache=None, output_attentions=None,
    output_hidden_states=None, return_dict=None,
    **kwargs,
):
    """Exact decoder safety-net patch used by legacy train_benchmarks.py."""
    if input_ids is not None:
        if inputs_embeds is None:
            scale = math.sqrt(self.config.d_model) if getattr(self.config, "scale_embedding", False) else 1.0
            inputs_embeds = self.embed_tokens(input_ids) * scale
        input_ids = None
    return _ORIGINAL_DECODER_FORWARD(
        self,
        input_ids=input_ids, attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        head_mask=head_mask, cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        **kwargs,
    )


def set_decoder_scale_mode(mode: str) -> None:
    """Select legacy or corrected decoder embeddings for one checkpoint."""
    if mode == "legacy_double":
        M2M100Model.forward = _legacy_m2m_model_forward
        M2M100Decoder.forward = _legacy_m2m_decoder_forward
    elif mode == "correct_single":
        M2M100Model.forward = _safe_m2m_model_forward
        M2M100Decoder.forward = _safe_m2m_decoder_forward
    else:
        raise ValueError(f"Unknown decoder scale mode: {mode}")


def choose_decoder_scale_mode(model_name: str, forced_mode: str, checkpoint: Path | None = None) -> str:
    if forced_mode != "auto":
        return forced_mode
    # A user may pass either ``lora_r128`` or its nested
    # ``lora_r128/best_chrf_checkpoint`` directory.  Inspect every parent of
    # the resolved checkpoint so both forms receive the right compatibility
    # path.  Exact-name matching intentionally keeps lora_r128_qkvo separate.
    names = {model_name.lower()}
    if checkpoint is not None:
        names.update(parent.name.lower() for parent in (checkpoint, *checkpoint.parents))
    return "legacy_double" if names & LEGACY_DOUBLE_SCALE_MODELS else "correct_single"


# Correct-single is the safe default for any newly trained model.  The loop
# below explicitly switches legacy Experiment-A checkpoints before evaluating.
set_decoder_scale_mode("correct_single")


def _contains_full_model(path: Path) -> bool:
    return (path / "config.json").exists() and any(
        (path / filename).exists()
        for filename in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
    )


def resolve_checkpoint(model_root: Path, name_or_path: str) -> Path:
    """Resolve a direct checkpoint, its best checkpoint child, or one adapter."""
    candidate = Path(name_or_path)
    if not candidate.is_absolute():
        candidate = model_root / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Model path does not exist: {candidate}")

    if (candidate / "adapter_config.json").exists() or _contains_full_model(candidate):
        return candidate

    best = candidate / "best_chrf_checkpoint"
    if (best / "adapter_config.json").exists() or _contains_full_model(best):
        return best

    preferred = sorted(candidate.rglob("best_chrf_checkpoint"))
    for checkpoint in preferred:
        if (checkpoint / "adapter_config.json").exists() or _contains_full_model(checkpoint):
            return checkpoint

    adapters = sorted(candidate.rglob("adapter_config.json"))
    if len(adapters) == 1:
        return adapters[0].parent
    if len(adapters) > 1:
        choices = "\n  ".join(str(item.parent) for item in adapters)
        raise RuntimeError(
            f"More than one adapter checkpoint was found below {candidate}. "
            f"Pass the exact checkpoint directory instead:\n  {choices}"
        )
    raise FileNotFoundError(
        f"No extracted adapter_config.json or full model weights found under {candidate}. "
        "Upload/extract the best_chrf_checkpoint directory, not only its .zip file."
    )


def load_dev_data(data_dir: Path) -> dict[str, tuple[list[str], list[str]]]:
    data: dict[str, tuple[list[str], list[str]]] = {}
    for lang in ALL_LANGS:
        source_path = data_dir / "dev" / lang / "dev.es"
        target_path = data_dir / "dev" / lang / f"dev.{lang}"
        if not source_path.exists() or not target_path.exists():
            raise FileNotFoundError(f"Missing dev pair for {lang.upper()}: {source_path}, {target_path}")
        source = [line.strip() for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        target = [line.strip() for line in target_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(source) != len(target):
            raise ValueError(f"Mismatched dev-set size for {lang.upper()}: {len(source)} source vs {len(target)} target")
        data[lang] = (source, target)
    return data


def torch_load_weights(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, found {type(tensor).__name__}")
    return tensor


def restore_embeddings(model, embeddings_path: Path) -> None:
    if not embeddings_path.exists():
        print("  [warning] embeddings.pt not found; using the backbone embedding rows.")
        return
    saved = torch_load_weights(embeddings_path)
    embedding = model.get_input_embeddings()
    if saved.shape != embedding.weight.shape:
        raise ValueError(
            f"Embedding shape mismatch in {embeddings_path}: saved {tuple(saved.shape)}, "
            f"model {tuple(embedding.weight.shape)}. Check that models/v0/tokenizer was uploaded."
        )
    with torch.no_grad():
        embedding.weight.copy_(saved.to(device=embedding.weight.device, dtype=embedding.weight.dtype))
    print("  Restored embeddings.pt.")


def load_model(checkpoint: Path, base_model_dir: Path, tokenizer: NllbTokenizer, device: torch.device):
    if not base_model_dir.exists():
        raise FileNotFoundError(f"Local NLLB backbone not found: {base_model_dir}")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    is_adapter = (checkpoint / "adapter_config.json").exists()

    if is_adapter:
        print(f"  Loading NLLB backbone: {base_model_dir}")
        model = AutoModelForSeq2SeqLM.from_pretrained(str(base_model_dir), torch_dtype=dtype)
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Applying PEFT adapter: {checkpoint}")
        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
        model.to(device)
        restore_embeddings(model, checkpoint / "embeddings.pt")
    else:
        print(f"  Loading full fine-tuned model: {checkpoint}")
        model = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint), torch_dtype=dtype).to(device)
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        # A fully saved model already contains its embedding table.  Do not
        # overwrite it with a sidecar tensor unless the model lacks the new rows.

    model.eval()
    return model


def generate(
    model,
    tokenizer: NllbTokenizer,
    sources: list[str],
    target_token: str,
    device: torch.device,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
) -> list[str]:
    target_id = tokenizer.convert_tokens_to_ids(target_token)
    if target_id is None or target_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer does not contain target language token {target_token}.")
    outputs_text: list[str] = []
    for start in tqdm(range(0, len(sources), batch_size), desc=f"  {target_token}", leave=False):
        batch = sources[start : start + batch_size]
        tokenizer.src_lang = "spa_Latn"
        # Dev sources are evaluated as supplied. The <=256 filter belongs only
        # to the training corpus, so source-side truncation is disabled here.
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        outputs_text.extend(text.strip() for text in tokenizer.batch_decode(generated, skip_special_tokens=True))
    if len(outputs_text) != len(sources):
        raise RuntimeError(f"Generation count mismatch: {len(outputs_text)} predictions for {len(sources)} sources")
    return outputs_text


def print_table(rows: list[dict[str, Any]]) -> str:
    columns = ["model", "avg_chrf", *[f"{lang}_chrf" for lang in ALL_LANGS]]
    headers = ["Model", "Avg chrF++", "QUY", "AYR", "SHP", "AGR", "CNI"]
    values = [[str(row["model"]), *[f"{float(row[column]):.2f}" for column in columns[1:]]] for row in rows]
    widths = [max(len(header), *(len(line[index]) for line in values)) for index, header in enumerate(headers)]
    separator = "-+-".join("-" * width for width in widths)
    lines = [" | ".join(header.ljust(width) for header, width in zip(headers, widths)), separator]
    lines.extend(" | ".join(value.ljust(width) for value, width in zip(line, widths)) for line in values)
    return "\n".join(lines)


def save_predictions(output_dir: Path, model_name: str, lang: str, sources: list[str], references: list[str], predictions: list[str]) -> None:
    path = output_dir / "predictions" / model_name / f"{lang}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["source_es", f"reference_{lang}", "prediction"])
        writer.writerows(zip(sources, references, predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all Experiment-A models with beam-4 corpus chrF++.")
    parser.add_argument("--models_dir", type=Path, default=None, help="Directory containing v0/, nllb-200-1.3B/, and model folders.")
    parser.add_argument("--data_dir", type=Path, default=None, help="Directory containing dev/<language>/ files.")
    parser.add_argument("--base_model_dir", type=Path, default=None, help="Local NLLB-200-1.3B backbone directory.")
    parser.add_argument("--tokenizer_dir", type=Path, default=None, help="Tokenizer directory; normally models/v0/tokenizer.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory for metrics and predictions.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Model folder names or absolute checkpoint paths.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Generated target-token cap; matches executed Experiment A/B code.")
    parser.add_argument(
        "--decoder_scale_mode",
        choices=("auto", "legacy_double", "correct_single"),
        default="auto",
        help=(
            "Decoder embedding compatibility mode. 'auto' uses legacy_double "
            "for original benchmark/FFT checkpoints and correct_single for "
            "newer checkpoints such as lora_r128_qkvo."
        ),
    )
    parser.add_argument("--continue_on_error", action="store_true", help="Write successful models and continue if one checkpoint fails.")
    args = parser.parse_args()
    if min(args.batch_size, args.num_beams, args.max_new_tokens) < 1:
        parser.error("batch_size, num_beams and max_new_tokens must be positive.")

    project_root = Path(__file__).resolve().parents[2]
    models_dir = (args.models_dir or project_root / "models").resolve()
    data_dir = (args.data_dir or project_root / "data_in").resolve()
    base_model_dir = (args.base_model_dir or models_dir / "nllb-200-1.3B").resolve()
    tokenizer_dir = (args.tokenizer_dir or models_dir / "v0" / "tokenizer").resolve()
    output_dir = (args.output_dir or models_dir / "experiment_A_beam4_evaluation").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. NLLB-1.3B beam-4 evaluation on CPU can take many hours per model.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("EXPERIMENT A: FINAL BEAM-4 EVALUATION")
    print(
        f"Device: {device} | batch_size={args.batch_size} | num_beams={args.num_beams} | "
        f"source_truncation=False | max_new_tokens={args.max_new_tokens} | "
        f"decoder_scale_mode={args.decoder_scale_mode}"
    )
    print(f"Models directory: {models_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 76)

    tokenizer = NllbTokenizer.from_pretrained(str(tokenizer_dir))
    dev_data = load_dev_data(data_dir)
    print("Dev-set sentences: " + ", ".join(f"{lang.upper()}={len(dev_data[lang][0])}" for lang in ALL_LANGS))
    metric = sacrebleu.metrics.CHRF(word_order=2)
    result_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for model_spec in args.models:
        model_name = Path(model_spec).name
        print(f"\n{'=' * 76}\nEvaluating {model_name}\n{'=' * 76}")
        model = None
        try:
            checkpoint = resolve_checkpoint(models_dir, model_spec)
            decoder_scale_mode = choose_decoder_scale_mode(model_name, args.decoder_scale_mode, checkpoint)
            set_decoder_scale_mode(decoder_scale_mode)
            print(f"  Decoder embedding compatibility: {decoder_scale_mode}")
            model = load_model(checkpoint, base_model_dir, tokenizer, device)
            per_language: dict[str, float] = {}
            for lang in ALL_LANGS:
                sources, references = dev_data[lang]
                predictions = generate(
                    model, tokenizer, sources, LANG_TO_NLLB[lang], device,
                    batch_size=args.batch_size,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                )
                score = metric.corpus_score(predictions, [references]).score
                per_language[lang] = score
                save_predictions(output_dir, model_name, lang, sources, references, predictions)
                print(f"  {lang.upper()}: chrF++ = {score:.2f}")

            row: dict[str, Any] = {
                "model": model_name,
                "checkpoint_path": str(checkpoint),
                "num_beams": args.num_beams,
                "decoder_scale_mode": decoder_scale_mode,
                "source_truncation": False,
                "max_new_tokens": args.max_new_tokens,
                "avg_chrf": sum(per_language.values()) / len(per_language),
            }
            row.update({f"{lang}_chrf": per_language[lang] for lang in ALL_LANGS})
            result_rows.append(row)
            (output_dir / f"{model_name}_beam{args.num_beams}_metrics.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  Average chrF++: {row['avg_chrf']:.2f}")
        except Exception as error:
            failures.append({"model": model_name, "error": repr(error)})
            print(f"\n[FAILED] {model_name}: {error}")
            if not args.continue_on_error:
                raise
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result_rows.sort(key=lambda row: row["model"])
    if result_rows:
        table = print_table(result_rows)
        print(f"\n{'=' * 76}\nFINAL BEAM-{args.num_beams} TABLE\n{'=' * 76}\n{table}")
        csv_path = output_dir / f"experiment_A_beam{args.num_beams}_results.csv"
        fields = [
            "model", "checkpoint_path", "num_beams", "decoder_scale_mode", "source_truncation", "max_new_tokens", "avg_chrf",
            *[f"{lang}_chrf" for lang in ALL_LANGS],
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result_rows)
        (output_dir / f"experiment_A_beam{args.num_beams}_table.txt").write_text(table + "\n", encoding="utf-8")
        (output_dir / f"experiment_A_beam{args.num_beams}_results.json").write_text(
            json.dumps(result_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nSaved final CSV: {csv_path}")

    if failures:
        failures_path = output_dir / f"experiment_A_beam{args.num_beams}_failures.json"
        failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved failures: {failures_path}")
        if not result_rows:
            raise RuntimeError("No model completed successfully.")


if __name__ == "__main__":
    main()
