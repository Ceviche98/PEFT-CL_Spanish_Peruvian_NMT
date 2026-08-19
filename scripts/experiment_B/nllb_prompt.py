import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Dict, Any, Union
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Attention, M2M100ForConditionalGeneration

class CaLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=1.0, dropout=0.0):
        super().__init__()
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # Match PEFT's stable mixed-precision LoRA path: A/B remain fp32 while
        # the frozen NLLB backbone can run in bf16/fp16.  Casting the parameters
        # themselves to the activation dtype here made CaLoRA train its adapters
        # in bf16, unlike the plain PEFT Q/V control.
        output_dtype = x.dtype
        adapter_input = x.to(dtype=self.lora_A.weight.dtype)
        update = self.lora_B(self.lora_A(self.dropout(adapter_input))) * self.scaling
        return update.to(dtype=output_dtype)

class CustomM2M100Attention(nn.Module):
    def __init__(self, original_attn: M2M100Attention, prompt_config: dict):
        super().__init__()
        self.embed_dim = original_attn.embed_dim
        self.num_heads = original_attn.num_heads
        self.dropout = original_attn.dropout
        self.head_dim = original_attn.head_dim
        self.scaling = original_attn.scaling
        self.is_decoder = original_attn.is_decoder
        self.is_causal = original_attn.is_causal
        self.layer_idx = getattr(original_attn, "layer_idx", None)

        # Base parameters
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.q_proj = original_attn.q_proj
        self.out_proj = original_attn.out_proj

        # CaLoRA specific configs
        self.lora_r = prompt_config.get("lora_r", 8)
        self.lora_alpha = prompt_config.get("lora_alpha", 1.0)
        self.lora_dropout = prompt_config.get("lora_dropout", 0.0)
        self.task_id = prompt_config.get("task_id", 0)
        
        # Current-task attention-only CaLoRA.  Q/K/V/O is the final
        # capacity-controlled configuration used by Experiment B: it adapts
        # the complete attention block while avoiding the much larger FFN
        # adapters (fc1/fc2).
        self.lora_q = CaLoRALinear(self.embed_dim, self.embed_dim, self.lora_r, self.lora_alpha, self.lora_dropout)
        self.lora_k = CaLoRALinear(self.embed_dim, self.embed_dim, self.lora_r, self.lora_alpha, self.lora_dropout)
        self.lora_v = CaLoRALinear(self.embed_dim, self.embed_dim, self.lora_r, self.lora_alpha, self.lora_dropout)
        self.lora_o = CaLoRALinear(self.embed_dim, self.embed_dim, self.lora_r, self.lora_alpha, self.lora_dropout)
        
        # Previous Tasks LoRA (Frozen)
        self.previous_lora_weights_q = nn.ModuleList([
            CaLoRALinear(self.embed_dim, self.embed_dim, self.lora_r, self.lora_alpha, self.lora_dropout)
            for _ in range(self.task_id)
        ])
        self.previous_lora_weights_k = nn.ModuleList([
            CaLoRALinear(self.embed_dim, self.embed_dim, self.lora_r, self.lora_alpha, self.lora_dropout)
            for _ in range(self.task_id)
        ])
        self.previous_lora_weights_v = nn.ModuleList([
            CaLoRALinear(self.embed_dim, self.embed_dim, self.lora_r, self.lora_alpha, self.lora_dropout)
            for _ in range(self.task_id)
        ])
        self.previous_lora_weights_o = nn.ModuleList([
            CaLoRALinear(self.embed_dim, self.embed_dim, self.lora_r, self.lora_alpha, self.lora_dropout)
            for _ in range(self.task_id)
        ])
        for previous_modules in (
            self.previous_lora_weights_q,
            self.previous_lora_weights_k,
            self.previous_lora_weights_v,
            self.previous_lora_weights_o,
        ):
            for p in previous_modules.parameters():
                p.requires_grad = False
        
        # This will hold the routing weights computed at the model level
        self.routing_weights = None

    def _blend_previous_adapters(self, current_update, adapter_input, previous_modules):
        """Apply inference-only routing to one Q/K/V/O adapter family.

        The shared-adapter Experiment-B path uses ``task_id=0`` and therefore
        returns the current update directly. Keeping this generic preserves
        compatibility with the older task-indexed inference path.
        """
        if self.task_id == 0 or self.routing_weights is None or self.training:
            return current_update

        rw = self.routing_weights.to(dtype=current_update.dtype)
        if rw.size(0) != current_update.size(0) and current_update.size(0) % max(1, rw.size(0)) == 0:
            rw = rw.repeat_interleave(current_update.size(0) // rw.size(0), dim=0)
        blended = current_update * rw[:, -1].unsqueeze(-1).unsqueeze(-1)
        for i in range(self.task_id):
            blended += previous_modules[i](adapter_input) * rw[:, i].unsqueeze(-1).unsqueeze(-1)
        return blended

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        past_key_values = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len = hidden_states.shape[:-1]
        src_len = key_value_states.shape[1] if is_cross_attention else tgt_len

        q_input_shape = (bsz, tgt_len, -1, self.head_dim)
        kv_input_shape = (bsz, src_len, -1, self.head_dim)

        # GET QUERY PROJ
        base_query_states = self.q_proj(hidden_states)
        
        # CaLoRA injection for Q
        lora_query_states = self._blend_previous_adapters(
            self.lora_q(hidden_states), hidden_states, self.previous_lora_weights_q
        )
        # IMPORTANT: routing/softmax blending across current + frozen prior-task adapters is only
        # meaningful when task identity is unknown (multi-task inference). During training of the
        # current task, the current adapter must get the FULL, unattenuated gradient. Blending it
        # here (weighted by a softmax over prompt keys) starves the new adapter's gradient whenever
        # an older, longer-trained key has a much larger norm than the freshly-initialized current
        # key -- which is exactly what happens across all these tasks, since every task shares the
        # same Spanish source sentence and the router has no other signal to discriminate on.
        query_states = (base_query_states + lora_query_states).view(*q_input_shape).transpose(1, 2)

        # CACHE LOGIC (Kept identical to HF)
        is_updated = False
        if past_key_values is not None:
            if hasattr(past_key_values, "is_updated"):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values
        else:
            base_key_states = self.k_proj(current_states)
            base_value_states = self.v_proj(current_states)
            
            # CaLoRA injection for K and V. K is adapted before cache storage,
            # so cached autoregressive and cross-attention keys remain correct.
            lora_key_states = self._blend_previous_adapters(
                self.lora_k(current_states), current_states, self.previous_lora_weights_k
            )
            lora_value_states = self._blend_previous_adapters(
                self.lora_v(current_states), current_states, self.previous_lora_weights_v
            )

            key_states = (base_key_states + lora_key_states).view(*kv_input_shape).transpose(1, 2)
            value_states = (base_value_states + lora_value_states).view(*kv_input_shape).transpose(1, 2)

            if past_key_values is not None:
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                if is_cross_attention and hasattr(past_key_values, "is_updated"):
                    past_key_values.is_updated[self.layer_idx] = True

        # Match Hugging Face's M2M100 SDPA rule exactly.  In cached decoding the
        # query has length 1 but key/value contain the whole generated prefix.
        # Passing ``is_causal=True`` in that case makes PyTorch build an upper-
        # left causal mask and the new token can attend only to the *first* key,
        # rather than every key in its prefix.  This is harmless during
        # teacher-forced training (where tgt_len > 1) but corrupts autoregressive
        # ``generate()`` and produces repetitive, off-language outputs.
        #
        # This is the same condition used by transformers' sdpa_attention_forward.
        use_causal_mask = (
            query_states.shape[2] > 1
            and attention_mask is None
            and self.is_causal
        )

        # Attention calculation using the same SDPA scale as M2M100.
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=use_causal_mask,
        )
        
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, -1).contiguous()
        # PEFT parity for an ``o_proj`` LoRA wrapper:
        # base_o(attn_output) + B(A(attn_output)) * scaling.
        lora_output_states = self._blend_previous_adapters(
            self.lora_o(attn_output), attn_output, self.previous_lora_weights_o
        )
        attn_output = self.out_proj(attn_output) + lora_output_states

        if isinstance(past_key_values, tuple):
            return attn_output, None, past_key_values
        return attn_output, None


class NLLBForConditionalGenerationWithCaLoRA(nn.Module):
    def __init__(self, model_name, prompt_config, **kwargs):
        super().__init__()
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name, **kwargs)
        self.config = self.model.config
        self.generation_config = getattr(self.model, "generation_config", None)
        self.main_input_name = getattr(self.model, "main_input_name", "input_ids")
        self.warnings_issued = getattr(self.model, "warnings_issued", {})
        self.prompt_config = prompt_config
        self.task_id = prompt_config.get("task_id", 0)
        self.hidden_dim = self.model.config.d_model
        self.trans_hidden_dim = prompt_config.get("trans_hidden_dim", 100)
        self.attn_temperature = prompt_config.get("attn_temperature", 1.0)
        
        # CaLoRA Prompt Routing components
        self.trans_input = nn.Sequential(
            nn.Linear(self.hidden_dim, self.trans_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.trans_hidden_dim, self.trans_hidden_dim)
        )
        self.prompt_key = nn.Parameter(torch.randn(1, self.trans_hidden_dim))
        if self.task_id > 0:
            self.previous_prompts_keys = nn.Parameter(torch.randn(self.task_id, self.trans_hidden_dim))
            self.previous_prompts_keys.requires_grad = False
            
        self.all_attn_weights = []
        self.is_inference = False
        
        # 1. Freeze entire base model EXCEPT embeddings and lm_head
        for name, param in self.model.named_parameters():
            if "embed_tokens" in name or "shared" in name or "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
        # 2. Replace attention layers (which will instantiate new trainable LoRA params)
        self._replace_attention_layers()
        
        # 3. Ensure CaLoRA Prompt Routing components are trainable
        for param in self.trans_input.parameters():
            param.requires_grad = True
        self.prompt_key.requires_grad = True
        # previous_prompts_keys is already set to requires_grad=False if task_id > 0

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            if "_modules" in self.__dict__ and "model" in self._modules:
                return getattr(self._modules["model"], name)
            if "model" in self.__dict__:
                return getattr(self.__dict__["model"], name)
            raise
        
    def _replace_attention_layers(self):
        # Replace Encoder layers
        for layer in self.model.model.encoder.layers:
            layer.self_attn = CustomM2M100Attention(layer.self_attn, self.prompt_config)
            
        # Replace Decoder layers
        for layer in self.model.model.decoder.layers:
            layer.self_attn = CustomM2M100Attention(layer.self_attn, self.prompt_config)
            layer.encoder_attn = CustomM2M100Attention(layer.encoder_attn, self.prompt_config)

    def _compute_routing_weights(self, encoder_hidden_states):
        # Compute global routing weights based on average encoder hidden state
        # encoder_hidden_states: (bsz, seq_len, hidden_dim)
        target_dtype = self.trans_input[0].weight.dtype
        avg_hidden = encoder_hidden_states.mean(dim=1).to(dtype=target_dtype) # (bsz, hidden_dim)
        query = self.trans_input(avg_hidden) # (bsz, trans_hidden_dim)
        
        if self.task_id > 0:
            keys = torch.cat([self.previous_prompts_keys, self.prompt_key], dim=0).to(dtype=target_dtype) # (task_id + 1, trans_hidden_dim)
        else:
            keys = self.prompt_key.to(dtype=target_dtype) # (1, trans_hidden_dim)
            
        # Attention logic: Q * K^T
        scores = torch.matmul(query, keys.T) / self.attn_temperature # (bsz, task_id + 1)
        weights = F.softmax(scores, dim=-1)
        
        if self.is_inference:
            self.all_attn_weights.append(weights.detach().cpu().numpy())
            
        return weights

    def _one_hot_routing_weights(self, task_idx, batch_size, device, dtype):
        """
        Builds a deterministic one-hot routing vector that activates exactly one adapter
        (task_idx in [0 .. self.task_id], where self.task_id itself = the current/active adapter,
        matching the column ordering used in _compute_routing_weights: [previous..., current]).

        Use this instead of the learned softmax router whenever the task/language being processed
        is actually known (e.g. per-language dev evaluation in the Triangular Evaluation Matrix).
        The learned router has no reliable signal to fall back on here anyway, since every task
        shares the same Spanish source sentence.
        """
        assert 0 <= task_idx <= self.task_id, f"task_idx {task_idx} out of range [0, {self.task_id}]"
        weights = torch.zeros(batch_size, self.task_id + 1, device=device, dtype=dtype)
        weights[:, task_idx] = 1.0
        return weights

    def _inject_routing_weights(self, weights):
        for layer in self.model.model.encoder.layers:
            layer.self_attn.routing_weights = weights
        for layer in self.model.model.decoder.layers:
            layer.self_attn.routing_weights = weights
            layer.encoder_attn.routing_weights = weights

    def _get_encoder_embeds(self, input_ids=None, **kwargs):
        embeds = kwargs.get("inputs_embeds")
        if embeds is not None:
            return embeds
        if input_ids is not None:
            embed_tokens = getattr(self.model.model.encoder, "embed_tokens", None) or self.model.get_input_embeddings()
            # M2M100ScaledWordEmbedding performs the configured scaling itself.
            # Keeping this representation on the model's native scale also keeps
            # the optional CaLoRA router numerically consistent with the encoder.
            return embed_tokens(input_ids)
        return None

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None, target_task_idx: Optional[int] = None, **kwargs):
        encoder_embeds = self._get_encoder_embeds(input_ids=input_ids, **kwargs)
        if encoder_embeds is not None:
            if target_task_idx is not None:
                bsz = encoder_embeds.size(0)
                routing_weights = self._one_hot_routing_weights(
                    target_task_idx, bsz, encoder_embeds.device, self.trans_input[0].weight.dtype
                )
            else:
                routing_weights = self._compute_routing_weights(encoder_embeds)
            self._inject_routing_weights(routing_weights)
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )
        
    def generate(self, input_ids=None, target_task_idx: Optional[int] = None, **kwargs):
        encoder_embeds = self._get_encoder_embeds(input_ids=input_ids, **kwargs)
        if encoder_embeds is not None:
            if target_task_idx is not None:
                bsz = encoder_embeds.size(0)
                routing_weights = self._one_hot_routing_weights(
                    target_task_idx, bsz, encoder_embeds.device, self.trans_input[0].weight.dtype
                )
            else:
                routing_weights = self._compute_routing_weights(encoder_embeds)
            self._inject_routing_weights(routing_weights)
        return self.model.generate(input_ids=input_ids, **kwargs)
