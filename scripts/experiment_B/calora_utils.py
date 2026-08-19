"""
Pure mathematical helper functions for Causal-Aware LoRA (CaLoRA) Continual Learning.
Extracted cleanly from CaLora/src/CaLoRA_cl_trainer.py to prevent namespace collisions,
Optional/dataclass import failures in modern transformers (cl_collator.py), and vendored
_inner_training_loop incompatibilities.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any, Union


def global_softmax_concat(tensor_list: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    """
    Converts causal-effect scores into *multiplicative gradient scales*, independently
    for each tensor.  Each returned tensor has mean one.

    NOTE: The CaLoRA paper defines the causal-effect weight per weight matrix:
    "Ê_Tt = Softmax(E_Tt)" where E_Tt in R^(dO x dI) -- i.e. normalized *within* a single LoRA
    The paper's softmax is an importance *distribution* (it sums to one).  It is not
    directly a usable optimizer multiplier: applying it as ``grad *= softmax(score)``
    shrinks the average gradient by ``1 / tensor.numel()``.  For a 1024 x 128
    LoRA matrix this is 7.6e-6; with AdamW's epsilon this is enough to stall the
    adapter, especially while LoRA-B is zero-initialised.  We therefore preserve
    softmax's relative attribution, but multiply it by ``numel`` so an uninformative
    (uniform) attribution is exactly the identity correction.

    Normalisation is deliberately per tensor.  It avoids unrelated layers competing
    for probability mass and keeps the expected correction invariant to rank/model size.
    """
    if not tensor_list:
        return []

    result = []
    for t in tensor_list:
        flat = t.flatten().to(device)
        if not torch.isfinite(flat).all():
            # A bad Taylor score must never poison a complete optimizer step.
            result.append(torch.ones_like(t, device=device))
            continue
        softmax_vals = torch.softmax(flat, dim=0)
        result.append((softmax_vals * flat.numel()).view(t.shape))
    return result


def compute_Et(loss: torch.Tensor, t_trainable_params: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    """
    Computes PaCA (Parameter-level Counterfactual Attribution) scales from a Taylor approximation.
    Uses the first-order Taylor term plus the legacy diagonal-Hessian approximation,
    then returns mean-one PaCA gradient scales.

    Note: p.grad is populated by backward() without create_graph=True, so the legacy
    Hessian branch cannot carry a graph and evaluates to zero.  Keeping the harmless
    zero fallback makes this a stable first-order PaCA implementation rather than
    silently pretending to calculate a second derivative.
    """
    loss_item = loss.item()
    grad_W_t = []
    for p in t_trainable_params:
        g = p.grad if p.grad is not None else torch.zeros_like(p, device=device)
        grad_W_t.append(g.to(device))
    
    hessian_W_t = []
    for g, p in zip(grad_W_t, t_trainable_params):
        try:
            grad_out = torch.ones_like(g, device=device, requires_grad=True)
            if g.requires_grad:
                second_grad = torch.autograd.grad(
                    g, p, 
                    grad_outputs=grad_out,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )[0]
                hessian_W_t.append(second_grad.to(device) if second_grad is not None else torch.zeros_like(p, device=device))
            else:
                hessian_W_t.append(torch.zeros_like(p, device=device))
        except RuntimeError as e:
            print(f"Error computing second gradient in compute_Et: {e}")
            hessian_W_t.append(torch.zeros_like(p, device=device))
    
    delta_W_t = [p.clone().detach().requires_grad_(False).to(device) for p in t_trainable_params]
        
    eps = 1e-6
    taylor_approx = [
        ((-g.detach() * d + 0.5 * (h.detach() + eps) * (d ** 2 + eps)) / max(1e-8, loss_item)).to(device)
        for g, h, d in zip(grad_W_t, hessian_W_t, delta_W_t)
    ]

    return global_softmax_concat(taylor_approx, device)


def vector_or_matrix_cosine_similarity(A: torch.Tensor, B: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Computes column-wise or vector cosine similarity and weights B by the sign of similarity.
    """
    device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    B = B.to(device)
    
    # 1D vector case
    if A.dim() == 1 and B.dim() == 1:
        if A.shape != B.shape:
            raise ValueError("1D input dimensions do not match.")
        cos_sim = F.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0))[0]
        sign_sim = torch.sign(cos_sim)
        return sign_sim * B
        
    # 2D matrix case
    elif A.dim() == 2 and B.dim() == 2:
        if A.shape != B.shape:
            raise ValueError("2D input dimensions do not match.")
        cos_sim = F.cosine_similarity(A.T, B.T)
        sign_sim = torch.sign(cos_sim)
        return B * sign_sim.reshape(1, -1)
    
    else:
        raise ValueError(f"Inputs must be 1D vectors or 2D matrices (got dim {A.dim()} and {B.dim()}).")


@torch.no_grad()
def build_lora_svd_projection_basis(B: torch.Tensor, device: Optional[torch.device] = None) -> dict[str, Any]:
    """Factorise one fixed historical CaGA gradient once.

    A historical gradient is unchanged throughout a later task.  Its SVD basis
    can therefore be cached at task start and reused for every optimizer step,
    producing the same orthogonal projection as repeatedly factorising ``B``.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_dtype = B.dtype
    compute_dtype = torch.float32 if original_dtype in (torch.bfloat16, torch.float16) else original_dtype
    B = B.detach().to(device, dtype=compute_dtype)
    eps = 1e-8

    if B.ndim == 1:
        return {
            "kind": "vector",
            "basis": B / (torch.norm(B) + eps),
            "source_shape": tuple(B.shape),
            "output_dtype": original_dtype,
        }
    if B.ndim != 2:
        raise ValueError(f"CaGA only supports 1D or 2D gradients, got {B.ndim}D.")

    U, S, _ = torch.linalg.svd(B, full_matrices=False)
    rank = torch.sum(S > S[0] * max(1e-6, eps)).item()
    return {
        "kind": "matrix",
        "basis": U[:, :rank],
        "source_shape": tuple(B.shape),
        "output_dtype": original_dtype,
    }


@torch.no_grad()
def lora_project_with_cached_basis(
    projection_basis: dict[str, Any], A: torch.Tensor, device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a current CaGA gradient using a cached historical SVD basis."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basis = projection_basis["basis"]
    if tuple(A.shape) != projection_basis["source_shape"]:
        raise ValueError(
            f"CaGA gradient shape mismatch: expected {projection_basis['source_shape']}, found {tuple(A.shape)}."
        )
    A = A.detach().to(device, dtype=basis.dtype)
    eps = 1e-8

    if projection_basis["kind"] == "vector":
        cos_sim = torch.dot(basis, A / (torch.norm(A) + eps))
        A_proj = cos_sim * basis * torch.norm(A)
    else:
        A_proj = basis @ (basis.T @ A)

    A_norm = torch.norm(A) + eps
    A_proj_norm = torch.norm(A_proj) + eps
    proj_ratio = (A_proj_norm / A_norm).clamp(max=1.0)
    A_proj_normalized = (A_proj / A_proj_norm).to(projection_basis["output_dtype"])
    return A_proj_normalized, proj_ratio


def lora_project_svd(B: torch.Tensor, A: torch.Tensor, device: Optional[torch.device] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference CaGA projection that factorises ``B`` on demand.

    Retained for unit tests and compatibility. Training should use
    :func:`build_lora_svd_projection_basis` once per historical gradient and
    :func:`lora_project_with_cached_basis` at every optimizer step.
    """
    projection_basis = build_lora_svd_projection_basis(B, device)
    return lora_project_with_cached_basis(projection_basis, A, device)
