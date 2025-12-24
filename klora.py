from typing import Optional, Union
import torch
from torch import nn

glo_count = 0


class KLoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_1_a: torch.Tensor,
        weight_1_b: torch.Tensor,
        weight_2_a: torch.Tensor,
        weight_2_b: torch.Tensor,
        average_ratio: float = 1.0,
        rank: int = 8,
        alpha: int = 8.0,
        beta: int = 0.5,
        sum_timesteps: int = 28000,
        pattern:str = "s*",
        device: Optional[Union[torch.device, str]] = "cuda",
        dtype: Optional[torch.dtype] = None,
        importance_measure: str = 'abs',  
        selection_strategy: str = 'fixed', 
        adaptive_threshold: float = 0.95,  
        percentile: float = 90.0,  
    ):
        super().__init__()
        self.device = device
        self.weight_1_a = weight_1_a.to(device)
        self.weight_1_b = weight_1_b.to(device)
        self.weight_2_a = weight_2_a.to(device)
        self.weight_2_b = weight_2_b.to(device)
        self.average_ratio = average_ratio
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.sum_timesteps = sum_timesteps
        self.out_features = out_features
        self.in_features = in_features
        self.forward_type = "merge"
        self.pattern = pattern
        self.importance_measure = importance_measure
        self.selection_strategy = selection_strategy
        self.adaptive_threshold = adaptive_threshold
        self.percentile = percentile

    def compute_importance_abs(self, matrix: torch.Tensor) -> torch.Tensor:
        return torch.abs(matrix)

    def compute_importance_squared(self, matrix: torch.Tensor) -> torch.Tensor:
        return matrix ** 2

    def compute_importance_frobenius(self, matrix: torch.Tensor) -> torch.Tensor:
        frobenius_norm = torch.norm(matrix, p='fro')
        return torch.ones_like(matrix) * (frobenius_norm / matrix.numel())

    def compute_importance_nuclear(self, matrix: torch.Tensor) -> torch.Tensor:
        try:
            U, S, V = torch.svd(matrix)
            nuclear_norm = S.sum()
            return torch.ones_like(matrix) * (nuclear_norm / matrix.numel())
        except:
            print("WARNING: SVD failed, falling back to absolute value")
            return torch.abs(matrix)

    def compute_importance_spectral(self, matrix: torch.Tensor) -> torch.Tensor:
        spectral_norm = torch.linalg.matrix_norm(matrix, ord=2)
        return torch.ones_like(matrix) * (spectral_norm / matrix.numel())

    def compute_importance(self, matrix: torch.Tensor) -> torch.Tensor:
        if self.importance_measure == 'abs':
            return self.compute_importance_abs(matrix)
        elif self.importance_measure == 'squared':
            return self.compute_importance_squared(matrix)
        elif self.importance_measure == 'frobenius':
            return self.compute_importance_frobenius(matrix)
        elif self.importance_measure == 'nuclear':
            return self.compute_importance_nuclear(matrix)
        elif self.importance_measure == 'spectral':
            return self.compute_importance_spectral(matrix)
        else:
            raise ValueError(f"Unknown importance_measure: {self.importance_measure}")

    def select_topk_fixed(self, importance_matrix: torch.Tensor, k: int) -> torch.Tensor:
        top_k_values, _ = torch.topk(importance_matrix.flatten(), k)
        return top_k_values.sum()

    def select_topk_adaptive(self, importance_matrix: torch.Tensor, k: int) -> torch.Tensor:
        flat_importance = importance_matrix.flatten()
        sorted_values, _ = torch.sort(flat_importance, descending=True)

        cumsum = torch.cumsum(sorted_values, dim=0)
        total = cumsum[-1]
        threshold_value = self.adaptive_threshold * total
        mask = cumsum >= threshold_value

        if mask.any():
            adaptive_k = torch.argmax(mask.float()).item() + 1
        else:
            adaptive_k = min(k, len(sorted_values))

        return sorted_values[:adaptive_k].sum()


    def select_topk(self, importance_matrix: torch.Tensor, k: int) -> torch.Tensor:
        if self.selection_strategy == 'fixed':
            return self.select_topk_fixed(importance_matrix, k)
        elif self.selection_strategy == 'adaptive':
            return self.select_topk_adaptive(importance_matrix, k)
        else:
            raise ValueError(f"Unknown selection_strategy: {self.selection_strategy}")

    def get_klora_weight(self, timestep):
        sum_timesteps = self.sum_timesteps
        k = self.weight_1_a.shape[1] * self.weight_2_a.shape[1]  
        alpha = self.alpha
        beta = self.beta
        gamma = self.average_ratio
        time_ratio = timestep % sum_timesteps
        matrix1 = self.weight_1_a @ self.weight_1_b  
        matrix2 = self.weight_2_a @ self.weight_2_b  
        importance_matrix1 = self.compute_importance(matrix1)  
        top_k_sum1 = self.select_topk(importance_matrix1, k)   
        importance_matrix2 = self.compute_importance(matrix2)  
        top_k_sum2 = self.select_topk(importance_matrix2, k)  
        scale = alpha * time_ratio / sum_timesteps + beta  
        if self.pattern == "s*":
            scale = scale % alpha  
        top_k_sum1 = top_k_sum1 / gamma  
        top_k_sum2 = top_k_sum2 * scale  
        temp_ratio = top_k_sum1 / top_k_sum2
        if temp_ratio > 1:
            return matrix1  
        else:
            return matrix2  

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        global glo_count
        orig_dtype = hidden_states.dtype
        dtype = self.weight_1_a.dtype

        if self.forward_type == "merge":
            glo_count += 1
            weight = self.get_klora_weight(glo_count)
        elif self.forward_type == "weight_1":
            weight = self.weight_1_a @ self.weight_1_b
        elif self.forward_type == "weight_2":
            weight = self.weight_2_a @ self.weight_2_b
        else:
            raise ValueError(self.forward_type)
        hidden_states = nn.functional.linear(hidden_states.to(dtype), weight=weight)
        return hidden_states.to(orig_dtype)


class KLoRALinearLayerInference(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight.dtype
        hidden_states = nn.functional.linear(
            hidden_states.to(dtype), weight=self.weight
        )
        return hidden_states.to(orig_dtype)