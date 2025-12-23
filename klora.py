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
        alpha: int = 1.5,
        beta: int = 0.5,
        sum_timesteps: int = 28000,
        pattern:str = "s*",
        device: Optional[Union[torch.device, str]] = "cuda",
        dtype: Optional[torch.dtype] = None,
        # ===== EXPERIMENTAL PARAMETERS (add these to test new methods) =====
        importance_measure: str = 'abs',  # Options: 'abs', 'squared', 'frobenius', 'nuclear', 'spectral'
        selection_strategy: str = 'fixed',  # Options: 'fixed', 'adaptive'
        adaptive_threshold: float = 0.95,  # For adaptive strategy: energy threshold (0.0-1.0)
        percentile: float = 90.0,  # For percentile strategy: percentile value (0.0-100.0)
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

        # ===== EXPERIMENTAL SETTINGS =====
        self.importance_measure = importance_measure
        self.selection_strategy = selection_strategy
        self.adaptive_threshold = adaptive_threshold
        self.percentile = percentile

    # ========================================================================
    # IMPORTANCE MEASURE METHODS (Paper Equation 1-2: ΔW'c = |ΔWc|, ΔW's = |ΔWs|)
    # ========================================================================

    def compute_importance_abs(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        ORIGINAL METHOD (Paper Equation 1-2)
        Absolute value: ΔW' = |ΔW|
        """
        return torch.abs(matrix)

    def compute_importance_squared(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        EXPERIMENTAL: L2 Norm (Element-wise squared)
        ΔW' = (ΔW)²
        """
        return matrix ** 2

    def compute_importance_frobenius(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        EXPERIMENTAL: Frobenius Norm
        ||ΔW||_F = sqrt(sum of all squared elements)
        """
        frobenius_norm = torch.norm(matrix, p='fro')
        # Normalize to make it comparable with element-wise methods
        return torch.ones_like(matrix) * (frobenius_norm / matrix.numel())

    def compute_importance_nuclear(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        EXPERIMENTAL: Nuclear Norm (Sum of Singular Values)
        ||ΔW||_* = sum of singular values
        Use: Measures true low-rank importance
        Pros: Theoretically optimal for low-rank matrices (LoRA is low-rank!)
        Cons: Expensive to compute (SVD), may fail for some matrices
        """
        try:
            U, S, V = torch.svd(matrix)
            nuclear_norm = S.sum()
            # Normalize to make it comparable
            return torch.ones_like(matrix) * (nuclear_norm / matrix.numel())
        except:
            # Fallback to abs if SVD fails (numerical instability)
            print("WARNING: SVD failed, falling back to absolute value")
            return torch.abs(matrix)

    def compute_importance_spectral(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        EXPERIMENTAL: Spectral Norm (Largest Singular Value)
        ||ΔW||_2 = largest singular value
        Use: Measures maximum directional amplification
        Pros: Captures dominant direction of transformation
        Cons: Ignores other directions, expensive to compute
        """
        spectral_norm = torch.linalg.matrix_norm(matrix, ord=2)
        # Normalize to make it comparable
        return torch.ones_like(matrix) * (spectral_norm / matrix.numel())

    def compute_importance(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        DISPATCHER: Select importance measure based on self.importance_measure
        To experiment, change self.importance_measure in __init__
        """
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

    # ========================================================================
    # TOP-K SELECTION METHODS (Paper Equation 3-4: Sum of Top-K elements)
    # ========================================================================

    def select_topk_fixed(self, importance_matrix: torch.Tensor, k: int) -> torch.Tensor:
        """
        ORIGINAL METHOD (Paper Equation 3-4)
        Fixed K selection: K = rc * rs (product of ranks)
        Use: Simple, consistent across all layers
        """
        top_k_values, _ = torch.topk(importance_matrix.flatten(), k)
        return top_k_values.sum()

    def select_topk_adaptive(self, importance_matrix: torch.Tensor, k: int) -> torch.Tensor:
        """
        EXPERIMENTAL: Adaptive K based on cumulative energy
        Selects K such that top-K elements capture X% of total energy
        Use: Adapts to each layer's importance distribution
        Pros: More principled, layer-specific
        Cons: May select very different K values per layer

        Example: If threshold=0.95, select smallest K where sum(top-K) >= 0.95 * sum(all)
        """
        flat_importance = importance_matrix.flatten()
        sorted_values, _ = torch.sort(flat_importance, descending=True)

        cumsum = torch.cumsum(sorted_values, dim=0)
        total = cumsum[-1]

        # Find K where cumulative sum >= threshold * total
        threshold_value = self.adaptive_threshold * total
        mask = cumsum >= threshold_value

        if mask.any():
            adaptive_k = torch.argmax(mask.float()).item() + 1
        else:
            # Fallback to original K if threshold too high
            adaptive_k = min(k, len(sorted_values))

        return sorted_values[:adaptive_k].sum()


    def select_topk(self, importance_matrix: torch.Tensor, k: int) -> torch.Tensor:
        """
        DISPATCHER: Select Top-K strategy based on self.selection_strategy
        To experiment, change self.selection_strategy in __init__
        """
        if self.selection_strategy == 'fixed':
            return self.select_topk_fixed(importance_matrix, k)
        elif self.selection_strategy == 'adaptive':
            return self.select_topk_adaptive(importance_matrix, k)
        else:
            raise ValueError(f"Unknown selection_strategy: {self.selection_strategy}")

    # ========================================================================
    # MAIN WEIGHT SELECTION METHOD
    # ========================================================================

    def get_klora_weight(self, timestep):
        """
        Main method to select between content and style LoRA weights

        ORIGINAL ALGORITHM (Paper Section 3.2, Algorithm 1):
        1. Compute ΔW matrices (Equation: ΔW = B·A)
        2. Take absolute values (Equation 1-2)
        3. Select Top-K elements (Equation 3-4, K from Equation 5)
        4. Apply scaling factors (Equation 7-10)
        5. Compare and select (Equation 6)

        EXPERIMENTAL MODIFICATIONS:
        - Step 2: Can use different importance measures (squared, nuclear, etc.)
        - Step 3: Can use different selection strategies (adaptive, percentile)
        """
        sum_timesteps = self.sum_timesteps
        k = self.weight_1_a.shape[1] * self.weight_2_a.shape[1]  # Equation 5: K = rc * rs
        alpha = self.alpha
        beta = self.beta
        gamma = self.average_ratio

        # ===== STEP 1: Compute LoRA weight matrices (ΔW = B·A) =====
        time_ratio = (timestep % sum_timesteps) / sum_timesteps
        t = torch.tensor(time_ratio, device=self.weight_1_a.device, dtype=self.weight_1_a.dtype)
        matrix1 = self.weight_1_a @ self.weight_1_b  # Content LoRA: ΔWc = Bc·Ac
        matrix2 = self.weight_2_a @ self.weight_2_b  # Style LoRA: ΔWs = Bs·As

        # ===== OPTION 1: ORIGINAL CODE (for baseline comparison) =====
        # Uncomment this block to use the original method
        # abs_matrix = torch.abs(matrix1)  # ΔW'c = |ΔWc|
        # top_k_values, _ = torch.topk(abs_matrix.flatten(), k)
        # top_k_sum1 = top_k_values.sum()
        # abs_matrix = torch.abs(matrix2)  # ΔW's = |ΔWs|
        # top_k_values, _ = torch.topk(abs_matrix.flatten(), k)
        # top_k_sum2 = top_k_values.sum()

        # ===== OPTION 2: EXPERIMENTAL CODE (uses new methods) =====
        # Comment this block to use original code instead
        importance_matrix1 = self.compute_importance(matrix1)  # STEP 2: Compute importance
        top_k_sum1 = self.select_topk(importance_matrix1, k)   # STEP 3: Select Top-K

        importance_matrix2 = self.compute_importance(matrix2)  # STEP 2: Compute importance
        top_k_sum2 = self.select_topk(importance_matrix2, k)   # STEP 3: Select Top-K

        # ===== STEP 4: Apply scaling factors (Equation 7-10) =====
        scale = alpha * t + beta  # Equation 7 (t is now scheduled)
        if self.pattern == "s*":
            scale = scale % alpha   # Equation 8 (alternative scale)

        # Apply gamma and scale (Equation 9-10)
        top_k_sum1 = top_k_sum1 / gamma  # Normalize by content/style ratio
        top_k_sum2 = top_k_sum2 * scale  # Apply time-dependent scaling

        # ===== STEP 5: Compare and select (Equation 6) =====
        temp_ratio = top_k_sum1 / top_k_sum2
        if temp_ratio > 1:
            return matrix1  # Use content LoRA
        else:
            return matrix2  # Use style LoRA

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
