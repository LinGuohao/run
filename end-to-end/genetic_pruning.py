"""
Genetic Algorithm-based Pruning: Attention-FFN Layer Decoupling with Loop Support

Core Idea:
- Original model: 40 layers × (1 Attention + 1 FFN) = 80 modules
- Module sequence: [A0, F0, A1, F1, A2, F2, ..., A39, F39]
- Genetic algorithm selects modules and determines loop counts
- **CONSTRAINT: Forward-only paths (maintain order, no backward connections)**

Chromosome Encoding (UPDATED with Loop Support):
- Ternary mask: [0,1,2,1,0,2,...] (length 80, values in {0,1,2})
- 0 = skip this module
- 1 = execute this module once (no loop)
- 2 = execute this module twice (participate in loop)
- Consecutive >=2 values form a "loop block" that repeats together
- Isolated 2 values execute 2 times independently

Example chromosomes:
1. [1,1,1,1,1,1,...] (all 80 modules, 1x each) - equivalent to 40-layer depth
2. [0,2,2,1,0,2,...] - path: F0→A1→F0→A1→F1→F2→F2 (loop block + singles)
3. [2,2,0,0,2,2,...] - path: A0→F0→A0→F0→A2→F2→A2→F2 (two loop blocks)

Loop Encoding Rules:
- Consecutive >=2 values form a loop block
- Loop blocks execute max(values) times together
- Example: [2,2,1] → execute [0,1] twice, then [2] once
- Example: [2,1,2] → execute [0] twice, [1] once, [2] twice (3 separate segments)

Valid constraints:
✓ Selected modules maintain original order
✓ Total parameters ≤ 50% of original (based on unique modules)
✓ Effective depth can be > 40 layers (through loops)

Fitness Function:
- Primary: Perplexity (lower is better)
- Constraint: Unique module parameters ≤ 50% of original

Genetic Operations:
- Selection: Tournament selection
- Crossover: Uniform crossover
- Mutation: Gradual transitions (0↔1↔2, no 0↔2 direct jumps)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Union
from dataclasses import dataclass
import copy
from tqdm import tqdm
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def decode_chromosome(chromosome: List[int]) -> List[int]:
    """
    Decode a loop-encoded chromosome into an execution path.

    Encoding rules (updated for multi-value support):
    - 0 = skip
    - 1 = execute once (no loop)
    - 2+ = execute N times (participate in loop or standalone)
    - Consecutive identical values form a "loop block" that repeats together
    - Different loop count values create separate blocks

    Args:
        chromosome: List[int], values in {0,1,2,...,max_loop_count}

    Returns:
        path: List[int], execution path (module indices in order)

    Example:
        [1,1,2,2,3,3,2,2] → [0,1, 2,3,2,3, 4,5,4,5,4,5, 6,7,6,7]
        - Block 1: modules 0,1 (execute 1x)
        - Block 2: modules 2,3 (execute 2x)
        - Block 3: modules 4,5 (execute 3x)
        - Block 4: modules 6,7 (execute 2x)
    """
    path = []
    i = 0

    while i < len(chromosome):
        # Skip 0
        if chromosome[i] == 0:
            i += 1
            continue

        # Execute once (value == 1)
        if chromosome[i] == 1:
            path.append(i)
            i += 1
            continue

        # Loop block - find consecutive identical values
        if chromosome[i] >= 2:
            current_loop_value = chromosome[i]
            block_modules = []

            # Find all consecutive positions with the exact same loop value
            while i < len(chromosome) and chromosome[i] == current_loop_value:
                block_modules.append(i)
                i += 1

            # Execute this specific loop block (current_loop_value) times
            for _ in range(current_loop_value):
                path.extend(block_modules)

            # Note: If there are more positions with different loop values,
            # they will be handled in separate blocks in the next iteration

    return path


def count_unique_modules(chromosome: List[int]) -> int:
    """
    Count the number of unique modules (non-zero values) in chromosome.

    Args:
        chromosome: List[int], values in {0,1,2,...,max_loop_count}

    Returns:
        count: int, number of modules with non-zero values

    Example:
        [0,1,2,2,0,3,1] → 5 (indices 1,2,3,5,6 are non-zero)
    """
    return sum(1 for gene in chromosome if gene != 0)


@dataclass
class Individual:
    """Individual in genetic algorithm population."""
    chromosome: List[int]  # Ternary mask for 80 modules [0/1/2, 0/1/2, ...]
    fitness: float = float('inf')  # PPL (lower is better)
    params_ratio: float = 0.0  # Parameter ratio compared to original (based on unique modules)
    is_valid: bool = False  # Whether satisfies parameter constraint
    num_modules: int = 0  # Number of unique selected modules (>0)
    effective_depth: int = 0  # Effective depth after loop expansion

    def __repr__(self):
        return f"Individual(fitness={self.fitness:.2f}, params={self.params_ratio:.2%}, modules={self.num_modules}/80, depth={self.effective_depth}, valid={self.is_valid})"


class DecoupledLlamaLayer(nn.Module):
    """A flexible layer that can be either Attention or FFN."""
    def __init__(self, module_type: str, module: nn.Module, layernorm: nn.Module = None):
        """
        Args:
            module_type: 'attention' or 'ffn'
            module: self_attn or mlp module
            layernorm: input_layernorm (for attn) or post_attention_layernorm (for ffn)
        """
        super().__init__()
        self.module_type = module_type
        self.module = module
        self.layernorm = layernorm

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        """Forward pass."""
        try:
            residual = hidden_states

            # Apply layernorm
            if self.layernorm is not None:
                hidden_states = self.layernorm(hidden_states)
                if hidden_states is None:
                    raise ValueError(f"layernorm returned None (type: {self.module_type})")

            # Apply module
            if self.module_type == 'attention':
                # Call attention with appropriate parameters based on transformers version
                if position_embeddings is not None:
                    # Check if the attention module's forward method accepts position_embeddings
                    import inspect
                    sig = inspect.signature(self.module.forward)
                    if 'position_embeddings' in sig.parameters:
                        attn_output = self.module(
                            hidden_states,
                            position_embeddings=position_embeddings,
                            attention_mask=attention_mask,
                        )
                    else:
                        attn_output = self.module(
                            hidden_states,
                            attention_mask=attention_mask,
                        )
                else:
                    # Old transformers (< 4.43): requires position_ids
                    batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]
                    position_ids = torch.arange(
                        0, seq_length, dtype=torch.long, device=hidden_states.device
                    ).unsqueeze(0).expand(batch_size, -1)
                    attn_output = self.module(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )

                if attn_output is None:
                    raise ValueError("attention module returned None")
                # Handle tuple output (hidden_states, attention_weights, ...)
                if isinstance(attn_output, tuple):
                    hidden_states = attn_output[0]
                else:
                    hidden_states = attn_output

                if hidden_states is None:
                    raise ValueError("attention hidden_states is None after tuple extraction")
            else:  # ffn
                hidden_states = self.module(hidden_states)
                if hidden_states is None:
                    raise ValueError("FFN module returned None")

            # Residual connection
            hidden_states = residual + hidden_states

            return hidden_states

        except Exception as e:
            import traceback
            import sys
            print(f"\n{'='*80}")
            print(f"Error in DecoupledLlamaLayer ({self.module_type}): {e}")
            print(f"{'='*80}")
            print("Full traceback:")
            traceback.print_exception(*sys.exc_info())
            print(f"{'='*80}\n")
            raise


class DecoupledLlamaModel(nn.Module):
    """
    LLaMA model with flexible module selection and loop support.

    Builds model from a ternary chromosome with loop support:
    [A0, F0, A1, F1, A2, F2, ..., A39, F39]
    Values in {0,1,2}: 0=skip, 1=once, 2=participate in loop
    """
    def __init__(self, original_model: nn.Module, chromosome: List[int], copy_components: bool = True):
        super().__init__()
        self.config = original_model.config
        self.chromosome = chromosome

        # Deep copy embeddings and norm to ensure device independence
        # This is critical for multi-GPU: each DecoupledLlamaModel copy gets its own components
        if copy_components:
            import copy as copy_module
            self.embed_tokens = copy_module.deepcopy(original_model.model.embed_tokens)
            self.norm = copy_module.deepcopy(original_model.model.norm)
            self.lm_head = copy_module.deepcopy(original_model.lm_head)
            self.original_layers = copy_module.deepcopy(original_model.model.layers)
            # Copy rotary_emb for LLaMA 3 position embeddings
            if hasattr(original_model.model, 'rotary_emb'):
                self.rotary_emb = copy_module.deepcopy(original_model.model.rotary_emb)
            else:
                self.rotary_emb = None
        else:
            # Direct reference (for backward compatibility with single-GPU)
            self.embed_tokens = original_model.model.embed_tokens
            self.norm = original_model.model.norm
            self.lm_head = original_model.lm_head
            self.original_layers = original_model.model.layers
            # Reference rotary_emb
            self.rotary_emb = getattr(original_model.model, 'rotary_emb', None)

        num_original_layers = len(self.original_layers)

        # Decode chromosome to execution path
        self.execution_path = decode_chromosome(chromosome)
        self.num_selected_modules = len(set(self.execution_path))  # Unique modules
        self.effective_depth = len(self.execution_path)

        # Build module lookup dictionary
        # Map module index → DecoupledLlamaLayer
        self.module_dict = nn.ModuleDict()

        for module_idx in set(self.execution_path):
            layer_idx = module_idx // 2
            is_attention = (module_idx % 2 == 0)
            orig_layer = self.original_layers[layer_idx]

            # Safely get modules using named_children()
            children = dict(orig_layer.named_children())

            if is_attention:
                self_attn = children.get('self_attn') or children.get('attention') or children.get('linear_attn')
                input_layernorm = children.get('input_layernorm') or children.get('ln_1')
                module = DecoupledLlamaLayer(
                    'attention',
                    self_attn,
                    input_layernorm
                )
            else:
                mlp = children.get('mlp') or children.get('ffn')
                post_attention_layernorm = children.get('post_attention_layernorm') or children.get('ln_2')
                module = DecoupledLlamaLayer(
                    'ffn',
                    mlp,
                    post_attention_layernorm
                )

            self.module_dict[str(module_idx)] = module

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass through selected modules following execution path (supports loops)."""
        try:
            # Check inputs
            if input_ids is None:
                raise ValueError("input_ids is None")

            hidden_states = self.embed_tokens(input_ids)

            if hidden_states is None:
                raise ValueError("embed_tokens returned None")

            # Generate position_embeddings for LLaMA 3 (if rotary_emb exists)
            position_embeddings = None
            if self.rotary_emb is not None:
                batch_size, seq_length = input_ids.shape
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

            # Pass through modules following execution path (supports loops)
            for module_idx in self.execution_path:
                layer = self.module_dict[str(module_idx)]
                hidden_states = layer(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)
                if hidden_states is None:
                    raise ValueError(f"Module {module_idx} returned None")

            # Final norm and lm_head
            hidden_states = self.norm(hidden_states)
            if hidden_states is None:
                raise ValueError("norm returned None")

            logits = self.lm_head(hidden_states)
            if logits is None:
                raise ValueError("lm_head returned None")

            return type('Output', (), {'logits': logits, 'hidden_states': hidden_states})()

        except Exception as e:
            print(f"Error in DecoupledLlamaModel.forward: {e}")
            print(f"  input_ids shape: {input_ids.shape if input_ids is not None else 'None'}")
            print(f"  num selected modules: {self.num_selected_modules}")
            print(f"  effective depth: {self.effective_depth}")
            raise


class GeneticPruner:
    """Genetic algorithm for module selection pruning."""

    def __init__(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
        n_calib_samples: int,
        calib_seqlen: int,
        population_size: int = 20,
        max_generations: int = 50,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
        crossover_type: str = "uniform",
        selection_method: str = "tournament",
        tournament_size: int = 3,
        top_percent: float = 0.5,
        max_param_ratio: float = 0.5,
        max_loop_count: int = 2,
        use_elite_pool: bool = True,
        elite_seed_pool_path: str = None,
        device: Union[str, List[str]] = "cuda",
        eval_samples: int = 10,
        checkpoint_dir: str = None,
        checkpoint_interval: int = 10,
    ):
        self.original_model = model
        self.calibration_data = calibration_data  # Tensor of shape (1, total_tokens)
        self.n_calib_samples = n_calib_samples
        self.calib_seqlen = calib_seqlen
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.crossover_type = crossover_type
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.top_percent = top_percent
        self.max_param_ratio = max_param_ratio
        self.max_loop_count = max_loop_count
        self.use_elite_pool = use_elite_pool
        self.elite_seed_pool_path = elite_seed_pool_path

        # Parse device parameter to support multi-GPU
        if isinstance(device, str):
            if ',' in device:
                # Multi-GPU format: "cuda:0,1,2,3" or "cuda:0, cuda:1, cuda:2"
                device_str = device.replace('cuda:', '').replace(' ', '')
                gpu_ids = [int(x) for x in device_str.split(',')]
                self.devices = [f"cuda:{gpu_id}" for gpu_id in gpu_ids]
            else:
                # Single GPU: "cuda" or "cuda:0"
                self.devices = [device]
        else:
            # List of devices: ["cuda:0", "cuda:1", ...]
            self.devices = device

        self.device = self.devices[0]  # Primary device for backward compatibility
        self.eval_samples = min(eval_samples, n_calib_samples)  # Don't exceed available samples
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval

        # Lock for thread-safe cache access
        self.cache_lock = threading.Lock()

        self.num_layers = len(model.model.layers)
        self.num_modules = self.num_layers * 2  # Each layer has attention + FFN
        self.original_params = self._count_parameters(model)

        # Calculate parameters for each module
        self.module_params = []
        for layer in model.model.layers:
            # Safely get attention module and norm using named_children()
            children = dict(layer.named_children())

            self_attn = children.get('self_attn') or children.get('attention') or children.get('linear_attn')
            input_layernorm = children.get('input_layernorm') or children.get('ln_1')

            mlp = children.get('mlp') or children.get('ffn')
            post_attention_layernorm = children.get('post_attention_layernorm') or children.get('ln_2')

            # Attention params
            attn_params = sum(p.numel() for p in self_attn.parameters()) if self_attn else 0
            attn_params += sum(p.numel() for p in input_layernorm.parameters()) if input_layernorm else 0
            self.module_params.append(attn_params)

            # FFN params
            ffn_params = sum(p.numel() for p in mlp.parameters()) if mlp else 0
            ffn_params += sum(p.numel() for p in post_attention_layernorm.parameters()) if post_attention_layernorm else 0
            self.module_params.append(ffn_params)

        # Embedding and final norm params (always kept)
        self.fixed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
        self.fixed_params += sum(p.numel() for p in model.model.norm.parameters())
        self.fixed_params += sum(p.numel() for p in model.lm_head.parameters())

        # Check if embeddings are tied to lm_head to avoid double-counting
        if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight') and \
           hasattr(model.model.embed_tokens, 'weight') and \
           id(model.lm_head.weight) == id(model.model.embed_tokens.weight):
            print("  [Note] Detected tied embeddings (lm_head weight == embed_tokens weight)")
            # Subtract embed_tokens parameters since they were counted twice
            self.fixed_params -= sum(p.numel() for p in model.model.embed_tokens.parameters())

        # Cache for evaluated chromosomes
        self.evaluated_cache = {}
        self.historical_elite_pool = []  # Maintain top 10% historical best

        # Create checkpoint directory if needed
        if self.checkpoint_dir:
            from pathlib import Path
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        print(f"Genetic Pruning Initialized:")
        print(f"  Original parameters: {self.original_params / 1e9:.2f}B")
        print(f"  Number of layers: {self.num_layers}")
        print(f"  Total modules: {self.num_modules} (Attention+FFN per layer)")
        print(f"  Fixed params (embed+norm+lm_head): {self.fixed_params / 1e9:.2f}B")
        print(f"  Flexible params: {sum(self.module_params) / 1e9:.2f}B")
        print(f"  Population size: {population_size}")
        print(f"  Max generations: {max_generations}")
        print(f"  Selection method: {selection_method}")
        print(f"  Crossover type: {crossover_type}")
        print(f"  Crossover rate: {crossover_rate:.1%}")
        print(f"  Mutation rate: {mutation_rate:.1%}")
        print(f"  Target param ratio: ≤{max_param_ratio:.0%}")
        print(f"  Max loop count: {max_loop_count}")
        print(f"  Devices: {self.devices}")

    def _count_parameters(self, model: nn.Module) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in model.parameters())

    def _map_chromosome_proportional(self, source_chrom: List[int], target_len: int) -> List[int]:
        """
        Map chromosome from source length to target length proportionally.

        This preserves the overall structure (front-middle-back pattern) better than truncation.
        Example: 80->64 mapping preserves back region activation that truncation would lose.

        Args:
            source_chrom: Source chromosome (e.g., length 80 for 13B)
            target_len: Target length (e.g., 64 for 7B)

        Returns:
            Mapped chromosome of target_len
        """
        source_len = len(source_chrom)
        if source_len == target_len:
            return source_chrom[:]

        target_chrom = []
        for i in range(target_len):
            # Map target position to source position (proportional)
            source_idx = int(i * source_len / target_len)
            # Clamp to valid range
            source_idx = min(source_idx, source_len - 1)
            target_chrom.append(source_chrom[source_idx])

        return target_chrom


    def _calculate_params_ratio(self, chromosome: List[int]) -> float:
        """Calculate parameter ratio for a chromosome (based on unique modules)."""
        selected_params = sum(
            self.module_params[i] for i, value in enumerate(chromosome) if value > 0
        )
        total_params = self.fixed_params + selected_params
        params_ratio = total_params / self.original_params
        return params_ratio

    def _repair_chromosome(self, chromosome: List[int]) -> List[int]:
        """
        Repair chromosome to satisfy constraints (ternary version).

        If params > max_ratio, randomly reduce module usage until satisfied.
        Ensure at least one module is selected.
        """
        chromosome = chromosome[:]  # Copy

        # Ensure at least first module is selected
        if sum(chromosome) == 0:
            chromosome[0] = 1

        # Reduce params if over limit
        while self._calculate_params_ratio(chromosome) > self.max_param_ratio:
            selected_indices = [i for i, v in enumerate(chromosome) if v > 0]
            if len(selected_indices) <= 1:
                break  # Can't remove more

            # Randomly drop one module (set to 0)
            drop_idx = random.choice(selected_indices)
            chromosome[drop_idx] = 0

        return chromosome

    def _process_elite_seeds(self, elite_pool: List[dict]) -> List[dict]:
        """
        Process elite seeds: clip loop counts, map chromosome length, adapt sparsity.

        Args:
            elite_pool: List of elite seeds with 'chromosome' and 'fitness' fields

        Returns:
            Processed elite pool
        """
        if not elite_pool:
            return []

        # Step 1: Clip chromosome values to current max_loop_count
        clipped_count = 0
        for seed in elite_pool:
            original_max = max(seed['chromosome'])
            if original_max > self.max_loop_count:
                seed['chromosome'] = [min(v, self.max_loop_count) for v in seed['chromosome']]
                clipped_count += 1

        if clipped_count > 0:
            print(f"     [Clip] {clipped_count} seeds adjusted to max_loop_count={self.max_loop_count}")

        # Step 2: Check chromosome length compatibility (handle different model sizes)
        expected_length = self.num_modules
        actual_length = len(elite_pool[0]['chromosome'])

        if actual_length != expected_length:
            print(f"  ⚠️  Elite chromosome length mismatch: {actual_length} → {expected_length}")

            if actual_length > expected_length:
                # Proportional mapping: preserves front-middle-back structure
                print(f"     [Proportional] Mapping {actual_length} -> {expected_length} positions")
                map_count = 0
                for seed in elite_pool:
                    if len(seed['chromosome']) > expected_length:
                        seed['chromosome'] = self._map_chromosome_proportional(
                            seed['chromosome'], expected_length
                        )
                        map_count += 1
                print(f"     [Proportional] Mapped {map_count} seeds (preserves back region)")
            else:
                # Pad: Extend with zeros (rare case, e.g., using 7B elites for 13B)
                print(f"     [Pad] Extending from {actual_length} to {expected_length} with zeros")
                for seed in elite_pool:
                    padding_needed = expected_length - len(seed['chromosome'])
                    seed['chromosome'].extend([0] * padding_needed)

        # Step 3: Check params_ratio compatibility (adapt to different sparsity levels)
        # CRITICAL: After mapping from different model size, ALWAYS check and adapt
        print(f"  🔍 Checking params_ratio compatibility...")

        adapt_count = 0
        valid_count = 0
        for seed in elite_pool:
            original_chrom = seed['chromosome'][:]
            current_ratio = self._calculate_params_ratio(original_chrom)

            if current_ratio <= self.max_param_ratio:
                valid_count += 1
                continue  # Already valid

            # Need to reduce params
            adapt_count += 1
            current_active = sum(1 for v in original_chrom if v > 0)

            # Calculate how many modules to remove
            # Use binary search to find right number of active modules
            target_ratio = self.max_param_ratio * 0.95  # 5% safety margin

            # Try reducing active modules
            active_positions = [(i, v) for i, v in enumerate(original_chrom) if v > 0]

            # Score positions: preserve front/back, remove middle
            scored = []
            for i, v in active_positions:
                # Front and back more important
                if i < int(self.num_modules * 0.2):  # Front 20%
                    importance = 1.0
                elif i >= int(self.num_modules * 0.8):  # Back 20%
                    importance = 0.9
                else:  # Middle 60%
                    importance = 0.3
                scored.append((i, importance, v))

            # Sort by importance (lowest first = remove first)
            scored.sort(key=lambda x: x[1])

            # Remove modules one by one until we meet the constraint
            for pos, imp, val in scored:
                if current_ratio <= target_ratio:
                    break
                original_chrom[pos] = 0
                current_ratio = self._calculate_params_ratio(original_chrom)

            seed['chromosome'] = original_chrom

        if adapt_count > 0:
            print(f"  ⚠️  {adapt_count}/{len(elite_pool)} seeds exceeded max_param_ratio")
            print(f"     [Adapt] Pre-sparsified to meet {self.max_param_ratio:.0%} constraint")
            print(f"     [Valid] {valid_count} seeds already valid")

        # Verify all seeds now meet the constraint
        final_check_failed = 0
        for seed in elite_pool:
            ratio = self._calculate_params_ratio(seed['chromosome'])
            if ratio > self.max_param_ratio:
                final_check_failed += 1

        if final_check_failed > 0:
            print(f"  ❌ CRITICAL: {final_check_failed}/{len(elite_pool)} seeds still exceed constraint after adapt!")
        else:
            print(f"  ✅ All {len(elite_pool)} seeds now meet {self.max_param_ratio:.0%} constraint")

        return elite_pool

    def load_elite_seed_pool(self):
        """Load elite seeds from configured path(s) or default locations.

        Supports:
        - Single file: 'elite_seed_pool_llama_13b_50.json'
        - Multiple files (pattern): 'elite_seed_pool_qwen25_14b_{50-90}.json'
          This will load all files from 50 to 90 and merge them.
        """
        import os
        import re

        # Check if path contains range pattern like {50-90}
        if self.elite_seed_pool_path and '{' in self.elite_seed_pool_path and '-' in self.elite_seed_pool_path:
            # Extract range pattern
            match = re.search(r'\{(\d+)-(\d+)\}', self.elite_seed_pool_path)
            if match:
                start_pct = int(match.group(1))
                end_pct = int(match.group(2))
                base_pattern = self.elite_seed_pool_path.replace(match.group(0), '{}')

                print(f"  📚 Loading multiple elite pools: {start_pct}% to {end_pct}%")

                all_seeds = []
                loaded_files = []

                for pct in range(start_pct, end_pct + 10, 10):  # 50, 60, 70, 80, 90
                    if pct > end_pct:
                        break

                    file_path = base_pattern.format(pct)
                    possible_paths = [
                        file_path,
                        os.path.join(os.path.dirname(__file__), "..", file_path),
                        os.path.join("/mnt/task_runtime/tmp/shwang/ProxSparse-main", file_path),
                    ]

                    for path in possible_paths:
                        if os.path.exists(path):
                            try:
                                with open(path, 'r') as f:
                                    seeds = json.load(f)
                                all_seeds.extend(seeds)
                                loaded_files.append(f"{pct}%({len(seeds)})")
                                print(f"    ✓ Loaded {len(seeds)} seeds from {pct}% pool")
                                break
                            except Exception as e:
                                print(f"    ⚠️  Error loading {path}: {e}")
                                continue

                if not all_seeds:
                    print(f"  ⚠️  No elite pools found for pattern {self.elite_seed_pool_path}")
                    return None

                print(f"  📊 Total collected: {len(all_seeds)} seeds from {', '.join(loaded_files)}")

                # Process all seeds: clip, map, adapt
                elite_pool = self._process_elite_seeds(all_seeds)

                # Evaluate all seeds on target model to get accurate PPL
                print(f"  🔄 Evaluating {len(elite_pool)} seeds on target model (7B)...")
                print(f"     This may take a while (~{len(elite_pool) * self.eval_samples * 3 / 60:.1f} minutes)")

                valid_seeds = []
                for i, seed in enumerate(elite_pool):
                    params_ratio = self._calculate_params_ratio(seed['chromosome'])

                    # Only evaluate valid seeds
                    if params_ratio <= self.max_param_ratio:
                        try:
                            ppl = self._evaluate_ppl(seed['chromosome'])
                            valid_seeds.append({
                                'chromosome': seed['chromosome'],
                                'fitness': ppl,
                                'source': seed.get('source', 'unknown'),
                                'params_ratio': params_ratio
                            })
                            if (i + 1) % 20 == 0:
                                print(f"     Progress: {i+1}/{len(elite_pool)} evaluated")
                        except Exception as e:
                            print(f"     ⚠️ Error evaluating seed {i}: {e}")
                            continue
                    else:
                        print(f"     ⚠️ Seed {i} invalid: params_ratio={params_ratio:.1%} > {self.max_param_ratio:.1%}")

                if len(valid_seeds) < 60:
                    print(f"  ⚠️  Only {len(valid_seeds)} valid seeds after evaluation (expected ≥60)")
                    print(f"     Using all {len(valid_seeds)} seeds")
                    elite_pool = valid_seeds
                else:
                    # Sort by PPL on target model and take top 60
                    valid_seeds.sort(key=lambda x: x['fitness'])
                    elite_pool = valid_seeds[:60]

                print(f"  ✅ Final elite pool: {len(elite_pool)} seeds")
                print(f"     PPL range (7B): {elite_pool[0]['fitness']:.2f} - {elite_pool[-1]['fitness']:.2f}")
                if len(elite_pool) >= 60:
                    avg_ppl = sum(s['fitness'] for s in elite_pool) / len(elite_pool)
                    avg_ratio = sum(s['params_ratio'] for s in elite_pool) / len(elite_pool)
                    print(f"     Avg PPL: {avg_ppl:.2f}")
                    print(f"     Avg params ratio: {avg_ratio:.1%}")

                return elite_pool

        # Original single-file logic
        if self.elite_seed_pool_path:
            possible_paths = [
                self.elite_seed_pool_path,
                os.path.join(os.path.dirname(__file__), "..", self.elite_seed_pool_path),
                os.path.join("/mnt/task_runtime/tmp/shwang/ProxSparse-main", self.elite_seed_pool_path),
            ]
        else:
            # Fallback to old default paths (13B 50% sparsity)
            possible_paths = [
                'elite_seed_pool_llama_13b_50.json',
                '../elite_seed_pool_llama_13b_50.json',
                os.path.join(os.path.dirname(__file__), '../elite_seed_pool_llama_13b_50.json'),
                '/z_data/shwang/ProxSparse-main/elite_seed_pool_llama_13b_50.json',
            ]

        for seed_pool_path in possible_paths:
            if os.path.exists(seed_pool_path):
                try:
                    with open(seed_pool_path, 'r') as f:
                        elite_pool = json.load(f)
                    print(f"  ✅ Loaded {len(elite_pool)} elite seeds from {seed_pool_path}")
                    print(f"     PPL range: {elite_pool[0]['fitness']:.2f} - {elite_pool[-1]['fitness']:.2f}")

                    # Process seeds using unified method
                    elite_pool = self._process_elite_seeds(elite_pool)

                    return elite_pool
                except Exception as e:
                    print(f"  ⚠️  Error loading {seed_pool_path}: {e}")
                    continue

        print(f"  ⚠️  Elite seed pool not found, falling back to old seeds")
        return None

    def initialize_population(self) -> List[Individual]:
        """Initialize population with diverse strategies (reordered by importance)."""
        population = []

        # ========================================================================
        # Special: Add standard transformer baseline for 100% parameter experiments
        # ========================================================================
        if self.max_param_ratio >= 0.99:  # Handle floating point comparison
            baseline_all_ones = [1] * self.num_modules
            population.append(Individual(chromosome=baseline_all_ones))
            print(f"  [Baseline] Added all-ones individual (standard transformer) for 100% params")

        # ========================================================================
        # ========================================================================
        # Dynamic Population Allocation
        # ========================================================================
        # Adjust tier sizes based on whether elite pool is used
        
        if self.use_elite_pool:
            # Standard allocation (with elite)
            num_elite = 60           # Elite-based individuals
            num_pattern = 40         # Pattern-based (backbone-aware)
            num_gradient = 15        # Gradient patterns
            num_layer_aware = 10     # Layer-aware patterns
            # Random fills the rest
            print(f"  [Config] Elite pool: ENABLED")
            print(f"  [Allocation] Elite={num_elite}, Pattern={num_pattern}, Gradient={num_gradient}, Layer={num_layer_aware}")
        else:
            # Rebalanced allocation (without elite)
            # Redistribute 60 elite slots to other tiers by importance
            num_elite = 0
            num_pattern = 60         # +20 (most important after elite)
            num_gradient = 25        # +10 (good exploration)
            num_layer_aware = 20     # +10 (structural priors)
            # Random gets +20 implicitly
            print(f"  [Config] Elite pool: DISABLED")
            print(f"  [Allocation] Pattern={num_pattern}, Gradient={num_gradient}, Layer={num_layer_aware}, Random=rest")
            print(f"  [Note] Using pure exploration - may need more generations")

        # TIER 1: Elite-based (MOST IMPORTANT) - 60 individuals
        # ========================================================================

        elite_pool = self.load_elite_seed_pool() if self.use_elite_pool else None
        num_elite_offspring = 0

        if not self.use_elite_pool:
            print(f"  [Elite] Skipped (use_elite_pool=False)")
            num_elite_offspring = 0
        elif elite_pool:
            # Mixed strategy: 40 original + 10 light mutation + 10 medium mutation = 60 total

            # Top 40: Keep original (best quality, no mutation)
            print(f"  [Elite-Pure] Preserving top 20 seeds without mutation")
            for i in range(min(20, len(elite_pool))):
                original = elite_pool[i]['chromosome'].copy()
                population.append(Individual(chromosome=original))

            # Next 10: Light mutation from top 40 seeds (5%-10% mutation)
            print(f"  [Elite-Light] 20 individuals with light mutation from top 40 seeds")
            top40_pool = elite_pool[:min(40, len(elite_pool))]
            for _ in range(20):
                weights = [1.0 / seed['fitness'] for seed in top40_pool]
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                seed = random.choices(top40_pool, weights=weights, k=1)[0]

                mutation_strength = random.choice([0.05, 0.10])  # Light mutation
                mutated = seed['chromosome'].copy()

                for i in range(len(mutated)):
                    if random.random() < mutation_strength:
                        if mutated[i] == 0:
                            mutated[i] = random.choice([1, 2]) if self.max_loop_count >= 2 else 1
                        elif mutated[i] >= self.max_loop_count:
                            mutated[i] = random.choice([0, self.max_loop_count - 1])
                        else:
                            mutated[i] = random.choice([0, mutated[i] - 1, mutated[i] + 1])

                mutated = self._repair_chromosome(mutated)
                population.append(Individual(chromosome=mutated))

            # Next 10: Medium mutation from all 60 seeds (15%-20% mutation)
            print(f"  [Elite-Medium] 20 individuals with medium mutation from all 60 seeds")
            for _ in range(20):
                weights = [1.0 / seed['fitness'] for seed in elite_pool]
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                seed = random.choices(elite_pool, weights=weights, k=1)[0]

                mutation_strength = random.choice([0.15, 0.20])  # Medium mutation
                mutated = seed['chromosome'].copy()

                for i in range(len(mutated)):
                    if random.random() < mutation_strength:
                        if mutated[i] == 0:
                            mutated[i] = random.choice([1, 2]) if self.max_loop_count >= 2 else 1
                        elif mutated[i] >= self.max_loop_count:
                            mutated[i] = random.choice([0, self.max_loop_count - 1])
                        else:
                            mutated[i] = random.choice([0, mutated[i] - 1, mutated[i] + 1])

                mutated = self._repair_chromosome(mutated)
                population.append(Individual(chromosome=mutated))

            print(f"  [Elite-Total] {len(population)} individuals (40 pure + 10 light + 10 medium)")
        else:
            print("  [Elite] Pool not found")
            for _ in range(5):
                chromosome = [random.choice([0, 1]) for _ in range(self.num_modules)]
                chromosome = self._repair_chromosome(chromosome)
                population.append(Individual(chromosome=chromosome))
            num_elite_offspring = 5

        # ========================================================================
        # TIER 2: Pattern-based (HIGH IMPORTANCE) - 40 individuals
        # ========================================================================

        # Pyramid patterns (BEST - matches 19.31 PPL best individual)
        num_gradient_actual = min(num_gradient, max(int(num_gradient * 0.67), self.population_size // 15))
        gradient_count = 0
        for pattern_type in ['pyramid', 'pyramid', 'increasing', 'decreasing']:
            for _ in range(max(1, num_gradient // 4)):
                if len(population) >= self.population_size:
                    break
                chromosome = []
                for layer_idx in range(self.num_layers):
                    if pattern_type == 'increasing':
                        prob = (layer_idx / self.num_layers) * 0.8 + 0.2
                    elif pattern_type == 'decreasing':
                        prob = (1 - layer_idx / self.num_layers) * 0.8 + 0.2
                    else:
                        dist_from_center = abs(layer_idx - self.num_layers // 2) / (self.num_layers // 2)
                        prob = dist_from_center * 0.6 + 0.3
                    for _ in range(2):
                        if random.random() < prob:
                            chromosome.append(random.choice([1, 2]) if self.max_loop_count >= 2 else 1)
                        else:
                            chromosome.append(0)
                chromosome = self._repair_chromosome(chromosome)
                population.append(Individual(chromosome=chromosome))
                gradient_count += 1
            if len(population) >= self.population_size:
                break
        print(f"  [Pyramid] {gradient_count} gradient patterns")

        # Block patterns (continuous modules like [2,2,2,2,2])
        num_blocks = min(15, max(10, self.population_size // 10))
        block_count = 0
        for _ in range(num_blocks):
            if len(population) >= self.population_size:
                break
            chromosome = []
            i = 0
            while i < self.num_modules:
                block_len = random.randint(2, 8)
                if random.random() < 0.4:
                    value = 0
                elif random.random() < 0.7:
                    value = 1
                else:
                    if self.max_loop_count < 2:
                        value = 1
                    else:
                        value = random.randint(2, self.max_loop_count)
                for _ in range(min(block_len, self.num_modules - i)):
                    chromosome.append(value)
                    i += 1
            chromosome = self._repair_chromosome(chromosome)
            population.append(Individual(chromosome=chromosome))
            block_count += 1
        print(f"  [Block] {block_count} block patterns")

        # Layer-aware (front/back dense, middle sparse)
        num_layer_aware_actual = min(num_layer_aware, max(int(num_layer_aware * 0.5), self.population_size // 20))
        layer_count = 0
        for _ in range(num_layer_aware):
            if len(population) >= self.population_size:
                break
            chromosome = []
            for layer_idx in range(self.num_layers):
                for _ in range(2):
                    # Adaptive thresholds: front 15%, back 15%
                    front_threshold = max(5, int(self.num_layers * 0.15))
                    back_threshold = self.num_layers - max(5, int(self.num_layers * 0.15))
                    if layer_idx < front_threshold:
                        prob = 0.9
                    elif layer_idx >= back_threshold:
                        prob = 0.7
                    else:
                        prob = 0.3
                    if random.random() < prob:
                        if random.random() < 0.7:
                            chromosome.append(1)
                        else:
                            if self.max_loop_count < 2:
                                chromosome.append(1)
                            else:
                                chromosome.append(random.randint(2, self.max_loop_count))
                    else:
                        chromosome.append(0)
            chromosome = self._repair_chromosome(chromosome)
            population.append(Individual(chromosome=chromosome))
            layer_count += 1
        print(f"  [Layer] {layer_count} layer-aware")

        # ========================================================================
        # TIER 3: Diversity (MEDIUM IMPORTANCE)
        # ========================================================================

        density_count = 0
        for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
            if len(population) >= self.population_size:
                break
            chromosome = []
            for _ in range(self.num_modules):
                if random.random() < density:
                    if random.random() < 0.3:
                        if self.max_loop_count < 2:
                            chromosome.append(1)
                        else:
                            chromosome.append(random.randint(2, self.max_loop_count))
                    else:
                        chromosome.append(1)
                else:
                    chromosome.append(0)
            chromosome = self._repair_chromosome(chromosome)
            population.append(Individual(chromosome=chromosome))
            density_count += 1
        if density_count > 0:
            print(f"  [Density] {density_count} random densities")

        attn_count = 0
        for keep_ratio in [0.3, 0.5, 0.7]:
            if len(population) >= self.population_size:
                break
            chromosome = []
            for i in range(self.num_modules):
                if i % 2 == 0:
                    chromosome.append(random.randint(1, self.max_loop_count) if random.random() < 0.3 else 1)
                else:
                    if random.random() < keep_ratio:
                        chromosome.append(random.randint(1, self.max_loop_count) if random.random() < 0.3 else 1)
                    else:
                        chromosome.append(0)
            chromosome = self._repair_chromosome(chromosome)
            population.append(Individual(chromosome=chromosome))
            attn_count += 1
        if attn_count > 0:
            print(f"  [Attn] {attn_count} attention-focused")

        ffn_count = 0
        for keep_ratio in [0.3, 0.5, 0.7]:
            if len(population) >= self.population_size:
                break
            chromosome = []
            for i in range(self.num_modules):
                if i % 2 == 0:
                    if random.random() < keep_ratio:
                        chromosome.append(random.randint(1, self.max_loop_count) if random.random() < 0.3 else 1)
                    else:
                        chromosome.append(0)
                else:
                    chromosome.append(random.randint(1, self.max_loop_count) if random.random() < 0.3 else 1)
            chromosome = self._repair_chromosome(chromosome)
            population.append(Individual(chromosome=chromosome))
            ffn_count += 1
        if ffn_count > 0:
            print(f"  [FFN] {ffn_count} FFN-focused")

        # ========================================================================
        # TIER 4: Basic (LOW IMPORTANCE)
        # ========================================================================

        if len(population) < self.population_size:
            target_modules = int(self.num_modules * 0.5)
            baseline_chromosome = [0] * self.num_modules
            selected_indices = random.sample(range(self.num_modules), target_modules)
            for idx in selected_indices:
                baseline_chromosome[idx] = 1
            baseline_chromosome = self._repair_chromosome(baseline_chromosome)
            population.append(Individual(chromosome=baseline_chromosome))

        kth_count = 0
        for k in [2, 3, 4, 5]:
            if len(population) >= self.population_size:
                break
            chromosome = [1 if i % k == 0 else 0 for i in range(self.num_modules)]
            chromosome = self._repair_chromosome(chromosome)
            population.append(Individual(chromosome=chromosome))
            kth_count += 1

        random_count = 0
        while len(population) < self.population_size:
            chromosome = [random.randint(0, self.max_loop_count) for _ in range(self.num_modules)]
            chromosome = self._repair_chromosome(chromosome)
            population.append(Individual(chromosome=chromosome))
            random_count += 1

        if 1 + kth_count + random_count > 0:
            print(f"  [Basic] {1 + kth_count + random_count} baseline/kth/random")

        print(f"\n[Init] Total: {len(population)} | Elite:{num_elite_offspring} Pattern:{gradient_count+block_count+layer_count} Diversity:{density_count+attn_count+ffn_count} Basic:{1+kth_count+random_count}")
        return population

    def evaluate_fitness(self, individual: Individual) -> Individual:
        """Evaluate fitness (PPL) for an individual."""
        # Check cache (thread-safe)
        chromosome_tuple = tuple(individual.chromosome)
        with self.cache_lock:
            if chromosome_tuple in self.evaluated_cache:
                cached = self.evaluated_cache[chromosome_tuple]
                individual.fitness = cached['fitness']
                individual.params_ratio = cached['params_ratio']
                individual.is_valid = cached['is_valid']
                individual.num_modules = cached['num_modules']
                individual.effective_depth = cached['effective_depth']
                return individual

        # Calculate stats
        params_ratio = self._calculate_params_ratio(individual.chromosome)
        num_modules = sum(1 for v in individual.chromosome if v > 0)  # Count unique modules
        execution_path = decode_chromosome(individual.chromosome)
        effective_depth = len(execution_path)  # Total executions

        individual.params_ratio = params_ratio
        individual.num_modules = num_modules
        individual.effective_depth = effective_depth

        # Check constraint
        if params_ratio > self.max_param_ratio or num_modules == 0:
            individual.is_valid = False
            individual.fitness = float('inf')
        else:
            individual.is_valid = True

            # Evaluate PPL
            try:
                ppl = self._evaluate_ppl(individual.chromosome)
                individual.fitness = ppl
            except Exception as e:
                print(f"    ⚠️  Evaluation failed: {e}")
                individual.fitness = float('inf')
                individual.is_valid = False

        # Cache result (thread-safe)
        with self.cache_lock:
            self.evaluated_cache[chromosome_tuple] = {
                'fitness': individual.fitness,
                'params_ratio': individual.params_ratio,
                'is_valid': individual.is_valid,
                'num_modules': individual.num_modules,
                'effective_depth': individual.effective_depth
            }

        return individual

    def _evaluate_ppl(self, chromosome: List[int]) -> float:
        """Evaluate perplexity for a chromosome (following eval_wanda_pruned_ppl.py style)."""
        # Build model
        try:
            decoupled_model = DecoupledLlamaModel(self.original_model, chromosome)
            decoupled_model = decoupled_model.to(self.device)
            decoupled_model.eval()
        except Exception as e:
            print(f"    Error building DecoupledLlamaModel: {e}")
            raise

        # Evaluate PPL using same method as eval_wanda_pruned_ppl.py
        loss_fct = nn.CrossEntropyLoss()
        nlls = []

        with torch.no_grad():
            for i in range(self.eval_samples):
                try:
                    # Extract batch: (1, seqlen)
                    batch = self.calibration_data[:, (i * self.calib_seqlen):((i + 1) * self.calib_seqlen)].to(self.device)

                    # Forward pass
                    outputs = decoupled_model(batch)

                    # Debug: check outputs type
                    if outputs is None:
                        raise ValueError(f"Model output is None for sample {i}")

                    # Get logits
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        # outputs might be the logits directly
                        logits = outputs

                    if logits is None:
                        raise ValueError(f"Logits is None for sample {i}, outputs type: {type(outputs)}")

                    # Shift for language modeling
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = batch[:, 1:].contiguous()

                    # Compute loss
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                    # Negative log likelihood
                    neg_log_likelihood = loss.float() * self.calib_seqlen
                    nlls.append(neg_log_likelihood)

                except Exception as e:
                    print(f"    Error in evaluation sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

        # Compute perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / (self.eval_samples * self.calib_seqlen))

        # Cleanup
        del decoupled_model
        torch.cuda.empty_cache()

        return ppl.item()


    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Tournament selection."""
        tournament = random.sample(population, self.tournament_size)
        winner = min(tournament, key=lambda ind: ind.fitness if ind.is_valid else float('inf'))
        return winner

    def select_two_different_parents(self, population: List[Individual], percent: float = 0.2) -> Tuple[Individual, Individual]:
        """Select two different parents from top percent of population."""
        # Get valid individuals and sort by fitness
        valid_pop = [ind for ind in population if ind.is_valid]
        if not valid_pop:
            raise ValueError("No valid individuals in population")

        sorted_pop = sorted(valid_pop, key=lambda ind: ind.fitness)

        # Select from top percent
        top_n = max(2, int(len(sorted_pop) * percent))  # At least 2
        top_individuals = sorted_pop[:top_n]

        # Randomly choose two different individuals
        parent1, parent2 = random.sample(top_individuals, 2)
        return parent1, parent2

    def select_weighted_parents(self, population: List[Individual], percent: float = 0.6) -> Tuple[Individual, Individual]:
        """
        Select two different parents from top percent with weighted probability.

        Weighting scheme (odd-number sequence):
        - Rank 1 (best) gets weight = 2n-1 (where n = number of top individuals)
        - Rank 2 gets weight = 2n-3
        - Rank 3 gets weight = 2n-5
        - ...
        - Rank n-1 gets weight = 3
        - Rank n (worst in top%) gets weight = 1

        Total weight sum = 1 + 3 + 5 + ... + (2n-1) = n²
        This gives quadratically higher selection pressure to better individuals.
        """
        # Get valid individuals and sort by fitness (ascending, so best is first)
        valid_pop = [ind for ind in population if ind.is_valid]
        if not valid_pop:
            raise ValueError("No valid individuals in population")

        sorted_pop = sorted(valid_pop, key=lambda ind: ind.fitness)

        # Select from top percent
        top_n = max(2, int(len(sorted_pop) * percent))  # At least 2
        top_individuals = sorted_pop[:top_n]

        # Calculate weights: odd-number sequence [2n-1, 2n-3, 2n-5, ..., 5, 3, 1]
        # Rank 1 (i=0) gets 2n-1, Rank 2 (i=1) gets 2n-3, ..., Rank n (i=n-1) gets 1
        weights = [2 * (top_n - i) - 1 for i in range(top_n)]

        # Normalize to probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        # Select two different parents using weighted sampling
        parent1_idx = random.choices(range(top_n), weights=probabilities, k=1)[0]

        # Select second parent (must be different)
        remaining_indices = list(range(top_n))
        remaining_indices.remove(parent1_idx)
        remaining_weights = [probabilities[i] for i in remaining_indices]
        remaining_weights_sum = sum(remaining_weights)
        remaining_probs = [w / remaining_weights_sum for w in remaining_weights]

        parent2_idx = random.choices(remaining_indices, weights=remaining_probs, k=1)[0]

        return top_individuals[parent1_idx], top_individuals[parent2_idx]

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Crossover with repair.

        Supports multiple crossover types:
        - 'uniform': Uniform crossover (each gene independently 50% from each parent)
        - 'onepoint': Single-point crossover (one random cut point)
        - 'twopoint': Two-point crossover (two random cut points)
        """
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        chrom_len = len(parent1.chromosome)

        if self.crossover_type == 'uniform':
            # Uniform crossover: each gene independently 50% from each parent
            child1_chromosome = []
            child2_chromosome = []

            for gene1, gene2 in zip(parent1.chromosome, parent2.chromosome):
                if random.random() < 0.5:
                    child1_chromosome.append(gene1)
                    child2_chromosome.append(gene2)
                else:
                    child1_chromosome.append(gene2)
                    child2_chromosome.append(gene1)

        elif self.crossover_type == 'onepoint':
            # Single-point crossover
            cut_point = random.randint(1, chrom_len - 1)
            child1_chromosome = parent1.chromosome[:cut_point] + parent2.chromosome[cut_point:]
            child2_chromosome = parent2.chromosome[:cut_point] + parent1.chromosome[cut_point:]

        elif self.crossover_type == 'twopoint':
            # Two-point crossover
            # Ensure point1 < point2
            point1 = random.randint(1, chrom_len - 2)
            point2 = random.randint(point1 + 1, chrom_len - 1)

            # Swap the middle segment
            child1_chromosome = (parent1.chromosome[:point1] +
                                parent2.chromosome[point1:point2] +
                                parent1.chromosome[point2:])
            child2_chromosome = (parent2.chromosome[:point1] +
                                parent1.chromosome[point1:point2] +
                                parent2.chromosome[point2:])
        else:
            raise ValueError(f"Unknown crossover type: {self.crossover_type}. Must be 'uniform', 'onepoint', or 'twopoint'.")

        # Repair
        child1_chromosome = self._repair_chromosome(child1_chromosome)
        child2_chromosome = self._repair_chromosome(child2_chromosome)

        child1 = Individual(chromosome=child1_chromosome)
        child2 = Individual(chromosome=child2_chromosome)

        return child1, child2

    def mutate(self, individual: Individual) -> Individual:
        """
        Multi-value mutation with gradual transitions (loop-aware).

        Mutation rules (gradual transitions only, no jumps > 1):
        - 0 → 1 (can only increase to 1)
        - 1 → 0 or 2 (can go either direction)
        - 2~(max_loop_count-1) → value-1 or value+1
        - max_loop_count → max_loop_count-1 (can only decrease by 1)
        """
        return self.mutate_with_rate(individual, self.mutation_rate)

    def mutate_with_rate(self, individual: Individual, mutation_rate: float) -> Individual:
        """
        Multi-value mutation with custom rate (for adaptive mutation).

        Same rules as mutate(), but with configurable rate.
        """
        mutated = copy.deepcopy(individual)

        for i in range(len(mutated.chromosome)):
            if random.random() < mutation_rate:
                current_value = mutated.chromosome[i]

                if current_value == 0:
                    # 0 can only mutate to 1
                    mutated.chromosome[i] = 1
                elif current_value == 1:
                    # 1 can mutate to 0 or 2 (50/50)
                    mutated.chromosome[i] = random.choice([0, 2])
                elif current_value == self.max_loop_count:
                    # max_loop_count can only decrease to max_loop_count-1
                    mutated.chromosome[i] = self.max_loop_count - 1
                else:
                    # Middle values can increase or decrease by 1
                    mutated.chromosome[i] = random.choice([current_value - 1, current_value + 1])

        # Repair
        mutated.chromosome = self._repair_chromosome(mutated.chromosome)

        return mutated

    def mutate_block_aware(self, individual: Individual) -> Individual:
        """
        Block-aware mutation: mutate consecutive modules together.
        
        Preserves/creates loop block structures like [2,2,2,2,2].
        """
        mutated = copy.deepcopy(individual)
        i = 0

        while i < len(mutated.chromosome):
            if random.random() < self.mutation_rate:
                # Block size: 2-6 consecutive modules
                block_size = random.randint(2, 6)
                block_end = min(i + block_size, len(mutated.chromosome))

                # Choose block value (weighted distribution)
                rand = random.random()
                if rand < 0.4:
                    value = 0  # 40% skip block
                elif rand < 0.7:
                    value = 1  # 30% single execution
                else:
                    if self.max_loop_count < 2:
                        value = 1
                    else:
                        value = random.randint(2, self.max_loop_count)

                # Apply to entire block
                for j in range(i, block_end):
                    mutated.chromosome[j] = value

                i = block_end
            else:
                i += 1

        mutated.chromosome = self._repair_chromosome(mutated.chromosome)
        return mutated

    def mutate_smart(self, individual: Individual, population_avg_fitness: float) -> Individual:
        """
        Performance-aware adaptive mutation.

        - Elite performers (fitness < 0.9*avg): low mutation rate (preserve genes)
        - Average performers: normal mutation rate
        - Poor performers (fitness > 1.1*avg): high mutation rate (explore)
        """
        if not individual.is_valid:
            return self.mutate_with_rate(individual, self.mutation_rate * 2.0)

        # Calculate performance tier
        if population_avg_fitness > 0:
            performance_ratio = individual.fitness / population_avg_fitness
        else:
            performance_ratio = 1.0

        # Adjust mutation rate based on performance
        if performance_ratio < 0.9:
            # Top performers - conservative mutation
            adjusted_rate = self.mutation_rate * 0.5
        elif performance_ratio < 1.1:
            # Average performers - normal mutation
            adjusted_rate = self.mutation_rate
        else:
            # Poor performers - aggressive mutation
            adjusted_rate = self.mutation_rate * 1.5

        return self.mutate_with_rate(individual, adjusted_rate)

    def local_search(self, individual: Individual, max_iterations: int = 5) -> Individual:
        """
        Hill climbing local search on elite individual.
        
        Try +1/-1 modifications at each position, accept only improvements.
        
        Args:
            individual: The individual to optimize
            max_iterations: Maximum number of improvement rounds
            
        Returns:
            Improved individual (or original if no improvement found)
        """
        current_best = copy.deepcopy(individual)

        for iteration in range(max_iterations):
            improved = False

            # Try modifying each position
            for i in range(len(current_best.chromosome)):
                original_value = current_best.chromosome[i]

                # Try increasing (+1)
                if original_value < self.max_loop_count:
                    neighbor = copy.deepcopy(current_best)
                    neighbor.chromosome[i] = original_value + 1
                    neighbor.chromosome = self._repair_chromosome(neighbor.chromosome)
                    neighbor = self.evaluate_fitness(neighbor)

                    if neighbor.is_valid and neighbor.fitness < current_best.fitness:
                        current_best = neighbor
                        improved = True
                        print(f"    [LocalSearch] Improved pos {i}: {original_value}->{original_value+1}, PPL={current_best.fitness:.2f}")
                        break

                # Try decreasing (-1)
                if original_value > 0:
                    neighbor = copy.deepcopy(current_best)
                    neighbor.chromosome[i] = original_value - 1
                    neighbor.chromosome = self._repair_chromosome(neighbor.chromosome)
                    neighbor = self.evaluate_fitness(neighbor)

                    if neighbor.is_valid and neighbor.fitness < current_best.fitness:
                        current_best = neighbor
                        improved = True
                        print(f"    [LocalSearch] Improved pos {i}: {original_value}->{original_value-1}, PPL={current_best.fitness:.2f}")
                        break

            if not improved:
                print(f"    [LocalSearch] Converged after {iteration+1} iterations")
                break

        return current_best

    def simulated_annealing(self, individual: Individual, T_init: float = 10.0, 
                           T_min: float = 0.1, alpha: float = 0.95) -> Individual:
        """
        Simulated annealing for final tuning.
        
        Accept worse solutions with probability exp(-delta/T) to escape local optima.
        
        Args:
            individual: Starting individual
            T_init: Initial temperature
            T_min: Minimum temperature (stopping criterion)
            alpha: Cooling rate (T *= alpha each step)
            
        Returns:
            Best individual found during annealing
        """
        current = copy.deepcopy(individual)
        best = copy.deepcopy(current)
        T = T_init

        iterations = 0
        improvements = 0

        while T > T_min:
            # Random neighbor: mutate one random position
            neighbor = copy.deepcopy(current)
            pos = random.randint(0, len(neighbor.chromosome) - 1)

            # Mutate the position
            if neighbor.chromosome[pos] == 0:
                neighbor.chromosome[pos] = random.choice([1, 2]) if self.max_loop_count >= 2 else 1
            elif neighbor.chromosome[pos] == self.max_loop_count:
                neighbor.chromosome[pos] = random.choice([0, self.max_loop_count - 1])
            else:
                neighbor.chromosome[pos] = random.choice([0, neighbor.chromosome[pos] - 1, neighbor.chromosome[pos] + 1])

            neighbor.chromosome = self._repair_chromosome(neighbor.chromosome)
            neighbor = self.evaluate_fitness(neighbor)

            if neighbor.is_valid:
                delta = neighbor.fitness - current.fitness

                # Accept if better, or with probability exp(-delta/T) if worse
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current = neighbor
                    if current.fitness < best.fitness:
                        best = current
                        improvements += 1
                        print(f"    [SA] T={T:.2f}, New best: {best.fitness:.2f}")

            T *= alpha
            iterations += 1

        print(f"    [SA] Finished: {iterations} iterations, {improvements} improvements")
        return best

    def save_checkpoint(self, generation: int, population: List[Individual], best_ever: Individual):
        """Save checkpoint to resume training."""
        if not self.checkpoint_dir:
            return

        checkpoint = {
            'generation': generation,
            'population': [
                {
                    'chromosome': ind.chromosome,
                    'fitness': ind.fitness,
                    'params_ratio': ind.params_ratio,
                    'is_valid': ind.is_valid,
                    'num_modules': ind.num_modules,
                    'effective_depth': ind.effective_depth
                }
                for ind in population
            ],
            'best_ever': {
                'chromosome': best_ever.chromosome,
                'fitness': best_ever.fitness,
                'params_ratio': best_ever.params_ratio,
                'is_valid': best_ever.is_valid,
                'num_modules': best_ever.num_modules,
                'effective_depth': best_ever.effective_depth
            },
            'evaluated_cache': {
                str(k): v for k, v in self.evaluated_cache.items()
            },
            'config': {
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'max_param_ratio': self.max_param_ratio,
                'num_modules': self.num_modules,
            }
        }

        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_gen{generation}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"  💾 Checkpoint saved: {checkpoint_path}")

        # Also save as latest
        latest_path = f"{self.checkpoint_dir}/checkpoint_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        print(f"\n📂 Loading checkpoint from: {checkpoint_path}")

        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        # Restore population
        population = []
        for ind_data in checkpoint['population']:
            ind = Individual(
                chromosome=ind_data['chromosome'],
                fitness=ind_data['fitness'],
                params_ratio=ind_data['params_ratio'],
                is_valid=ind_data['is_valid'],
                num_modules=ind_data['num_modules'],
                effective_depth=ind_data.get('effective_depth', 0)  # Handle old checkpoints
            )
            population.append(ind)

        # Restore best ever
        best_data = checkpoint['best_ever']
        best_ever = Individual(
            chromosome=best_data['chromosome'],
            fitness=best_data['fitness'],
            params_ratio=best_data['params_ratio'],
            is_valid=best_data['is_valid'],
            num_modules=best_data['num_modules'],
            effective_depth=best_data.get('effective_depth', 0)  # Handle old checkpoints
        )

        # Restore cache
        self.evaluated_cache = {
            eval(k): v for k, v in checkpoint['evaluated_cache'].items()
        }

        generation = checkpoint['generation']

        print(f"✓ Checkpoint loaded:")
        print(f"  Generation: {generation}")
        print(f"  Population size: {len(population)}")
        print(f"  Best fitness: {best_ever.fitness:.2f}")
        print(f"  Best effective depth: {best_ever.effective_depth}")
        print(f"  Cache size: {len(self.evaluated_cache)}")

        return generation, population, best_ever

    def evolve(self, resume_from: str = None) -> Individual:
        """Main genetic algorithm loop with checkpoint support."""
        print("\n" + "="*80)
        print("Starting Genetic Algorithm Evolution")
        print("="*80)

        start_generation = 0

        # Try to resume from checkpoint
        if resume_from:
            start_generation, population, best_ever = self.load_checkpoint(resume_from)
            start_generation += 1  # Start from next generation
            print(f"\n▶️  Resuming from generation {start_generation}")
        else:
            # Initialize
            population = self.initialize_population()

            # Evaluate initial
            print("\nEvaluating initial population...")
            for idx, individual in enumerate(tqdm(population, desc="Initial evaluation")):
                self.evaluate_fitness(individual)
                if idx < 3:
                    print(f"  Individual {idx}: {individual}")

            best_ever = min((ind for ind in population if ind.is_valid),
                           key=lambda x: x.fitness, default=None)

            if best_ever is None:
                raise ValueError("No valid individuals in initial population!")

            print(f"\nInitial best: {best_ever}")

            # Save initial checkpoint
            if self.checkpoint_dir:
                self.save_checkpoint(0, population, best_ever)

        # Evolution
        for generation in range(start_generation, self.max_generations):
            print(f"\n{'='*80}")
            print(f"Generation {generation + 1}/{self.max_generations}")
            print(f"{'='*80}")

            new_population = []

            # Update historical elite pool (maintain top 10% size across all generations)
            elite_pool_size = max(1, int(self.population_size * 0.1))

            # Add current generation's valid individuals to candidate pool
            valid_current = [ind for ind in population if ind.is_valid]
            candidate_pool = self.historical_elite_pool + valid_current

            # Remove duplicates (same chromosome)
            unique_candidates = []
            seen_chromosomes = set()
            for ind in candidate_pool:
                chrom_tuple = tuple(ind.chromosome)
                if chrom_tuple not in seen_chromosomes:
                    unique_candidates.append(ind)
                    seen_chromosomes.add(chrom_tuple)

            # Sort by fitness and keep top elite_pool_size
            unique_candidates.sort(key=lambda x: x.fitness)
            self.historical_elite_pool = [copy.deepcopy(ind) for ind in unique_candidates[:elite_pool_size]]

            # Add all historical elites to new_population
            for ind in self.historical_elite_pool:
                new_population.append(copy.deepcopy(ind))

            print(f"💎 Historical elite pool: {len(self.historical_elite_pool)} individuals")
            if self.historical_elite_pool:
                best_in_pool = self.historical_elite_pool[0]
                worst_in_pool = self.historical_elite_pool[-1]
                print(f"   Best: PPL={best_in_pool.fitness:.2f}, Worst: PPL={worst_in_pool.fitness:.2f}")

            # Update best_ever
            if self.historical_elite_pool:
                current_best = self.historical_elite_pool[0]
                if current_best.fitness < best_ever.fitness:
                    best_ever = copy.deepcopy(current_best)
                    print(f"🎉 New historical best: PPL={best_ever.fitness:.2f}")

            # Adaptive mutation rate for long runs (300+ generations)
            # Early stage (0-33%): High exploration (1.5x base rate)
            # Mid stage (33-66%): Normal exploration (1.0x base rate)
            # Late stage (66-100%): Fine-tuning (0.5x base rate)
            if self.max_generations >= 200:
                progress = generation / self.max_generations
                if progress < 0.33:
                    current_mutation_rate = self.mutation_rate * 1.5
                    phase = "Exploration"
                elif progress < 0.66:
                    current_mutation_rate = self.mutation_rate
                    phase = "Balanced"
                else:
                    current_mutation_rate = self.mutation_rate * 0.5
                    phase = "Fine-tuning"
                print(f"  Phase: {phase}, Mutation rate: {current_mutation_rate:.4f}")
            else:
                current_mutation_rate = self.mutation_rate

            # Generate offspring
            offspring_to_evaluate = []  # Collect offspring for batch evaluation
            # Calculate population average fitness for smart mutation
            valid_individuals = [ind for ind in population if ind.is_valid]
            if valid_individuals:
                population_avg_fitness = np.mean([ind.fitness for ind in valid_individuals])
            else:
                population_avg_fitness = float('inf')


            while len(new_population) < self.population_size:
                # Select parents based on selection method
                if self.selection_method == "top20":
                    parent1, parent2 = self.select_two_different_parents(population, percent=0.2)
                elif self.selection_method == "topNw":
                    parent1, parent2 = self.select_weighted_parents(population, percent=self.top_percent)
                else:  # tournament (default)
                    parent1 = self.tournament_selection(population)
                    parent2 = self.tournament_selection(population)

                child1, child2 = self.crossover(parent1, parent2)
                # Mixed mutation: 30% block-aware, 70% smart mutation
                if random.random() < 0.3:
                    child1 = self.mutate_block_aware(child1)
                    child2 = self.mutate_block_aware(child2)
                else:
                    child1 = self.mutate_smart(child1, population_avg_fitness)
                    child2 = self.mutate_smart(child2, population_avg_fitness)

                new_population.extend([child1, child2])
                offspring_to_evaluate.extend([child1, child2])

            # Trim to population size
            new_population = new_population[:self.population_size]

            # Batch evaluate offspring (multi-GPU or sequential)
            elite_count = 1 if current_best else 0
            offspring_to_evaluate = offspring_to_evaluate[:self.population_size - elite_count]

            for child in offspring_to_evaluate:
                self.evaluate_fitness(child)

            # Update new_population with evaluated offspring
            new_population[elite_count:] = offspring_to_evaluate[:self.population_size - elite_count]

            population = new_population

            # Update best
            if current_best and current_best.fitness < best_ever.fitness:
                best_ever = copy.deepcopy(current_best)
                print(f"🎉 New best found: {best_ever}")

            # Stats
            valid_individuals = [ind for ind in population if ind.is_valid]
            if valid_individuals:
                avg_fitness = np.mean([ind.fitness for ind in valid_individuals])
                print(f"  Valid individuals: {len(valid_individuals)}/{len(population)}")
                print(f"  Average fitness: {avg_fitness:.2f}")

            # Save checkpoint periodically
            if self.checkpoint_dir and (generation + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(generation + 1, population, best_ever)

            # Local search every 10 generations on current best
            if current_best and (generation + 1) % 10 == 0 and generation > 0:
                print(f"  [LocalSearch] Applying to current best (PPL={current_best.fitness:.2f})...")
                improved_best = self.local_search(current_best, max_iterations=5)

                if improved_best.fitness < current_best.fitness:
                    print(f"  [LocalSearch] Success: {current_best.fitness:.2f} -> {improved_best.fitness:.2f}")
                    # Replace in population
                    for idx, ind in enumerate(population):
                        if ind.chromosome == current_best.chromosome:
                            population[idx] = improved_best
                            break
                    
                    # Update best_ever if needed
                    if improved_best.fitness < best_ever.fitness:
                        best_ever = improved_best
                        print(f"  [LocalSearch] Updated best_ever: {best_ever.fitness:.2f}")

        print("\n" + "="*80)
        print("Evolution Complete!")
        print("="*80)
        print(f"Best individual: {best_ever}")

        # Final tuning with Simulated Annealing
        print("" + "="*80)
        print("Applying Simulated Annealing to final best...")
        print("="*80)
        sa_best = self.simulated_annealing(best_ever, T_init=10.0, T_min=0.1, alpha=0.95)
        
        if sa_best.fitness < best_ever.fitness:
            print(f"[SA] Improved final best: {best_ever.fitness:.2f} -> {sa_best.fitness:.2f}")
            best_ever = sa_best
        else:
            print(f"[SA] No improvement (best remains {best_ever.fitness:.2f})")


        # Save final checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint(self.max_generations, population, best_ever)

        return best_ever

    def build_pruned_model(self, best_individual: Individual) -> nn.Module:
        """Build final pruned model with loop support."""
        print("\n" + "="*80)
        print("Building Final Pruned Model (with Loop Support)")
        print("="*80)

        pruned_model = DecoupledLlamaModel(self.original_model, best_individual.chromosome)

        # Decode execution path
        execution_path = decode_chromosome(best_individual.chromosome)

        # Analysis
        print(f"\nModel Analysis:")
        print(f"  Total modules: {self.num_modules}")
        print(f"  Unique modules selected: {best_individual.num_modules}")
        print(f"  Effective depth: {best_individual.effective_depth} (after loop expansion)")
        print(f"  Pruned modules: {self.num_modules - best_individual.num_modules}")
        print(f"  Parameter ratio: {best_individual.params_ratio:.2%}")
        print(f"  Perplexity: {best_individual.fitness:.2f}")

        # Parameter efficiency analysis
        if best_individual.effective_depth > 0:
            efficiency = best_individual.effective_depth / max(best_individual.num_modules, 1)
            print(f"  Parameter efficiency: {efficiency:.2f}x (each module contributes {efficiency:.2f} layers)")

        # Show chromosome encoding (first 20)
        print(f"\nChromosome encoding (first 20): {best_individual.chromosome[:20]}" +
              (" ..." if len(best_individual.chromosome) > 20 else ""))

        # Show execution path
        print(f"\nExecution path (first 30 modules):")
        path_str = []
        for module_idx in execution_path[:30]:
            layer_idx = module_idx // 2
            module_type = 'A' if module_idx % 2 == 0 else 'F'
            path_str.append(f"{module_type}{layer_idx}")

        print(f"  {' → '.join(path_str)}" + (" → ..." if len(execution_path) > 30 else ""))

        # Detect and report loops
        loop_count = sum(1 for v in best_individual.chromosome if v >= 2)
        if loop_count > 0:
            print(f"\n  Loop statistics:")
            print(f"    Modules with loops (value >=2): {loop_count}")
            print(f"    Total executions: {best_individual.effective_depth}")
            print(f"    Unique modules: {best_individual.num_modules}")

        return pruned_model


def prune_model_genetic(
    model: nn.Module,
    calibration_data: torch.Tensor,
    n_calib_samples: int,
    calib_seqlen: int,
    population_size: int = 20,
    max_generations: int = 50,
    mutation_rate: float = 0.05,
    crossover_rate: float = 0.8,
    crossover_type: str = "uniform",
    selection_method: str = "tournament",
    top_percent: float = 0.5,
    max_param_ratio: float = 0.5,
    max_loop_count: int = 2,
    use_elite_pool: bool = True,
        elite_seed_pool_path: str = None,
        device: str = "cuda",
    eval_samples: int = 10,
    checkpoint_dir: str = None,
    checkpoint_interval: int = 10,
    resume_from: str = None,
) -> Tuple[nn.Module, Individual]:
    """
    Main function: Prune model using genetic algorithm.

    Args:
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: Save checkpoint every N generations
        resume_from: Path to checkpoint file to resume from
        top_percent: Top percent for topNw selection (0.0-1.0)

    Returns:
        pruned_model: Model with selected modules
        best_individual: Best individual found
    """
    ga = GeneticPruner(
        model=model,
        calibration_data=calibration_data,
        n_calib_samples=n_calib_samples,
        calib_seqlen=calib_seqlen,
        population_size=population_size,
        max_generations=max_generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        crossover_type=crossover_type,
        selection_method=selection_method,
        top_percent=top_percent,
        max_param_ratio=max_param_ratio,
        max_loop_count=max_loop_count,
        use_elite_pool=use_elite_pool,
        elite_seed_pool_path=elite_seed_pool_path,
        device=device,
        eval_samples=eval_samples,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
    )

    best_individual = ga.evolve(resume_from=resume_from)
    pruned_model = ga.build_pruned_model(best_individual)

    return pruned_model, best_individual
