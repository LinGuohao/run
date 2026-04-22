"""
Main script for Genetic Algorithm-based Pruning.

This script runs genetic algorithm to find optimal Attention-FFN module combinations.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import random
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
import json
from pathlib import Path

from genetic_pruning import prune_model_genetic


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

# ======================== 修改后 ========================
def create_calibration_data(tokenizer, ctx_len, samples, dataset_path=None, seed=42):
    """Create calibration data supporting local path or fallback to WikiText-2."""
    from datasets import load_dataset, load_from_disk
    import os

    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset_path and os.path.exists(dataset_path):
        print(f"Loading local dataset from: {dataset_path}...")
        try:
            # 尝试直接读取本地 huggingface dataset 格式
            dataset = load_dataset(dataset_path, split='train')
        except Exception as e:
            # 如果不是原始脚本格式，尝试作为 load_from_disk 读取
            print(f"Fallback to load_from_disk: {e}")
            dataset = load_from_disk(dataset_path)
            if 'train' in dataset:
                dataset = dataset['train']
    else:
        print(f"Loading WikiText-2 dataset from HuggingFace Hub...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    # Concatenate all text
    text = "\n\n".join(dataset['text'])

    # Tokenize
    print(f"  Tokenizing...")
    trainenc = tokenizer(text, return_tensors='pt')

    # Extract samples
    input_ids = trainenc.input_ids

    if input_ids is None or input_ids.shape[1] == 0:
        raise ValueError("Tokenization failed: input_ids is empty")

    total_tokens = input_ids.shape[1]
    nsamples = total_tokens // ctx_len

    if nsamples == 0:
        raise ValueError(f"Not enough tokens ({total_tokens}) for even one sample of length {ctx_len}")

    # Limit to requested samples
    nsamples = min(nsamples, samples)

    # Trim to fit exact number of samples
    input_ids = input_ids[:, :(nsamples * ctx_len)]

    print(f"✓ Calibration data created: {nsamples} samples of length {ctx_len}")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Total tokens: {input_ids.shape[1]:,}")
    print(f"  Device: {input_ids.device}")
    print(f"  Dtype: {input_ids.dtype}")

    return input_ids, nsamples, ctx_len


def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm-based Pruning")
    parser.add_argument('--model', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='Model name or path')
    parser.add_argument('--population_size', type=int, default=20,
                        help='Population size for genetic algorithm')
    parser.add_argument('--max_generations', type=int, default=50,
                        help='Maximum number of generations')
    parser.add_argument('--mutation_rate', type=float, default=0.05,
                        help='Mutation rate (probability of flipping a bit)')
    parser.add_argument('--crossover_rate', type=float, default=0.8,
                        help='Crossover rate')
    parser.add_argument('--crossover_type', type=str, default='uniform',
                        choices=['uniform', 'onepoint', 'twopoint'],
                        help='Crossover type: uniform, onepoint, or twopoint')
    parser.add_argument('--selection_method', type=str, default='tournament',
                        choices=['tournament', 'top20', 'topNw'],
                        help='Selection method: tournament, top20, or topNw (weighted, use --top_percent to control)')
    parser.add_argument('--top_percent', type=float, default=0.5,
                        help='Top percent for topNw selection (0.0-1.0), e.g., 0.5 for top 50%%, 0.6 for top 60%%')
    parser.add_argument('--tournament_size', type=int, default=3,
                        help='Tournament size for selection')
    parser.add_argument('--max_param_ratio', type=float, default=0.5,
                        help='Maximum parameter ratio')
    parser.add_argument('--max_loop_count', type=int, default=2,
                        help='Maximum loop count for modules')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to local dataset directory')
    parser.add_argument('--use_elite_pool', type=lambda x: x.lower() == 'true', 
                        default=True,
                        help='Whether to use elite seed pool (default: True). Set to False for fresh exploration.')
    parser.add_argument('--elite_seed_pool_path', type=str, default=None,
                        help='Path to elite seed pool JSON file (default: auto-generated from model and sparsity)')
    parser.add_argument('--eval_samples', type=int, default=10,
                        help='Number of samples for PPL evaluation per individual')
    parser.add_argument('--ctx_len', type=int, default=1024,
                        help='Context length for calibration')
    parser.add_argument('--calibration_samples', type=int, default=100,
                        help='Number of calibration samples (will use up to available)')
    parser.add_argument('--cache_dir', type=str, default="/z_data/pretrained",
                        help='Cache directory for model weights')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for pruned model')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save checkpoints (default: output_dir/checkpoints)')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Save checkpoint every N generations')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint file to resume from')

    args = parser.parse_args()

    seed_everything(args.seed)

    # Set output directory
    if args.output_dir is None:
        model_name = args.model.split('/')[-1]
        args.output_dir = f"{model_name}-genetic_pruned_p{args.population_size}_g{args.max_generations}"

    # Set checkpoint directory
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"{args.output_dir}/checkpoints"

    print("="*80)
    print("Genetic Algorithm-based Pruning")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Population size: {args.population_size}")
    print(f"Max generations: {args.max_generations}")
    print(f"Mutation rate: {args.mutation_rate}")
    print(f"Crossover rate: {args.crossover_rate}")
    print(f"Crossover type: {args.crossover_type}")
    print(f"Selection method: {args.selection_method}")
    print(f"Max param ratio: {args.max_param_ratio}")
    print(f"Eval samples per individual: {args.eval_samples}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Checkpoint interval: {args.checkpoint_interval} generations")
    if args.resume_from:
        print(f"Resume from: {args.resume_from}")
    print("="*80)

    # Load model and tokenizer
    print("\\nLoading model and tokenizer...")
    HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Parse device string to get first GPU for model loading
    # Multi-GPU format: "cuda:0,1,2,3" → load on "cuda:0", evaluate on all GPUs
    if isinstance(args.device, str) and ',' in args.device:
        load_device = args.device.split(',')[0].strip()
        if not load_device.startswith('cuda:'):
            load_device = f"cuda:{load_device}"
        print(f"  Multi-GPU mode detected: {args.device}")
        print(f"  Loading model to primary device: {load_device}")
    else:
        load_device = args.device
        print(f"  Loading to device: {load_device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        token=HF_TOKEN
    )

    # Move to primary device only (not the full multi-GPU string)
    print(f"  Moving model to {load_device}...")
    model = model.to(load_device)

    print(f"✓ Model loaded: {args.model}")
    # Count parameters before eval() to avoid potential issues
    total_params = model.num_parameters() if hasattr(model, 'num_parameters') else sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e9:.2f}B")
    print(f"  Model device: {load_device}")

    model.eval()

    # ======================== 修改后 ========================
    # Create calibration data
    print("\nCreating calibration data...")
    calibration_data, n_calib_samples, calib_seqlen = create_calibration_data(
        tokenizer,
        args.ctx_len,
        args.calibration_samples,
        dataset_path=args.dataset_path,  # <--- 加入这一行
        seed=args.seed
    )

    # Run genetic algorithm
    print("\\nStarting Genetic Algorithm Pruning...")
    pruned_model, best_individual = prune_model_genetic(
        model=model,
        calibration_data=calibration_data,
        n_calib_samples=n_calib_samples,
        calib_seqlen=calib_seqlen,
        population_size=args.population_size,
        max_generations=args.max_generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        crossover_type=args.crossover_type,
        selection_method=args.selection_method,
        top_percent=args.top_percent,
        max_param_ratio=args.max_param_ratio,
        max_loop_count=args.max_loop_count,
        use_elite_pool=args.use_elite_pool,
        elite_seed_pool_path=args.elite_seed_pool_path,
        device=args.device,
        eval_samples=args.eval_samples,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
    )

    # Save pruned model
    print("\\n" + "="*80)
    print("Saving Pruned Model")
    print("="*80)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model (DecoupledLlamaModel is not a PreTrainedModel, so save weights only)
    print(f"  Saving model weights...")
    torch.save(pruned_model.state_dict(), str(output_path / "model_weights.pt"))

    # Save config
    import json
    config_dict = {
        "model_type": "DecoupledLlamaModel",
        "base_model": args.model,
        "chromosome": best_individual.chromosome,
        "num_selected_modules": best_individual.num_modules,
        "hidden_size": pruned_model.config.hidden_size,
        "vocab_size": pruned_model.config.vocab_size,
        "num_original_layers": len(pruned_model.original_layers),
    }
    with open(output_path / "model_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    tokenizer.save_pretrained(str(output_path))

    # Save pruning info
    pruning_info = {
        "pruning_method": "Genetic Algorithm",
        "population_size": args.population_size,
        "max_generations": args.max_generations,
        "mutation_rate": args.mutation_rate,
        "crossover_rate": args.crossover_rate,
        "crossover_type": args.crossover_type,
        "selection_method": args.selection_method,
        "max_param_ratio": args.max_param_ratio,
        "best_fitness": best_individual.fitness,
        "best_params_ratio": best_individual.params_ratio,
        "best_num_modules": best_individual.num_modules,
        "best_chromosome": best_individual.chromosome,
        "original_total_params": total_params,
        "pruned_total_params": int(total_params * best_individual.params_ratio),
    }

    with open(output_path / "pruning_info.json", 'w') as f:
        json.dump(pruning_info, f, indent=2)

    print(f"✓ Pruned model saved to: {args.output_dir}")
    print(f"✓ Pruning info saved to: {args.output_dir}/pruning_info.json")

    # Summary
    print("\\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Best chromosome: {best_individual.chromosome[:20]}..." if len(best_individual.chromosome) > 20 else f"Best chromosome: {best_individual.chromosome}")
    print(f"Modules selected: {best_individual.num_modules}/{len(best_individual.chromosome)}")
    print(f"Parameter ratio: {best_individual.params_ratio:.2%}")
    print(f"Perplexity: {best_individual.fitness:.2f}")
    print("="*80)

    print(f"\\nTo evaluate the pruned model, run:")
    print(f"python eval/eval_genetic_pruned_ppl.py --model {args.output_dir}")


if __name__ == "__main__":
    main()
