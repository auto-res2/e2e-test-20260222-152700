"""Evaluation script to aggregate results and create comparison plots."""

# [VALIDATOR FIX - Attempt 1]
# [PROBLEM]: evaluate.py was being called with Hydra-style arguments (results_dir="..." run_ids='...') but the script used argparse which expects --results_dir and --run_ids
# [CAUSE]: Mismatch between workflow calling convention (Hydra) and script argument parsing (argparse)
# [FIX]: Changed from argparse to Hydra for consistency with other scripts (src/main.py) and to match workflow calling convention
#
# [OLD CODE]:
# import argparse
# def parse_args():
#     parser = argparse.ArgumentParser(description="Evaluate experiment results")
#     parser.add_argument("--results_dir", type=str, required=True, help="Results directory path")
#     parser.add_argument("--run_ids", type=str, required=True, help="JSON string list of run IDs to evaluate")
#     parser.add_argument("--wandb_entity", type=str, default="airas", help="WandB entity")
#     parser.add_argument("--wandb_project", type=str, default="20260222-152700", help="WandB project")
#     return parser.parse_args()
#
# [NEW CODE]:
import json
import os
from pathlib import Path
from typing import Dict, List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb


def fetch_wandb_run(entity: str, project: str, run_id: str) -> Dict:
    """
    Fetch WandB run data by display name.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name
        
    Returns:
        Dictionary with run data
    """
    api = wandb.Api()
    
    # Find run by display name
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": run_id},
        order="-created_at"
    )
    
    if len(runs) == 0:
        print(f"Warning: No WandB run found for {run_id}")
        return None
    
    run = runs[0]  # Most recent run with that name
    
    # Get summary metrics
    summary = dict(run.summary)
    
    # Get config
    config = dict(run.config)
    
    return {
        "run_id": run_id,
        "summary": summary,
        "config": config,
        "url": run.url,
    }


def load_local_metrics(results_dir: Path, run_id: str) -> Dict:
    """
    Load metrics from local results directory.
    
    Args:
        results_dir: Results directory path
        run_id: Run ID
        
    Returns:
        Dictionary with metrics
    """
    metrics_file = results_dir / run_id / "metrics.json"
    
    if not metrics_file.exists():
        print(f"Warning: Metrics file not found for {run_id}: {metrics_file}")
        return None
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    return metrics


def create_comparison_plots(
    all_metrics: Dict[str, Dict],
    output_dir: Path,
) -> List[str]:
    """
    Create comparison plots for all runs.
    
    Args:
        all_metrics: Dictionary mapping run_id to metrics
        output_dir: Output directory for plots
        
    Returns:
        List of generated file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # 1. Accuracy comparison
    fig, ax = plt.subplots()
    
    run_ids = list(all_metrics.keys())
    accuracies = [all_metrics[rid]["accuracy"] for rid in run_ids]
    
    colors = ['#1f77b4' if 'comparative' in rid else '#ff7f0e' for rid in run_ids]
    
    bars = ax.bar(range(len(run_ids)), accuracies, color=colors)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison Across Methods')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    output_file = output_dir / "comparison_accuracy.pdf"
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    generated_files.append(str(output_file))
    print(f"Generated: {output_file}")
    
    # 2. Token usage comparison
    fig, ax = plt.subplots()
    
    avg_tokens = [all_metrics[rid]["avg_tokens_per_sample"] for rid in run_ids]
    
    bars = ax.bar(range(len(run_ids)), avg_tokens, color=colors)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha='right')
    ax.set_ylabel('Average Tokens per Sample')
    ax.set_title('Token Usage Comparison')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.1f}',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    output_file = output_dir / "comparison_tokens.pdf"
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    generated_files.append(str(output_file))
    print(f"Generated: {output_file}")
    
    # 3. Catastrophic error rate comparison
    fig, ax = plt.subplots()
    
    cat_errors = [all_metrics[rid].get("catastrophic_error_rate", 0) for rid in run_ids]
    
    bars = ax.bar(range(len(run_ids)), cat_errors, color=colors)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha='right')
    ax.set_ylabel('Catastrophic Error Rate (>10% off)')
    ax.set_title('Catastrophic Error Rate Comparison')
    ax.set_ylim(0, max(cat_errors) * 1.2 if cat_errors else 1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    output_file = output_dir / "comparison_catastrophic_errors.pdf"
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    generated_files.append(str(output_file))
    print(f"Generated: {output_file}")
    
    # 4. Accuracy vs Token usage scatter
    fig, ax = plt.subplots()
    
    for i, rid in enumerate(run_ids):
        color = colors[i]
        label = 'Baseline' if 'comparative' in rid else 'Proposed'
        marker = 'o' if 'strict' in rid else 's'
        ax.scatter(
            avg_tokens[i],
            accuracies[i],
            c=color,
            s=200,
            marker=marker,
            label=f"{label} ({'strict' if 'strict' in rid else 'long'})",
            alpha=0.7
        )
        ax.annotate(
            rid,
            (avg_tokens[i], accuracies[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    ax.set_xlabel('Average Tokens per Sample')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Token Usage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "comparison_accuracy_vs_tokens.pdf"
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    generated_files.append(str(output_file))
    print(f"Generated: {output_file}")
    
    return generated_files


def create_per_run_plots(
    run_id: str,
    metrics: Dict,
    output_dir: Path,
) -> List[str]:
    """
    Create per-run visualizations.
    
    Args:
        run_id: Run ID
        metrics: Metrics dictionary
        output_dir: Output directory
        
    Returns:
        List of generated file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    # Simple bar chart of key metrics
    fig, ax = plt.subplots()
    
    metric_names = ['Accuracy', 'Catastrophic\nError Rate']
    metric_values = [
        metrics['accuracy'],
        metrics.get('catastrophic_error_rate', 0)
    ]
    
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
    ax.set_ylabel('Rate')
    ax.set_title(f'Key Metrics: {run_id}')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    output_file = output_dir / f"{run_id}_metrics.pdf"
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    generated_files.append(str(output_file))
    print(f"Generated: {output_file}")
    
    return generated_files


# [VALIDATOR FIX - Attempt 2]
# [PROBLEM]: Hydra with config_path=None creates a struct-mode config that rejects CLI overrides like results_dir=... 
#            Error: "Could not override 'results_dir'. Key 'results_dir' is not in struct"
# [CAUSE]: When config_path=None, Hydra creates an empty DictConfig in struct mode by default. 
#          Keys not in the initial (empty) struct cannot be overridden from CLI.
# [FIX]: Use ConfigStore to register a base config with struct mode explicitly disabled.
#        This allows CLI overrides while still providing defaults.
#
# [OLD CODE]:
# @hydra.main(version_base=None, config_path=None)
# def main(cfg: DictConfig):
#     """Main evaluation function."""
#     # Disable struct mode to allow CLI overrides like results_dir=... and run_ids=...
#     OmegaConf.set_struct(cfg, False)
#
# [NEW CODE]:
# Create base config dict with struct mode disabled
base_config = OmegaConf.create({
    "results_dir": ".research/results",
    "run_ids": [],
    "wandb_entity": "airas",
    "wandb_project": "20260222-152700",
})
OmegaConf.set_struct(base_config, False)  # Disable struct mode BEFORE registration

# Register config with struct mode disabled
cs = ConfigStore.instance()
cs.store(name="evaluate_config", node=base_config)

@hydra.main(version_base=None, config_path=None, config_name="evaluate_config")
def main(cfg: DictConfig):
    """Main evaluation function."""
    # struct mode is disabled, CLI overrides work
    
    # Extract parameters from config or use direct CLI overrides
    # Note: We don't use config_path="../config" because evaluate.py doesn't need the run config,
    # and the main config.yaml has run: ??? which is required and would fail
    results_dir = Path(cfg.get("results_dir", ".research/results"))
    
    # run_ids can be a string (JSON) or a list
    run_ids_raw = cfg.get("run_ids")
    if isinstance(run_ids_raw, str):
        run_ids = json.loads(run_ids_raw)
    elif isinstance(run_ids_raw, (list, tuple)):
        run_ids = list(run_ids_raw)
    else:
        raise ValueError(f"run_ids must be a JSON string or list, got: {type(run_ids_raw)}")
    
    # Try to get from wandb config if available, otherwise use defaults
    wandb_entity = cfg.get("wandb_entity", cfg.get("wandb", {}).get("entity", "airas"))
    wandb_project = cfg.get("wandb_project", cfg.get("wandb", {}).get("project", "20260222-152700"))
    
    print(f"Evaluating runs: {run_ids}")
    print(f"Results directory: {results_dir}")
    print()
    
    # Collect metrics for all runs
    all_metrics = {}
    
    for run_id in run_ids:
        print(f"Processing {run_id}...")
        
        # Try to load from local files first
        metrics = load_local_metrics(results_dir, run_id)
        
        if metrics is None:
            # Try to fetch from WandB
            print(f"  Attempting to fetch from WandB...")
            wandb_data = fetch_wandb_run(wandb_entity, wandb_project, run_id)
            if wandb_data:
                metrics = wandb_data["summary"]
        
        if metrics is None:
            print(f"  Warning: Could not find metrics for {run_id}, skipping...")
            continue
        
        all_metrics[run_id] = metrics
        
        # Export per-run metrics
        run_output_dir = results_dir / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = run_output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Exported metrics to: {metrics_file}")
        
        # Create per-run plots
        per_run_files = create_per_run_plots(run_id, metrics, run_output_dir)
        
        print()
    
    if not all_metrics:
        print("Error: No metrics found for any runs")
        return
    
    # Create comparison directory
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Create aggregated metrics
    print("Creating aggregated metrics...")
    
    # Determine primary metric and best runs
    primary_metric = "accuracy"
    metrics_by_run = {
        rid: {k: v for k, v in m.items()}
        for rid, m in all_metrics.items()
    }
    
    proposed_runs = {rid: m for rid, m in all_metrics.items() if "proposed" in rid}
    baseline_runs = {rid: m for rid, m in all_metrics.items() if "comparative" in rid}
    
    best_proposed = None
    best_proposed_value = -float('inf')
    if proposed_runs:
        for rid, m in proposed_runs.items():
            if m[primary_metric] > best_proposed_value:
                best_proposed = rid
                best_proposed_value = m[primary_metric]
    
    best_baseline = None
    best_baseline_value = -float('inf')
    if baseline_runs:
        for rid, m in baseline_runs.items():
            if m[primary_metric] > best_baseline_value:
                best_baseline = rid
                best_baseline_value = m[primary_metric]
    
    gap = None
    if best_proposed and best_baseline:
        gap = best_proposed_value - best_baseline_value
    
    aggregated = {
        "primary_metric": primary_metric,
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
    }
    
    agg_file = comparison_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Exported aggregated metrics to: {agg_file}")
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    comparison_files = create_comparison_plots(all_metrics, comparison_dir)
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {agg_file}")
    for f in comparison_files:
        print(f"  - {f}")
    print()
    
    # Print summary
    print("Summary:")
    print(f"  Primary metric: {primary_metric}")
    if best_proposed:
        print(f"  Best proposed: {best_proposed} ({best_proposed_value:.4f})")
    if best_baseline:
        print(f"  Best baseline: {best_baseline} ({best_baseline_value:.4f})")
    if gap is not None:
        print(f"  Gap: {gap:+.4f}")


if __name__ == "__main__":
    main()
