"""Main orchestration script for running experiments."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.inference import run_inference, run_sanity_validation


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for running experiments.
    
    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print(f"Running experiment: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print("=" * 80)
    print()
    
    # Print config
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print()
    
    # Apply mode-specific overrides
    if cfg.mode == "sanity_check":
        print("Applying sanity_check mode overrides...")
        # Reduce samples for sanity check
        if cfg.run.dataset.num_samples > 10:
            OmegaConf.update(cfg, "run.dataset.num_samples", 10, merge=False)
        print(f"  - Reduced samples to {cfg.run.dataset.num_samples}")
        print()
    
    # Determine task type and run appropriate script
    task_type = cfg.run.inference.task_type
    
    if task_type == "math_reasoning":
        # This is an inference task
        print(f"Running inference task: {task_type}")
        print()
        
        metrics = run_inference(cfg)
        
        # Run sanity validation if in sanity_check mode
        if cfg.mode == "sanity_check":
            print()
            print("=" * 80)
            print("Running sanity validation...")
            print("=" * 80)
            run_sanity_validation(metrics, cfg)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    print()
    print("=" * 80)
    print("Experiment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
