"""
Phase 1 Orchestrator Scripts for ImageTrust v2.0.

Resource-aware data generation and feature extraction pipeline.

Scripts:
--------
run_synthetic_generation.py
    Resource-aware synthetic data generation (social media + screenshots).
    Uses multiprocessing with configurable CPU workers.
    Supports checkpoint/resume for long-running jobs.

run_embedding_extraction.py
    GPU-optimized embedding extraction (ResNet-50, EfficientNet, ViT).
    Supports large batch sizes (256), AMP, and NIQE quality scoring.
    Outputs compressed .npz shards with checkpoint support.

run_phase1_pipeline.py
    Complete Phase 1 orchestrator combining both phases.
    Supports process priority control for background execution.

Launchers:
----------
launch_phase1_windows.bat
    Windows batch file launcher with priority control.

launch_phase1_windows.ps1
    Windows PowerShell launcher with advanced options.

launch_phase1_linux.sh
    Linux/macOS bash launcher with nice priority support.
    Supports background execution with nohup.

Usage:
------
Windows (low priority):
    start /LOW python scripts/orchestrator/run_phase1_pipeline.py ...

Windows (PowerShell):
    .\\scripts\\orchestrator\\launch_phase1_windows.ps1 -Priority Low

Linux (low priority):
    nice -n 10 python scripts/orchestrator/run_phase1_pipeline.py ...

Linux (background):
    ./scripts/orchestrator/launch_phase1_linux.sh --priority low --background

Hardware Target:
    - GPU: RTX 5080 (16GB VRAM), batch_size=256, 85% memory limit
    - CPU: AMD 7800X3D (8 cores), 6 workers for synthetic generation
"""
