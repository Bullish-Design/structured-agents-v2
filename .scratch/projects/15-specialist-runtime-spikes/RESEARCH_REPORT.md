# Specialist Runtime Spike Report

Status: in progress  
Started: 2026-07-21  
Baseline: `main` at `90725a56f28c6a5a09c0a93a31afcb15f3dfa504`

## Objective

Implement and prove the highest-value unresolved project-14 spikes on the two local 12 GB NVIDIA RTX 3060 GPUs:

- corrected real DBOS queue workflow and strict typed result;
- scoped bypass and automated approval composition;
- crash/recovery boundaries;
- vLLM and SGLang XGrammar profiles;
- vLLM and SGLang multi-LoRA scheduling;
- mixed adapter plus distinct-constraint concurrency with engine evidence.

## Safety and mutation boundary

The owner explicitly authorized changing, updating, stopping, or replacing local LLM runners. Existing runner configuration and GPU/process state will be recorded before mutation. GPU profiles will be qualified one at a time on isolated ports or by deliberately stopping the service that owns the selected GPU. No external effect target will be used; crash and authority probes use isolated local fixtures.

Pre-existing project-12, project-13, and project-14 research trees are user work and will be preserved.

## Initial host state

- GPU 0: RTX 3060 12,288 MiB; 8,401 MiB used by llama.cpp profile on port 8001.
- GPU 1: RTX 3060 12,288 MiB; 10,803 MiB used by vLLM profile on port 8000.
- PostgreSQL listens on loopback port 5432.
- SGLang is not listening on port 8002.

## Investigation log

### 2026-07-21 — baseline started

Repository instructions and the project-14 implementation/spike plans were reread. No `AGENTS.md` exists in the checkout. The first implementation target is SP-01 because all batch, authority, and crash probes depend on a correct durable typed workflow boundary.
