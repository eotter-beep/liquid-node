# LiquidNode
We have a new mascot, Nodey!

<img width="32" height="32" alt="New Piskel-1 png" src="https://github.com/user-attachments/assets/5ba06263-8a8d-4333-a15a-450629febaee" />

Liquid Node allows you to make servers without the cloud, utilizing Ubuntu

## Memory capsule simulation

The repository now includes `memory_capsule_env.py`, a Gymnasium-based
reinforcement learning environment that simulates ten independent memory
capsules. Each capsule mirrors the memory footprint of a Docker workload, and a
tabular Q-learning agent learns to route incoming memory requests without
overflowing any capsule. Run the module directly to execute a short training
session:

```bash
python memory_capsule_env.py
```

The script prints the recent rewards, a moving-average trend, and an evaluation
score so you can inspect how well the agent learned to balance the workload.
