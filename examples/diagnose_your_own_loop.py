"""Diagnose a training loop that logs to nothing but a Python list.

Run: python examples/diagnose_your_own_loop.py
"""

from rldoctor import diagnose, run_from_records

# Whatever your loop already builds each step. Keys can be your framework's
# names (verl's `actor/entropy`, TRL's `frac_reward_zero_std`, ...) or your own;
# rldoctor resolves what it recognises and reports what it does not.
history = [
    {
        "step": step,
        "reward": 0.2 + 0.4 * (1 - 2.718 ** (-step / 60)),
        "actor/entropy": 0.6 * 2.718 ** (-step / 45) + 0.01,
        "actor/pg_clipfrac": 0.04,
        "response_length/mean": 400 + step,
        "grad_norm": 0.5,
        "perf/time_per_step": 38.0,
    }
    for step in range(300)
]

run = run_from_records(history, config={"algorithm": "grpo", "rollout": {"n": 8}, "num_gpus": 8})
result = diagnose(run)

print(result.headline)
print()
result.print()

# Machine-readable, for a dashboard or an alerting hook.
worst = result.worst
print(f"\nworst severity: {worst.name}  |  exit code for CI: {result.exit_code()}")
