"""
A simple baseline policy for comparison.
"""
import torch


def evaluate(args):
    env = args.env
    MeanQ = 0

    for iter_count in range(args.run_eval):
        curr_phase = torch.randint(0, 2, [1])
        env.simulate(curr_phase)
        next_state = env.states.clone()
        MeanQ += (torch.sum(next_state) - MeanQ) / (iter_count + 1)

    print(f"Baseline: Mean queue length after evaluation is: {MeanQ}")
