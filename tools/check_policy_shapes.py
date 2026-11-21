import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from NeurNetworks import ActorCritic


def main():
    device = "cuda:0"
    policy = ActorCritic().to(device)
    obs = {
        "depth": torch.zeros(16, 1, 64, 64, device=device),
        "state": torch.zeros(16, 8, device=device),
    }
    actions, log_probs, values = policy.act(obs)
    print("actions", tuple(actions.shape))
    print("log_probs", tuple(log_probs.shape))
    print("values", tuple(values.shape))


if __name__ == "__main__":
    main()
