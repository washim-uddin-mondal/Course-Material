import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import random
from collections import deque
import os


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(self.state_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output


def train(args):
    batch_size = args.mini_batch_size

    main_net = DQN(4, 2)
    target_net = DQN(4, 2)

    target_net.load_state_dict(main_net.state_dict())
    optimizer = optim.Adam(main_net.parameters())

    env = args.env
    replay_memory = deque([], maxlen=args.memory_capacity)

    MeanQ = torch.tensor([0.])         # Rolling average
    QVec = deque([], maxlen=args.rolling_window)

    for iter_count in range(args.run):
        curr_state = env.states.clone()
        # It is important to clone, otherwise it'll pass a pointer

        if random.uniform(0, 1) < args.explore_prob:
            curr_phase = torch.randint(0, 2, [1])  # Choose a random phase
        else:
            curr_phase = torch.argmin(main_net(curr_state)).unsqueeze(0)

        env.simulate(curr_phase)
        next_state = env.states.clone()
        cost = torch.sum(next_state)/args.scaling

        QVec.append(torch.sum(next_state))

        if iter_count < args.rolling_window:
            MeanQ += (QVec[-1] - MeanQ) / (iter_count + 1)
        else:
            MeanQ += (QVec[-1] - QVec[0]) / args.rolling_window

        replay_memory.append([curr_state, curr_phase, next_state, cost])

        if batch_size < len(replay_memory):
            batch_samples = random.sample(replay_memory, batch_size)

            batch_curr_states = torch.stack([samples[0] for samples in batch_samples])
            batch_curr_phases = torch.stack([samples[1] for samples in batch_samples])
            batch_next_states = torch.stack([samples[2] for samples in batch_samples])
            batch_costs = torch.stack([samples[3] for samples in batch_samples])

            batch_target = batch_costs + args.gamma*torch.min(target_net(batch_next_states), dim=-1)[0]
            batch_estimate = main_net(batch_curr_states)[range(batch_size), batch_curr_phases[:, 0]]
            error = (batch_target.detach() - batch_estimate).pow(2)
            loss = torch.mean(error)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if iter_count % args.update_cycle == 0:
            target_net.load_state_dict(main_net.state_dict())
            print(f"DQN: Iteration:{iter_count + 1} and mean queue length: {MeanQ}")

    if not os.path.exists('Models'):
        os.mkdir('Models')
    torch.save(main_net.state_dict(), 'Models/DQN.pkl')


def evaluate(args):
    main_net = DQN(4, 2)

    if not os.path.exists('Models/DQN.pkl'):
        raise ValueError('Model does not exist.')
    main_net.load_state_dict(torch.load('Models/DQN.pkl'))

    env = args.env
    MeanQ = 0

    for iter_count in range(args.run_eval):
        curr_state = env.states.clone()
        curr_phase = torch.argmin(main_net(curr_state)).unsqueeze(0)

        env.simulate(curr_phase)
        next_state = env.states.clone()
        MeanQ += (torch.sum(next_state) - MeanQ) / (iter_count + 1)

    print(f"DQN: Mean queue after evaluation is: {MeanQ}")
