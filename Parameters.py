import torch
from Environment import SingleIntersection


class Parameters:
    def __init__(self):
        self.arr_rates = torch.tensor([2., 3., 3., 2.])
        self.dept_rates = torch.tensor([10., 10., 10., 10.])
        self.env = SingleIntersection(self.arr_rates, self.dept_rates)
        self.run = 10 ** 4
        self.run_eval = 10 ** 4
        self.gamma = 0.99
        self.update_cycle = 100
        self.memory_capacity = 10 ** 6
        self.mini_batch_size = 10
        self.explore_prob = 0.05
        self.hidden_size = 32
        self.scaling = 100
        self.rolling_window = 10 ** 4
