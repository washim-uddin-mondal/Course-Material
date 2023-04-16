import torch

""" 
Four way intersection. Poisson arrival, poisson departure. 
Straight crossing, no turns.
"""


class SingleIntersection:
    def __init__(self, arr_rates, dept_rates):
        self.arr_rates = arr_rates
        self.dept_rates = dept_rates
        self.states = torch.zeros([4])

    def simulate(self, phase):
        if phase == 0:
            phase_vec = torch.tensor([1, 1, 0, 0])
        else:
            phase_vec = torch.tensor([0, 0, 1, 1])
        curr_arr = torch.poisson(self.arr_rates)
        curr_dept = torch.poisson(self.dept_rates)
        curr_dept = torch.min(curr_dept, self.states)*phase_vec
        self.states += (-curr_dept + curr_arr)
