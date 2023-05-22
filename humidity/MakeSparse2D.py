"""
@author: 
"""

"""
This class modifies the uniform data to a sparse grid.
"""
import torch
import numpy as np
import pdb
from scipy import interpolate

class MakeSparse2D:
    # this class handles sparse distributions for 2d and the sphere projected to a cartesian grid
    def __init__(self, number_points_x, number_points_y):
        # the data must be equispaced
        self.number_points_x = number_points_x
        self.number_points_y = number_points_y

    def fixed_simple_ce(self, growth, center, uniform, number_points):
        if uniform > 1:
            # define the sides of the uniform region
            left_side = center - uniform//2
            right_side = center + uniform//2
            
            # define the number of points beyond each side of the uniform region
            number_left = np.floor(left_side**(1/growth))+1
            number_right = np.floor((number_points - right_side)**(1/growth))+1

            # define the positions of points to each side
            points_left = torch.flip(left_side - torch.round(torch.pow(torch.arange(number_left), growth)), [0])
            points_right = right_side + torch.round(torch.pow(torch.arange(number_right), growth))

            uniform_region = torch.arange(left_side+1, right_side, dtype=torch.float)
            con_exp = torch.cat((points_left, uniform_region, points_right))
  

        elif uniform == 0:
            # not necessarily symmetric
            # define the number of points beyond each side of the uniform region
            number_left = np.floor(center**(1/growth))+1
            number_right = np.floor((number_points - center)**(1/growth)) + 1

            # define the positions of points to each side
            points_left = torch.flip(center - torch.round(torch.pow(torch.arange(number_left), growth)), [0])
            points_right = center + torch.round(torch.pow(torch.arange(number_right), growth)) - 1
            
            con_exp = torch.cat((points_left, points_right[2:]))
        return con_exp, number_left

    def generate_ce_distribution(self, growth_x, growth_y, center_x, center_y, uniform_x, uniform_y):
        con_exp_x, number_left = self.fixed_simple_ce(growth_x, center_x, uniform_x, self.number_points_x)
        con_exp_y, number_bottom = self.fixed_simple_ce(growth_y, center_y, uniform_y, self.number_points_y)

        return con_exp_x.int(), con_exp_y.int(), int(number_left), int(number_bottom)

    def get_sparse_data(self, data, x, y):
        return torch.index_select(torch.index_select(data, 1, y), 2, x)