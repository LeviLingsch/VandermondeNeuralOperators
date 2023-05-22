"""
@author: 
"""

"""
This code will modify the uniform data to the nonequispaced lattice.
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

    def shear_distribution(self, data, center_points, growth, uniform):
        # the data must have the shape 
        if growth == 1:
            return data, torch.arange(self.number_points_x)
        # center points shold be a list in order of where the highest gradients are
        ce_left = self.fixed_simple_ce(growth, center_points[0], uniform, self.number_points_x//2 - 1)[0]
        ce_right = self.number_points_x - ce_left.flip(0) - 1
        
        ce_left[0] = 0
        ce_right[-1] = self.number_points_x - 1
        
        sparse_distribution = torch.cat((ce_left, ce_right))
        sparse_data = torch.index_select(data, -2, sparse_distribution.int())

        return sparse_data, sparse_distribution
