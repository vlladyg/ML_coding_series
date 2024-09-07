# Contains group base class and cyclic group implementation
import torch
import numpy as np


class GroupBase(torch.nn.Module):

    def __init__(self, dimension, identity):
        """Implementing a group

        @param dimension: Dimensionality of the group (number of dimensions in the basis of the algebra).
        @param identity: Identity element of the group

        """
        super().__init__()
        self.dimension = dimension
        self.register_buffer('identity', torch.Tensor(identity))


    def elements(self):
        """Obtain a tensor containing all group elements in this group

        """

        raise NotImplementedError()

    def product(self, h, h_prime):
        """ Defines group product on two group elements.

        @param h: group element 1
        @param h_prime: group element 2

        """

        raise NotImplementedError()

    def inverse(self, h):

        """ Defines inverse for group element

        @ param h: A group element from subgroup H

        """

        raise NotADirectoryError()


    def left_action_on_R2(self, h, x):
        """ Group action of an element from the subgroup H on a vector in R2.

        @param h: A group element from subgroup H.
        @param x: Vectors in R2.
        """

        raise NotImplementedError()



    def determinant(self, h):
        """Calculate the determinant of the representation of a group element h.


        @param h: group element
        """

        raise NotImplementedError()

    def normalize_group_parametrization(self, h):
        """ Map the group elements to an interval [-1, 1]. We use this to create
        a standardized input for obtaining weights over the group


        @param h: group element
        """

        raise NotImplementedError()
        
        

        
class CyclicGroup(GroupBase):

    def __init__(self, order):
        """Implementing a group

        @param dimension: Dimensionality of the group (number of dimensions in the basis of the algebra).
        @param identity: Identity element of the group

        """
        super().__init__(dimension = 1, identity = [0.])

        assert order > 1

        self.order = torch.tensor(order)

    def elements(self):
        """Obtain a tensor containing all group elements in this group.


        @ return elements: Tensor containing group elements of shape [self.order]
        """

        return torch.linspace(start = 0,
                              end = 2 * np.pi * float(self.order - 1) / float(self.order),
                              steps = self.order,
                              device = self.identity.device
                              )


    def product(self, h, h_prime):
        """ Defines group product on two group elements of the cyclic group C4.

        @param h: group element 1
        @param h_prime: group element 2

        @ returns product: Tensor containing h \cdot h_prime with \cdot the group action
        """

        product = torch.remainder(h + h_prime, 2 * np.pi)

        return product


    def inverse(self, h):

        """ Defines inverse for group element of the cyclic group C4

        @ param h: A group element

        @ returns inverse: Tensor containing h^{-1}
        """

        inverse = torch.remainder(-h, 2 * np.pi)

        return inverse


    def left_action_on_R2(self, h, x):
        """ Group action of an element from the subgroup H on a vector in R2.

        @param h: A group element from subgroup H.
        @param x: Vectors in R2.

        @ returns transformed_x: Tensor containing \rho(h) x.
        """

        # Transform the vector x with h, recall that we are working with a left-regular representation,
        # meaning we transform vectors in R^2 through left-matrix multiplication
        transformed_x = torch.tensordot(self.matrix_representation(h), x, dims = 1)
        return transformed_x

    def matrix_representation(self, h):
        """ Obtain a matrix representation in R^2 for an element h.

        @ param h: A group element.

        @returns representation: Tensor containing matrix representation of h, shape [2, 2]
        """
        cos_t = torch.cos(h)
        sin_t = torch.sin(h)


        representation = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]],
                                      device = self.identity.device)

        return representation



    def normalize_group_elements(self, h):
        """ Map the group elements to an interval [-1, 1]. We use this to create
        a standardized input for obtaining weights over the group


        @param h: group element

        @return normalized_h: Tensor containing normalized value corresponding to element h
        """
        largest_elem = 2 * np.pi * (self.order - 1) / self.order
        normalized_h = (2*h / largest_elem) - 1

        return normalized_h