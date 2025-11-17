import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients


class ToyModel(nn.Module):
    r"""
    Example toy model from the original paper (page 10)

    https://arxiv.org/pdf/1703.01365.pdf


    f(x1, x2) = RELU(ReLU(x1) - 1 - ReLU(x2))
    """

    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        relu_out1 = F.relu(input1)
        relu_out2 = F.relu(input2)
        return F.relu(relu_out1 - 1 - relu_out2)


def rl():
    model = ToyModel()

    # defining model input tensors
    input1 = torch.tensor([3.0], requires_grad=True)
    input2 = torch.tensor([1.0], requires_grad=True)

    # defining baselines for each input tensor
    baseline1 = torch.tensor([0.0])
    baseline2 = torch.tensor([0.0])

    # defining and applying integrated gradients on ToyModel and the
    ig = IntegratedGradients(model)
    attributions, approximation_error = ig.attribute(
        (input1, input2),
        baselines=(baseline1, baseline2),
        method="gausslegendre",
        return_convergence_delta=True,
    )
    print()
    print(attributions)
    print()
    print(approximation_error)


class ToySoftmaxModel(nn.Module):
    r"""
    Model architecture from:

    https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
    """

    def __init__(self, num_in, num_hidden, num_out):
        super().__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_out = num_out
        self.lin1 = nn.Linear(num_in, num_hidden)
        self.lin2 = nn.Linear(num_hidden, num_hidden)
        self.lin3 = nn.Linear(num_hidden, num_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        lin1 = F.relu(self.lin1(input))
        lin2 = F.relu(self.lin2(lin1))
        lin3 = self.lin3(lin2)
        return self.softmax(lin3)


def classifi():
    num_in = 12  # input dim
    # basicaly range but for tensors
    input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)
    print(input)
    # 10-class classification model
    model = ToySoftmaxModel(num_in, 20, 6)

    # attribution score will be computed with respect to target class
    target_class_index = 1

    # applying integrated gradients on the SoftmaxModel and input data point
    ig = IntegratedGradients(model)
    attributions, approximation_error = ig.attribute(
        input, target=target_class_index, return_convergence_delta=True
    )

    # The input and returned corresponding attribution have the
    # same shape and dimensionality.

    print()
    print(attributions)


# rl()
classifi()
