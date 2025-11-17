import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import os


class ToySoftmaxModel(nn.Module):
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


def generate_triangle():
    # Random points in range [-10, 10]
    pts = torch.randn(3, 2) * 10

    # Flatten
    x = pts.flatten()

    # Compute centroid
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()

    # Determine quadrant
    if cx > 0 and cy > 0:
        label = 0
    elif cx < 0 and cy > 0:
        label = 1
    elif cx < 0 and cy < 0:
        label = 2
    else:
        label = 3

    return x, label


def make_dataset(n):
    xs = []
    labels = []
    for _ in range(n):
        x, label = generate_triangle()
        xs.append(x)
        labels.append(label)
    return torch.stack(xs), torch.tensor(labels)


def train(model):
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X, Y = make_dataset(5000)
    print()
    for epoch in range(200):
        print(f"Epoch {epoch}/200", end="\r")
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out.log(), Y)
        loss.backward()
        optimizer.step()
    print()
    torch.save(model.state_dict(), MODEL_PATH)


MODEL_PATH = "test_tiangle.pth"
# 6 (3*(a,b)), 4 quadrants
model = ToySoftmaxModel(6, 32, 4)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # put model in inference mode
else:
    train(model)

test_inp = torch.tensor([[5.0, 7.0, -6.0, -8.0, 5.0, 4.0]])  # shape (1, 6)
print(model(test_inp))


# attribution score will be computed with respect to target class
target_class_index = 0

# applying integrated gradients on the SoftmaxModel and input data point
ig = IntegratedGradients(model)
attributions, approximation_error = ig.attribute(
    test_inp, target=target_class_index, return_convergence_delta=True
)


print()
print(attributions)
