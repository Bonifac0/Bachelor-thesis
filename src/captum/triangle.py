import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import os
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

test_inp = torch.tensor([[2.0, 2.0, 3.0, 2.0, 2.0, 3.0]])  # shape (1, 6)


# attribution score will be computed with respect to target class
target_class_index = 0

# applying integrated gradients on the SoftmaxModel and input data point
ig = IntegratedGradients(model)
attributions, approximation_error = ig.attribute(
    test_inp, target=target_class_index, return_convergence_delta=True
)


print()
print(attributions)


# ----------------------------
# Integrated Gradients
# ----------------------------
ig = IntegratedGradients(model)
all_attributions = []

for cls in range(4):
    attr, _ = ig.attribute(test_inp, target=cls, return_convergence_delta=True)
    # reshape to (3,2) for the three points
    all_attributions.append(attr.reshape(3, 2).detach().numpy())

# ----------------------------
# Visualization
# ----------------------------
pts = test_inp.reshape(3, 2)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

xlim = [-10, 10]
ylim = [-10, 10]
for cls in range(4):
    ax = axes[cls]
    # compute magnitude of attribution per point for coloring
    magnitudes = (all_attributions[cls]).sum(axis=1)
    sc = ax.scatter(
        pts[:, 0].numpy(),
        pts[:, 1].numpy(),
        s=200,
        c=magnitudes,
        cmap="viridis",
    )
    if cls == 0:
        rect = patches.Rectangle((0, 0), xlim[1], ylim[1], color="yellow", alpha=0.2)
    elif cls == 1:
        rect = patches.Rectangle(
            (xlim[0], 0), -xlim[0], ylim[1], color="yellow", alpha=0.2
        )
    elif cls == 2:
        rect = patches.Rectangle(
            (xlim[0], ylim[0]), -xlim[0], -ylim[0], color="yellow", alpha=0.2
        )
    else:
        rect = patches.Rectangle(
            (0, ylim[0]), xlim[1], -ylim[0], color="yellow", alpha=0.2
        )
    ax.add_patch(rect)
    ax.set_title(f"Class {cls}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.colorbar(sc, ax=ax)

plt.tight_layout()
plt.savefig("test_triangle_importances.png")  # save figure to file
