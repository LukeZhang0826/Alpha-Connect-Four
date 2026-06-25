"""
Export model_2_ConnectFour.pt to ONNX for use with onnxruntime-web.
Run with: py -3.10 scripts/export_onnx.py
Output: web/public/model.onnx
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np

# onnx installed to short path to work around Windows MAX_PATH limit
import sys as _sys
_sys.path.insert(0, "C:\\op")

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, row_count, column_count, action_size, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * row_count * column_count, action_size)
        )
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * row_count * column_count, 1),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


def main():
    device = torch.device("cpu")
    row_count, column_count, action_size = 6, 7, 7
    num_resBlocks, num_hidden = 9, 128

    model = ResNet(row_count, column_count, action_size, num_resBlocks, num_hidden, device)

    weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_2_ConnectFour.pt")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded weights from {weights_path}")

    dummy_input = torch.zeros(1, 3, row_count, column_count, dtype=torch.float32)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "public")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["state"],
        output_names=["policy", "value"],
        dynamic_axes={
            "state": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX model to {out_path}")

    # Verify shapes
    with torch.no_grad():
        policy, value = model(dummy_input)
    print(f"policy shape: {tuple(policy.shape)}  (expected (1, 7))")
    print(f"value  shape: {tuple(value.shape)}   (expected (1, 1))")
    assert tuple(policy.shape) == (1, 7), "policy shape mismatch"
    assert tuple(value.shape) == (1, 1), "value shape mismatch"
    print("Shape assertions passed. Export successful.")


if __name__ == "__main__":
    main()
