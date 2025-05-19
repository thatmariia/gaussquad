import numpy as np
from gaussquad import wquad, wquad_nodes_weights


result = wquad(
    fn=lambda x: x**2,
    weight_fn=lambda x: np.ones_like(x),
    interval=(0, 1),
    degree=3,
    moment_method="exact",
)

print(f"Integral of x^2 from 0 to 1 with weight function 1: {result}")  # ≈ 0.3333


result = wquad(
    fn=lambda x: x**2,
    weight_fn=lambda x: np.exp(-x),
    interval=(0, 5),
    degree=5,
    moment_method="legendre",  # approximation for moments
)

print(f"Integral of x^2 from 0 to 5 with weight function exp(-x): {result}")  # ≈ 1.7507


nodes, weights = wquad_nodes_weights(
    weight_fn=lambda x: np.exp(-x),
    interval=(0, 5),
    degree=5,
    moment_method="legendre",  # approximation for moments
)

print("Nodes and weights for the integral of x^2 from 0 to 5 with weight function exp(-x):")
for node, weight in zip(nodes, weights):
    print(f"Node: {node:.10}, Weight: {weight:.10}")

print("Now, let's use these nodes and weights to compute the integral of x^2:")
integral = np.sum(weights * nodes**2)
print(f"Integral of x^2 from 0 to 5 with weight function exp(-x): {integral}")  # ≈ 1.7507
