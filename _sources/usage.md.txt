# Usage

The `gaussquad` package provides tools to compute Gauss quadrature **nodes and weights** for arbitrary weight functions on finite intervals.

You can use it in two main ways:

- `wquad`: to directly compute the integral of a weighted function.
- `wquad_nodes_weights`: to compute nodes and weights for use in custom integration routines.

Below, we provide examples of both methods. 
You can also find the examples in [src/examples.py](https://github.com/thatmariia/gaussquad/blob/main/src/examples.py).

---

## Basic Example: Integrate a simple function

Let’s integrate $ \int_0^1 w(x) \cdot x^2 dx $ using a constant weight function $ w(x) = 1 $:

```python
import numpy as np
from gaussquad import wquad

result = wquad(
    fn=lambda x: x**2,
    weight_fn=lambda x: np.ones_like(x),
    interval=(0, 1),
    degree=3,
    moment_method="exact"
)

print(f"Integral of x^2 from 0 to 1 with weight function 1: {result}")  # ≈ 0.3333
```

---

## Use custom weight functions

You can integrate under arbitrary weights. For example, you can integrate
$ \int_0^5 w(x) \cdot x^2 dx $
with $ w(x) = \exp(-x) $.

```python
import numpy as np
from gaussquad import wquad

result = wquad(
    fn=lambda x: x**2,
    weight_fn=lambda x: np.exp(-x),
    interval=(0, 5),
    degree=5,
    moment_method="legendre"  # approximation for moments
)

print(f"Integral of x^2 from 0 to 5 with weight function exp(-x): {result}")  # ≈ 1.7507
```

If you want to use the nodes and weights yourself, use:

```python
import numpy as np
from gaussquad import wquad_nodes_weights

nodes, weights = wquad_nodes_weights(
    weight_fn=lambda x: np.exp(-x),
    interval=(0, 5),
    degree=5,
    moment_method="legendre"
)

print("Nodes:", nodes)
print("Weights:", weights)

# Then you can manually compute the integral:

integral = np.sum(weights * nodes**2)
print(f"Integral of x^2 from 0 to 5 with weight function exp(-x): {integral}")  # ≈ 1.7507
```
