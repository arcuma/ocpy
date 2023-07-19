# ocpy

## Overview
Optimal control problem (OCP) solver implemented in Python.
- DDP (Differential Dynamic Programming)
- iLQR (iterative Linear Quadratic Regulator)

are currently implemented.

## Requirements
- Python3
  - SymPy
  - NumPy
  - Numba
  - Matplotlib
  - seaborn

## Examples
```txt
lqr.ipynb (while this can be solved easily by DARE)
cartpole.ipynb
hexacopter.ipynb
```
https://github.com/arcuma/ocpy/assets/67198327/b5489ac3-135a-4320-ba95-66a13170ba46

## References
1. [D. H. Jacobson and D. Q. Mayne, Differential Dynamic Programming, Elsevier, 1970.](https://doi.org/10.1016/B978-0-12-012710-8.50010)

1. [Y. Tassa, T. Erez and E. Todorov, Synthesis and stabilization of complex behaviors through online trajectory optimization, 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems,  pp. 4906-4913, 2012.](https://doi.org/10.1109/IROS.2012.6386025)
