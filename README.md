# ocpy

## Overview
Optimal control problem (OCP) solver implemented in Python.
- DDP (Differential Dynamic Programming)
- iLQR (iterative Linear Quadratic Regulator)

are currently implemented.

You can also perform model predictive control (MPC) using these optimal control solver. Class for MPC is implemented.

## Requirements
- Python3
  - SymPy
  - NumPy
  - SciPy
  - Numba
  - Matplotlib
  - seaborn

## Usage
1. Clone or Download.
   ``` sh
   git clone https://github.com/arcuma/ocpy.git
   ```
   or "Code" >> "Download ZIP" on this page.

1. Install Requirements.
   ``` sh
   pip3 install -r requirements.txt
   ```
1. Run examples below.

## Examples
- OCP
```txt
lqr.ipynb
cartpole.ipynb
hexacopter.ipynb
```
- MPC
``` txt
cartpole_mpc.ipynb
gexacopter_mpc.ipynb
```

https://github.com/arcuma/ocpy/assets/67198327/b5489ac3-135a-4320-ba95-66a13170ba46

https://github.com/arcuma/ocpy/assets/67198327/8829f420-46f0-4b6c-b0a9-57a556cdf8be

## References
1. [D. H. Jacobson and D. Q. Mayne, Differential Dynamic Programming, Elsevier, 1970.](https://doi.org/10.1016/B978-0-12-012710-8.50010-8)

1. [Y. Tassa, T. Erez and E. Todorov, Synthesis and stabilization of complex behaviors through online trajectory optimization, 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems,  pp. 4906-4913, 2012.](https://doi.org/10.1109/IROS.2012.6386025)
