# ocpy

## Overview
Optimal control problem (OCP) solver implemented in Python.
- DDP (Differential Dynamic Programming)
- iLQR (iterative Linear Quadratic Regulator)
- Riccati Recursion

are currently implemented.

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
1. Run examples or formulate your own problem.

## Examples
In /examples/,
- LQR
- cartpole
- hexacopter
- pendubot

![cartpole](https://github.com/arcuma/ocpy/assets/67198327/993a40f6-a61c-47ae-9c83-ff5393b514c7)
![hexacopter](https://github.com/arcuma/ocpy/assets/67198327/5b72074a-f4df-4ff7-abf2-e36f38c094a7)
![pendubot](https://github.com/arcuma/ocpy/assets/67198327/002f2e13-8079-4d83-a208-8cc161f04c55)

## References
1. [D. H. Jacobson and D. Q. Mayne, Differential Dynamic Programming, Elsevier, 1970.](https://doi.org/10.1016/B978-0-12-012710-8.50010-8)

1. [Y. Tassa, T. Erez and E. Todorov, Synthesis and stabilization of complex behaviors through online trajectory optimization, 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems,  pp. 4906-4913, 2012.](https://doi.org/10.1109/IROS.2012.6386025)

1. [S. Katayama, Fast Model Predictive Control of Robotic Systems with Rigid Contacts. Kyoto University, 2022.](https://doi.org/10.14989/doctor.k24266)

1. [J. Nocedal and S.J. Wright, Numerical Optimization (2nd ed.). Springer, 2006.](https://doi.org/10.1007/978-0-387-40065-5)
