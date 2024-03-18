from scipy.integrate import solve_ivp
import numpy as np

# Define the differential equation
def ode(x, y):
    return y[1], x - y[0]

# Define the boundary conditions
a, b = 0, 1
A, B = 0, 2

# Define the shooting method
def shooting_method(guess):
    sol = solve_ivp(ode, [a, b], [A, guess], t_eval=[b])
    return sol.y[0, 0] - B

# Solve using root finding (e.g., bisection method)
guess_low, guess_high = 0, 5
while guess_high - guess_low > 1e-6:
    guess_mid = (guess_low + guess_high) / 2
    if shooting_method(guess_low) * shooting_method(guess_mid) < 0:
        guess_high = guess_mid
    else:
        guess_low = guess_mid

# Get the final solution
sol = solve_ivp(ode, [a, b], [A, guess_high], t_eval=np.linspace(a, b, 100))
print(sol.y[0])
