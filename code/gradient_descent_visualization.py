import numpy as np
import matplotlib.pyplot as plt

# Define the cost function J(θ) = (θ - 3)^2
def cost_function(theta):
    return (theta - 3)**2

# Define the gradient of the cost function: ∇J(θ) = 2(θ - 3)
def gradient(theta):
    return 2 * (theta - 3)

# Initialize parameters
theta = 0  # Starting value for θ
learning_rate = 0.1  # Step size for updates
iterations = 50  # Number of iterations

# Track θ and cost values for visualization
theta_history = [theta]
cost_history = [cost_function(theta)]

# Gradient Descent loop
for _ in range(iterations):
    grad = gradient(theta)  # Compute the gradient
    theta -= learning_rate * grad  # Update θ
    theta_history.append(theta)
    cost_history.append(cost_function(theta))

# Plotting cost function convergence
plt.figure(figsize=(8, 6))
plt.plot(range(iterations + 1), cost_history, marker='o', label="Cost Function")
plt.title("Cost Function Convergence Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.grid()
plt.show()
# Plotting the parameter updates
theta_values = np.linspace(0, 5, 100)  # Range of θ values
cost_values = cost_function(theta_values)  # Corresponding J(θ)

plt.figure(figsize=(8, 6))
plt.plot(theta_values, cost_values, label='Cost Function $J(\\theta)$', color='blue')
for i in range(len(theta_history) - 1):
    plt.arrow(theta_history[i], cost_function(theta_history[i]),
              theta_history[i + 1] - theta_history[i],
              cost_function(theta_history[i + 1]) - cost_function(theta_history[i]),
              color='red', head_width=0.15, head_length=0.1)
plt.scatter(theta_history[0], cost_function(theta_history[0]), color='orange', label='Start (θ=0)')
plt.scatter(3, cost_function(3), color='green', label='Minimum (θ=3)')
plt.title("Visualization of Parameter Updates")
plt.xlabel("$\\theta$")
plt.ylabel("$J(\\theta)$")
plt.legend()
plt.grid()
plt.show()
