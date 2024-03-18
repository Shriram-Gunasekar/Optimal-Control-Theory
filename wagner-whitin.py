import numpy as np

# Parameters
demand = [10, 20, 15, 30]  # Demand for each period
setup_cost = 100  # Setup cost for production
holding_cost = 2  # Holding cost per unit per period

# Initialize arrays to store optimal costs and order quantities
num_periods = len(demand)
optimal_costs = np.zeros(num_periods)
order_quantities = np.zeros(num_periods, dtype=int)

# Initialize first period costs
optimal_costs[0] = setup_cost  # Setup cost for the first period
order_quantities[0] = demand[0]  # Order quantity is equal to demand in the first period

# Dynamic programming to fill in optimal costs and order quantities
for t in range(1, num_periods):
    # Initialize minimum cost to infinity
    min_cost = float('inf')
    best_order = 0

    # Iterate through all possible orders from previous periods
    for i in range(t + 1):
        # Calculate setup cost for the current period
        if i == 0:
            setup = setup_cost
        else:
            setup = 0

        # Calculate total cost for ordering quantity i in the current period
        holding = holding_cost * (sum(demand[i:t+1]) - i)
        total_cost = setup + holding + optimal_costs[i - 1] if i > 0 else holding + optimal_costs[i]

        # Update minimum cost and best order quantity
        if total_cost < min_cost:
            min_cost = total_cost
            best_order = i

    # Store optimal cost and order quantity for current period
    optimal_costs[t] = min_cost
    order_quantities[t] = best_order

# Print optimal production plan and minimum total cost
print("Optimal Production Plan:")
for t in range(num_periods):
    print(f"Period {t + 1}: Produce {order_quantities[t]} units")
print(f"Minimum Total Cost: {optimal_costs[-1]}")
