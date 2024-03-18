import numpy as np

# Parameters
demand_rate = 20  # units per day
unit_cost = 1  # cost per unit
holding_cost = 0.5  # holding cost per unit per day
perishability_cost = 2  # cost per unit per day
days = 30  # planning horizon

# Initialize arrays to store optimal costs and ordering quantities
optimal_costs = np.zeros((days + 1, demand_rate + 1))
order_quantities = np.zeros((days + 1, demand_rate + 1), dtype=int)

# Dynamic programming to fill in optimal costs and order quantities
for t in range(1, days + 1):
    for d in range(1, demand_rate + 1):
        min_cost = float('inf')
        best_order = 0
        for q in range(1, d + 1):
            # Calculate total cost for ordering quantity q
            ordering_cost = q * unit_cost
            holding_cost_total = q * holding_cost * (d - q + 1) / 2
            perishability_cost_total = q * perishability_cost * (days - t + 1)
            total_cost = ordering_cost + holding_cost_total + perishability_cost_total + optimal_costs[t - 1][d - q + 1]

            # Update minimum cost and best order quantity
            if total_cost < min_cost:
                min_cost = total_cost
                best_order = q

        # Store optimal cost and order quantity for current time period and demand level
        optimal_costs[t][d] = min_cost
        order_quantities[t][d] = best_order

# Find optimal policy and minimum cost
optimal_policy = []
t = days
d = demand_rate
min_cost = optimal_costs[days][demand_rate]
while t > 0 and d > 0:
    optimal_policy.append((t, order_quantities[t][d]))
    d -= order_quantities[t][d]
    t -= 1

# Print optimal policy and minimum cost
print("Optimal Policy:")
for t, q in reversed(optimal_policy):
    print(f"Day {days - t + 1}: Order {q} units")
print(f"Minimum Cost: {min_cost}")
