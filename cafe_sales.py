import numpy as np

units = np.array([[30, 20, 10],   # rows: 4 days
                  [25, 30, 15],   # cols: Tea, Coffee, Cake
                  [40, 25, 20],
                  [35, 15, 25]])
price = np.array([100, 150, 200])

# Revenue of every item on every day (broadcasting)
revenue = units * price

# Revenue per day (sum across items)
revenue_per_day = revenue.sum(axis=1)
print("Revenue per day:", revenue_per_day)

# Units sold per item (sum across days)
units_per_item = units.sum(axis=0)
print("Units sold per item:", units_per_item)

# Best day index (highest total revenue)
best_day = revenue_per_day.argmax()
print("Best day index:", best_day)

# Coffee units for the last two days (column 1, rows 2 and 3)
coffee_last_two = units[2:, 1]
print("Coffee, last two days:", coffee_last_two)

# Total revenue of the whole week (required for the second commit)
total_revenue = revenue.sum()
print("Total revenue of the week:", total_revenue)