import numpy as np

# Example 2D array of [x, y] values
arr = np.array([[1, 2], 
                [3, 4], 
                [5, 6], 
                [7, 8], 
                [9, 10]])

# Define the range for x and y
x_min, x_max = 3, 8  # Select x between 3 and 8
y_min, y_max = 2, 9  # Select y between 2 and 9

# Apply the conditions for both x and y
selected_values = arr[(arr[:, 0] >= x_min) & (arr[:, 0] <= x_max) & (arr[:, 1] >= y_min) & (arr[:, 1] <= y_max)]

print(selected_values)