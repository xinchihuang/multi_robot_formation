import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Determine the common range for both axes
# You might want to base this on your data's min and max values
axis_range = [0, 10]  # For example, setting both axes from 0 to 10

# Create figure and axes
fig, ax = plt.subplots()

# Plot data
ax.plot(x, y)

# Customize grid
ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)

# Set the same range for both x and y axes
ax.set_xlim(axis_range)
ax.set_ylim(axis_range)

# Show plot
plt.show()
