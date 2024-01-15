import matplotlib.pyplot as plt

# Create a new figure
fig, ax = plt.subplots()

# Define the circle's properties
circle = plt.Circle((0.5, 0.5), 0.4, color='blue', fill=False)

# Add the circle to the axes
ax.add_artist(circle)

# Set the aspect of the plot to be equal, so the circle isn't skewed
ax.set_aspect('equal')

# Show the plot
plt.show()
