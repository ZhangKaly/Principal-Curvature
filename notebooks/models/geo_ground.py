import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, Eq, sqrt, ImplicitRegion, Region, random_point
from sympy.calculus.util import function_range
from sympy.geometry import Point3D, Sphere
from sympy.vector import CoordSys3D, ParametricRegion, ParametricRegion3D
from sympy.vector.vector import ImplicitRegion3D
from sympy.vector.point import Point

# Define symbolic variables
x, y, z, R, r = symbols('x y z R r')

# Define the implicit region
implicit_region = ImplicitRegion(
    (R - sqrt(x**2 + y**2))**2 + z**2 - r**2 <= 0, 
    (x, -4, 4), (y, -4, 4), (z, -4, 4)
)

# Discretize the region boundary
boundary_region = implicit_region.boundary
discretized_boundary = boundary_region.discretize(bounding_box=(-4, 4))

# Define the geodesic data
data = []

# Seed for randomization
np.random.seed(123)

# Generate a random point on the region boundary
random_point_on_boundary = random_point(discretized_boundary)

# Generate a random direction vector on the unit sphere
random_direction = Sphere(Point(0, 0, 0), 1).random_point()

# Create a geodesic trajectory
trajectory = []

# Initialize the point and direction
current_point = random_point_on_boundary
current_direction = random_direction
step_size = 0.01

for _ in range(1000):  # Run for a maximum of 1000 steps
    trajectory.append(current_point)
    # Update the point using the direction vector
    current_point += step_size * current_direction
    # Terminate if the point is outside the region
    if not boundary_region.contains(current_point):
        break

# Plot the region boundary and geodesic trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the region boundary
boundary_points = discretized_boundary.points
ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2], c='blue', marker='o', s=5)

# Plot the geodesic trajectory
trajectory_points = np.array([[point[0], point[1], point[2]] for point in trajectory])
ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], color='red')

plt.savefig('torus_geo.png')
