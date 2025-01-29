import meshpy.triangle as triangle
import numpy as np
import matplotlib.pyplot as plt

# Antenna Parameters (in meters)
patch_radius = 0.02  # Radius of the patch
substrate_radius = 0.03  # Radius of the substrate


# Generate points along a circle's perimeter
def generate_circle_points(center, radius, num_points=100):
    """Generate points along a circle's perimeter."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = np.column_stack((center[0] + radius * np.cos(angles),
                              center[1] + radius * np.sin(angles)))
    return points


# Generate boundary points for the substrate and patch
substrate_boundary = generate_circle_points(center=(0, 0), radius=substrate_radius, num_points=200)
patch_boundary = generate_circle_points(center=(0, 0), radius=patch_radius, num_points=100)


# Prepare MeshPy input
def create_mesh_info():
    """Set up the MeshPy input for the constrained Delaunay triangulation."""
    info = triangle.MeshInfo()

    # Add substrate boundary points
    info.set_points(np.vstack([substrate_boundary]))

    # Define substrate boundary as facets
    info.set_facets([[i, (i + 1) % len(substrate_boundary)] for i in range(len(substrate_boundary))])

    # Add the patch as a hole
    info.holes.append([0, 0])  # Point inside the patch circle

    return info


# Create the mesh
mesh_info = create_mesh_info()
mesh = triangle.build(mesh_info, max_volume=1e-6)  # max_volume controls triangle size

# Plot the results
plt.figure(figsize=(8, 8))

# Plot the triangles
plt.triplot(
    [v[0] for v in mesh.points],  # X-coordinates
    [v[1] for v in mesh.points],  # Y-coordinates
    mesh.elements,  # Triangle connectivity
    color="blue",
    label="Mesh"
)

# Plot all points
plt.scatter(
    [v[0] for v in mesh.points],
    [v[1] for v in mesh.points],
    color="red",
    s=10,
    label="Mesh Points"
)

# Overlay the patch boundary (red circle)
patch_circle = plt.Circle((0, 0), patch_radius, edgecolor="red", fill=False, linewidth=2, label="Patch Boundary")
plt.gca().add_patch(patch_circle)

# Overlay the substrate boundary (blue circle)
substrate_circle = plt.Circle((0, 0), substrate_radius, edgecolor="blue", fill=False, linewidth=2,
                              label="Substrate Boundary")
plt.gca().add_patch(substrate_circle)

# Adjust plot settings
plt.gca().set_aspect('equal')
plt.xlim(-substrate_radius - 0.01, substrate_radius + 0.01)
plt.ylim(-substrate_radius - 0.01, substrate_radius + 0.01)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Circular Patch Antenna Mesh (MeshPy)")
plt.legend()
plt.grid(True)
plt.show()

# Print the number of triangles
print(f"Number of triangles generated: {len(mesh.elements)}")
