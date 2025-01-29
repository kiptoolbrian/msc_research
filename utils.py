import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import meshpy.triangle as triangle

# Define geometry
outer_rectangle = [
    [0, 0],
    [10, 0],
    [10, 8],
    [0, 8],
    [0, 0]
]

inner_t = [
    [1, 4],
    [1, 7],
    [9, 7],
    [9, 4],
    [6, 4],
    [6, 0],
    [3.5, 0],
    [3.5, 4],
    [2, 4]
]

# Flatten lists for MeshPy
all_points = outer_rectangle + inner_t

# Create facets (connections between points for the boundaries)
outer_facets = [(i, i + 1) for i in range(len(outer_rectangle) - 1)]
inner_facets = [(len(outer_rectangle) + i, len(outer_rectangle) + i + 1) for i in range(len(inner_t) - 1)]
all_facets = outer_facets + inner_facets

# Mesh generation using MeshPy
def mesh_generator():
    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(all_points)
    mesh_info.set_facets(all_facets)

    # Generate mesh (adjust `max_volume` for finer mesh)
    mesh = triangle.build(mesh_info, max_volume= .4)
    return mesh

mesh = mesh_generator()

# Plotting the mesh
def plot_mesh(mesh):
    points = np.array(mesh.points)
    elements = np.array(mesh.elements)

    plt.figure(figsize=(8, 6))
    plt.triplot(points[:, 0], points[:, 1], elements, color='black')
    plt.scatter(points[:, 0], points[:, 1], color='red', s=2)
    plt.title("Meshed Patch Antenna")
    plt.gca().set_aspect("equal")
    plt.show()

plot_mesh(mesh)
