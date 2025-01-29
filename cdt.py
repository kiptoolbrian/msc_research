import numpy as np
import matplotlib.pyplot as plt
import meshpy.triangle as triangle

def generate_initial_cdt(points, facets):
    """ Generate an initial Constrained Delaunay Triangulation (CDT) mesh."""
    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(facets)
    mesh = triangle.build(info, max_volume= 1)  # Initial coarse mesh
    return np.array(mesh.points), np.array(mesh.elements)

def add_random_points(domain, num_points=30):
    """ Generate random points inside the domain to make the mesh unstructured. """
    x_min, x_max = min(p[0] for p in domain), max(p[0] for p in domain)
    y_min, y_max = min(p[1] for p in domain), max(p[1] for p in domain)
    random_points = np.random.uniform([x_min, y_min], [x_max, y_max], (num_points, 2)).tolist()
    return random_points

def plot_mesh(points, elements, title="Mesh"):
    plt.figure(figsize=(8, 6))
    for tri in elements:
        pts = points[tri]
        plt.fill(pts[:,0], pts[:,1], edgecolor='black', fill=False)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Define the T-shaped patch antenna geometry
outer_rectangle = [
    [0, 0], [10, 0], [10, 8], [0, 8], [0, 0]
]

inner_t = [
    [1, 4], [1, 7], [9, 7], [9, 4], [6, 4], [6, 0], [3.5, 0], [3.5, 4], [2, 4]
]

# Combine boundary and inner structure
domain_points = outer_rectangle + inner_t
random_interior_points = add_random_points(domain_points, num_points=50)
points = np.array(domain_points + random_interior_points)
facets = [[i, i+1] for i in range(len(domain_points) - 1)]

# Generate Initial CDT Mesh
initial_points, initial_elements = generate_initial_cdt(points, facets)
plot_mesh(initial_points, initial_elements, title="Unstructured CDT Mesh")
