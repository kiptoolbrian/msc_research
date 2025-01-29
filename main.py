import numpy as np
import matplotlib.pyplot as plt
import meshpy.triangle as triangle


def generate_initial_cdt(points, facets, max_area=0.2):
    """ Generate a Constrained Delaunay Triangulation (CDT) mesh for the T-patch."""
    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(facets)
    mesh = triangle.build(info, max_volume=max_area)  # Initial coarse mesh
    return np.array(mesh.points), np.array(mesh.elements)


def advancing_front_refinement(points, elements, min_angle=30):
    """Refines the mesh using Advancing Front Method (AFM) with Steiner points."""
    from scipy.spatial import Delaunay

    points = points.tolist()
    elements = elements.tolist()

    def quality(tri):
        """Calculate aspect ratio based on minimum triangle angle."""
        a, b, c = [np.linalg.norm(points[tri[i]] - points[tri[(i + 1) % 3]]) for i in range(3)]
        angles = np.arccos(np.clip([(b ** 2 + c ** 2 - a ** 2) / (2 * b * c),
                                    (a ** 2 + c ** 2 - b ** 2) / (2 * a * c),
                                    (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)], -1, 1))
        return np.min(np.degrees(angles))

    while True:
        to_refine = [tri for tri in elements if quality(tri) < min_angle]
        if not to_refine:
            break

        new_tris = []
        for tri in to_refine:
            midpoints = [(np.array(points[tri[i]]) + np.array(points[tri[(i + 1) % 3]])) / 2 for i in range(3)]
            new_idx = len(points)
            points.extend(midpoints)

            new_tris.extend([
                [tri[0], new_idx, new_idx + 1],
                [tri[1], new_idx + 1, new_idx + 2],
                [tri[2], new_idx + 2, new_idx]
            ])
            elements.remove(tri)

        elements.extend(new_tris)

        # Re-triangulate to maintain a valid Delaunay structure
        tri = Delaunay(points)
        elements = tri.simplices.tolist()

    return np.array(points), np.array(elements)


def plot_mesh(points, elements, title="Mesh"):
    plt.figure(figsize=(8, 6))
    for tri in elements:
        pts = points[tri]
        plt.fill(pts[:, 0], pts[:, 1], edgecolor='blue', fill=False)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plot_inner_t(inner_t):
    """Plots the profile of the inner T-shape."""
    plt.figure(figsize=(6, 6))
    plt.fill(inner_t[:, 0], inner_t[:, 1], color='red', alpha=0.6, edgecolor='red')
    plt.title("Inner T-Shape Profile")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(min(inner_t[:, 0]) - 1, max(inner_t[:, 0]) + 1)
    plt.ylim(min(inner_t[:, 1]) - 1, max(inner_t[:, 1]) + 1)
    plt.grid(True)
    plt.show()


# Define T-shaped patch geometry
outer_rectangle = np.array([[0, 0], [10, 0], [10, 8], [0, 8]])
inner_t = np.array([[1, 4], [1, 7], [9, 7], [9, 4], [6, 4], [6, 0], [3.5, 0], [3.5, 4], [2, 4]])

# Combine points and define facets (ensuring closed loops)
points = np.vstack((outer_rectangle, inner_t))
facets = [[i, (i + 1) % len(outer_rectangle)] for i in range(len(outer_rectangle))] + \
         [[len(outer_rectangle) + i, len(outer_rectangle) + (i + 1) % len(inner_t)] for i in range(len(inner_t))]

# Generate Initial CDT Mesh
initial_points, initial_elements = generate_initial_cdt(points, facets)
plot_mesh(initial_points, initial_elements, title="Initial CDT Mesh")
# Call the function to visualize the inner T-shape
plot_inner_t(inner_t)

# Apply AFM Refinement
refined_points, refined_elements = advancing_front_refinement(initial_points, initial_elements)
plot_mesh(refined_points, refined_elements, title="Refined Mesh with AFM")

