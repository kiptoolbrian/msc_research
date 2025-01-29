import numpy as np
import matplotlib.pyplot as plt
import meshpy.triangle as triangle


def generate_initial_cdt(points, facets, max_triangles_patch=101, max_triangles_substrate=277):
    """ Generate an initial Constrained Delaunay Triangulation (CDT) mesh with a specific number of triangles."""
    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(facets)
    max_volume = 0.5  # Initial guess for max volume
    iteration = 0

    while iteration < 20:  # Prevent infinite loops
        mesh = triangle.build(info, max_volume=max_volume)
        if not mesh.elements:
            print("Warning: No elements generated! Adjusting max_volume.")
            max_volume *= 0.9
            iteration += 1
            continue

        patch_triangles, substrate_triangles = count_triangles(np.array(mesh.points), np.array(mesh.elements))
        print(
            f"Iteration {iteration}: Patch={patch_triangles}, Substrate={substrate_triangles}, Max Volume={max_volume}")

        if patch_triangles == max_triangles_patch and substrate_triangles == max_triangles_substrate:
            print("Final mesh achieved!")
            break
        max_volume *= 0.9  # Reduce max volume to increase triangle count
        iteration += 1

    return np.array(mesh.points), np.array(mesh.elements, dtype=int)


def count_triangles(points, elements):
    """ Count the number of triangles in the patch and the substrate. """
    patch_count = 0
    substrate_count = 0

    for tri in elements:
        centroid = np.mean(points[tri], axis=0)
        if 1 <= centroid[0] <= 9 and 0 <= centroid[1] <= 7:
            patch_count += 1
        else:
            substrate_count += 1

    return patch_count, substrate_count


def plot_mesh(points, elements, title="Initial CDT Mesh"):
    if elements is None or len(elements) == 0:
        print("Error: No elements found for plotting!")
        return

    print(f"Plotting mesh with {len(elements)} triangles")

    plt.figure(figsize=(8, 6))
    for tri in elements:
        pts = points[tri]
        plt.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], 'k-')
        plt.plot([pts[1, 0], pts[2, 0]], [pts[1, 1], pts[2, 1]], 'k-')
        plt.plot([pts[2, 0], pts[0, 0]], [pts[2, 1], pts[0, 1]], 'k-')

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
points = np.array(domain_points)
facets = [[i, i + 1] for i in range(len(domain_points) - 1)]

# Generate Initial CDT Mesh with specified triangle count
initial_points, initial_elements = generate_initial_cdt(points, facets)

# Ensure mesh elements exist before plotting
if initial_elements is not None and len(initial_elements) > 0:
    plot_mesh(initial_points, initial_elements, title="Initial CDT Mesh")
else:
    print("No valid elements generated for plotting.")

# Count the number of triangles in the patch and substrate
patch_triangles, substrate_triangles = count_triangles(initial_points, initial_elements)
print(f"Number of triangles in Patch: {patch_triangles}")
print(f"Number of triangles in Substrate: {substrate_triangles}")
