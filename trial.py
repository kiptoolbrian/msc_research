import numpy as np
import matplotlib.pyplot as plt
import meshpy.triangle as triangle
from matplotlib.path import Path
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from shapely.geometry import Polygon


def generate_initial_cdt(points_, facets_, max_area=0.2):
    """ Generate a Constrained Delaunay Triangulation (CDT) mesh for the T-patch."""
    info = triangle.MeshInfo()
    info.set_points(points_)
    info.set_facets(facets_)
    mesh = triangle.build(info, max_volume=max_area)  # Initial coarse mesh
    return np.array(mesh.points), np.array(mesh.elements)


def advancing_front_refinement(points_, elements, min_angle=20):
    """Refines the mesh using Advancing Front Method (AFM) with Steiner points."""
    from scipy.spatial import Delaunay

    points_ = points_.tolist()
    elements = elements.tolist()

    def quality(tri):
        """Calculate aspect ratio based on minimum triangle angle."""
        pts = np.array(points_)  # Ensure points is a NumPy array
        a, b, c = [np.linalg.norm(pts[tri[i]] - pts[tri[(i + 1) % 3]]) for i in range(3)]
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
            midpoints = [(np.array(points_[tri[i]]) + np.array(points_[tri[(i + 1) % 3]])) / 2 for i in range(3)]
            new_idx = len(points_)
            points_.extend(midpoints)

            new_tris.extend([
                [tri[0], new_idx, new_idx + 1],
                [tri[1], new_idx + 1, new_idx + 2],
                [tri[2], new_idx + 2, new_idx]
            ])
            elements.remove(tri)

        elements.extend(new_tris)

        # Re-triangulate to maintain a valid Delaunay structure
        tri = Delaunay(points_)
        elements = tri.simplices.tolist()

    return np.array(points_), np.array(elements)


def plot_mesh(points_, elements, title="Mesh"):
    plt.figure(figsize=(8, 6))
    for tri in elements:
        pts = points_[tri]
        plt.fill(pts[:, 0], pts[:, 1], edgecolor='black', fill=False)

    # Superimpose the inner T-shape in red
    plt.plot(inner_t[:, 0], inner_t[:, 1], color='red', linewidth=2)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def plot_inner_t_mesh(points_, elements):
    """Plots only the triangles inside the inner T-shape."""
    plt.figure(figsize=(8, 6))
    # Convert inner_t to a shapely Polygon
    inner_t_polygon = Polygon(inner_t)

    # Expand the boundary outward by 0.2
    expanded_t_polygon = inner_t_polygon.buffer(0.2)

    # Get the new expanded boundary coordinates
    expanded_inner_t = np.array(expanded_t_polygon.exterior.coords)

    # Define a path representing the inner T-shape polygon
    inner_t_path = Path(expanded_inner_t)

    for tri in elements:
        pts = points_[tri]

        # Check if all three vertices of the triangle are inside or on the inner T-shape boundary
        inside_or_on_boundary = np.all(
            inner_t_path.contains_points(pts) | np.isclose(pts[:, None], inner_t).all(axis=2).any(axis=1))

        if inside_or_on_boundary:
            plt.fill(pts[:, 0], pts[:, 1], edgecolor='blue', fill=False)
            plt.scatter(pts[:, 0], pts[:, 1], color='green', s=10)  # Mark triangle vertices

    plt.title("Inner T-Shape Mesh")
    # Superimpose the inner T-shape in red and outer_rectangle in black
    plt.plot(inner_t[:, 0], inner_t[:, 1], color='red', linewidth=2)
    plt.plot(outer_rectangle[:, 0], outer_rectangle[:, 1], color='black', linewidth=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def plot_dielectric_mesh(points_, elements):
    """Plots only the triangles outside the inner T-shape."""
    plt.figure(figsize=(8, 6))
    # Convert inner_t to a shapely Polygon
    inner_t_polygon = Polygon(inner_t)

    # Expand the boundary outward by 0.2
    expanded_t_polygon = inner_t_polygon.buffer(0.2)

    # Get the new expanded boundary coordinates
    expanded_inner_t = np.array(expanded_t_polygon.exterior.coords)

    # Define a path representing the inner T-shape polygon
    inner_t_path = Path(expanded_inner_t)

    for tri in elements:
        pts = points_[tri]

        # Check if all three vertices of the triangle are inside or on the inner T-shape boundary
        inside_or_on_boundary = np.all(
            inner_t_path.contains_points(pts) | np.isclose(pts[:, None], inner_t).all(axis=2).any(axis=1))

        if not inside_or_on_boundary:
            plt.fill(pts[:, 0], pts[:, 1], edgecolor='blue', fill=False)
            plt.scatter(pts[:, 0], pts[:, 1], color='green', s=10)  # Mark triangle vertices

    plt.title("Dielectric Mesh")
    # Superimpose the inner T-shape in red and outer_rectangle in black
    plt.plot(inner_t[:, 0], inner_t[:, 1], color='red', linewidth=2)
    plt.plot(outer_rectangle[:, 0], outer_rectangle[:, 1], color='black', linewidth=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def compute_inner_t_statistics(points_, elements):
    """Computes descriptive statistics of the triangles inside or on the boundary of the T-shape."""
    inner_t_polygon = Polygon(inner_t)
    expanded_t_polygon = inner_t_polygon.buffer(0.2)
    expanded_inner_t = np.array(expanded_t_polygon.exterior.coords)
    inner_t_path = Path(expanded_inner_t)

    areas = []
    aspect_ratios = []
    areas_ = []
    aspect_ratios_ = []

    for tri in elements:
        pts = points_[tri]

        inside_or_on_boundary = np.all(
            inner_t_path.contains_points(pts) | np.isclose(pts[:, None], inner_t).all(axis=2).any(axis=1))

        if inside_or_on_boundary:
            a, b, c = [np.linalg.norm(pts[i] - pts[(i + 1) % 3]) for i in range(3)]
            s = (a + b + c) / 2  # Semi-perimeter
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))  # Heron's formula
            aspect_ratio = min(a, b, c) / max(a, b, c)
            areas.append(area)
            aspect_ratios.append(aspect_ratio)

        else:
            a, b, c = [np.linalg.norm(pts[i] - pts[(i + 1) % 3]) for i in range(3)]
            s = (a + b + c) / 2  # Semi-perimeter
            area_ = np.sqrt(s * (s - a) * (s - b) * (s - c))  # Heron's formula
            aspect_ratio_ = min(a, b, c) / max(a, b, c)
            areas_.append(area_)
            aspect_ratios_.append(aspect_ratio_)

    print("Descriptive Statistics for Inner T-Shape Triangles:")
    print(f"Total Triangles count: {len(areas_)}")
    # print(f"Mean Area: {np.mean(areas):.4f}, \n Std Dev: {np.std(areas):.4f}")
    print(
        f" Mean Aspect Ratio: {np.mean(aspect_ratios_):.6f},\n Std Dev: {np.std(aspect_ratios_):.6f}, \n min: {np.min(aspect_ratios_):.6f}, \n 25%: {np.percentile(aspect_ratios_, 25):.6f}")
    print(
        f" 50%: {np.percentile(aspect_ratios_, 50):.6f}, \n 75%: {np.percentile(aspect_ratios_, 75):.6f}, \n Max: {np.max(aspect_ratios_):.6f}")

    print("Descriptive Statistics for Dielectric Triangles:")
    print(f"Total Triangles count: {len(areas)}")
    # print(f"Mean Area: {np.mean(areas):.4f}, \n Std Dev: {np.std(areas):.4f}")
    print(
        f" Mean Aspect Ratio: {np.mean(aspect_ratios):.6f},\n Std Dev: {np.std(aspect_ratios):.6f}, \n min: {np.min(aspect_ratios):.6f}, \n 25%: {np.percentile(aspect_ratios, 25):.6f}")
    print(
        f" 50%: {np.percentile(aspect_ratios, 50):.6f}, \n 75%: {np.percentile(aspect_ratios, 75):.6f}, \n Max: {np.max(aspect_ratios):.6f}")


def solve_electrostatics(points_, elements, boundary_conditions_, permittivity_):
    """Solves Poisson's or Laplace's equation using FEM."""
    n = len(points_)
    A = lil_matrix((n, n))
    b = np.zeros(n)

    for tri in elements:
        pts = points_[tri]
        area = 0.5 * abs(np.linalg.det(np.hstack((pts, np.ones((3, 1))))))
        for i in range(3):
            for j in range(3):
                A[tri[i], tri[j]] += area * permittivity_ * (1 if i == j else 0.5)

    for idx, value in boundary_conditions_.items():
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = value

    potential_ = spsolve(A.tocsr(), b)
    return potential_


# Visualize the Potential and Field
def visualize_potential_and_field(points_, elements, potential_):
    """Visualizes the electrostatic potential and field distribution."""
    plt.figure(figsize=(8, 6))

    # Interpolate the potential at element centroids
    face_potential = np.mean(potential_[elements], axis=1)

    plt.tripcolor(points_[:, 0], points_[:, 1], elements, facecolors=face_potential, cmap='jet', shading='flat')
    plt.colorbar(label="Potential (V)")

    # Compute electric field E = -∇V
    Ex, Ey = np.zeros(len(points_)), np.zeros(len(points_))
    for tri in elements:
        pts = points_[tri]
        V = potential_[tri]

        # Compute gradient for each triangle using finite differences
        A = np.vstack([pts.T, np.ones(3)]).T  # Affine transformation matrix
        b_x = np.array([V[0], V[1], V[2]])
        b_y = np.array([V[0], V[1], V[2]])

        gradV = np.linalg.lstsq(A, b_x, rcond=None)[0][:2]  # Least squares to find gradient
        Ex[tri] = -gradV[0]  # Negative gradient in x
        Ey[tri] = -gradV[1]  # Negative gradient in y

    plt.quiver(points_[:, 0], points_[:, 1], Ex, Ey, color='black', scale=50)
    plt.title("Electrostatic Potential and Field Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# Define T-shaped patch geometry
outer_rectangle = np.array([[0, 0], [10, 0], [10, 8], [0, 8], [0, 0]])
inner_t = np.array(
    [[1.2, 4], [1, 4.2], [1, 6.8], [1.2, 7], [8.8, 7], [9, 6.8], [9, 4.2], [8.8, 4], [6.2, 4], [6, 3.8], [6, 0.2],
     [5.8, 0], [3.7, 0], [3.5, 0.2], [3.5, 3.8], [3.3, 4], [1.2, 4]])

# Combine points and define facets (ensuring closed loops)
points = np.vstack((outer_rectangle, inner_t))
facets = [[i, (i + 1) % len(outer_rectangle)] for i in range(len(outer_rectangle))] + \
         [[len(outer_rectangle) + i, len(outer_rectangle) + (i + 1) % len(inner_t)] for i in range(len(inner_t))]

# Generate Initial CDT Mesh
initial_points, initial_elements = generate_initial_cdt(points, facets)
plot_mesh(initial_points, initial_elements, title="Initial CDT Mesh")

# Apply AFM Refinement
refined_points, refined_elements = advancing_front_refinement(initial_points, initial_elements)
print(refined_points)
print(refined_elements)
plot_mesh(refined_points, refined_elements, title="Refined Mesh with AFM")

# Plot inner T-shape mesh separately
plot_inner_t_mesh(refined_points, refined_elements)

# Plot dielectric mesh separately
plot_dielectric_mesh(refined_points, refined_elements)

# Compute statistics for inner T-shape triangles
compute_inner_t_statistics(refined_points, refined_elements)

# Electrostatics:
boundary_conditions = {0: 1.0, 10: 0.0}  # Example: Fixed potential at two nodes
permittivity = 1.0  # Example: Uniform permittivity throughout
potential = solve_electrostatics(refined_points, refined_elements, boundary_conditions, permittivity)
visualize_potential_and_field(refined_points, refined_elements, potential)
