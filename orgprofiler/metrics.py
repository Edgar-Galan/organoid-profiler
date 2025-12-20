import math
import numpy as np
from typing import Tuple
from scipy.spatial import ConvexHull

def calculate_feret_features_from_points(points_xy: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Calculate Feret diameter features from a set of points.

    Returns:
        max_feret: Maximum Feret diameter
        min_feret: Minimum Feret diameter (caliper width)
        feret_angle: Angle of maximum Feret diameter in degrees
        feret_x: X-coordinate of Feret diameter start point
        feret_y: Y-coordinate of Feret diameter start point
    """
    if points_xy.shape[0] < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Find convex hull of the points
    convex_hull = ConvexHull(points_xy)
    hull_points = points_xy[convex_hull.vertices]

    # Find maximum distance between hull points (max Feret diameter)
    max_distance_squared = 0.0
    best_start_index = 0
    best_end_index = 0

    for i in range(len(hull_points)):
        for j in range(i + 1, len(hull_points)):
            delta_x = hull_points[j, 0] - hull_points[i, 0]
            delta_y = hull_points[j, 1] - hull_points[i, 1]
            distance_squared = delta_x * delta_x + delta_y * delta_y

            if distance_squared > max_distance_squared:
                max_distance_squared = distance_squared
                best_start_index = i
                best_end_index = j

    max_feret = math.sqrt(max_distance_squared)
    feret_angle = math.degrees(math.atan2(
        hull_points[best_end_index, 1] - hull_points[best_start_index, 1],
        hull_points[best_end_index, 0] - hull_points[best_start_index, 0]
    ))
    feret_x = float(hull_points[best_start_index, 0])
    feret_y = float(hull_points[best_start_index, 1])

    def calculate_width_for_edge(point_0, point_1):
        """Calculate the width perpendicular to an edge."""
        edge_vector_x = point_1[0] - point_0[0]
        edge_vector_y = point_1[1] - point_0[1]
        edge_length = math.hypot(float(edge_vector_x), float(edge_vector_y))

        if edge_length == 0:
            return float("inf")

        # Normal vector (perpendicular to edge)
        normal_x = -edge_vector_y / edge_length
        normal_y = edge_vector_x / edge_length

        # Project all hull points onto the normal
        projections = hull_points @ np.array([normal_x, normal_y], dtype=float)
        return float(projections.max() - projections.min())

    # Find minimum width (caliper width)
    min_feret = float("inf")
    for i in range(len(hull_points)):
        width = calculate_width_for_edge(hull_points[i], hull_points[(i + 1) % len(hull_points)])
        if width < min_feret:
            min_feret = width

    return float(max_feret), float(min_feret), float(feret_angle), feret_x, feret_y

