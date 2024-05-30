# hinhhoc.py

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from itertools import combinations
import string
import os

class Constraint:
    def __init__(self, coefficients, operator, rhs):
        self.coefficients = np.array(coefficients)
        self.operator = operator
        self.rhs = rhs
    
    def satisfies(self, point):
        result = np.dot(self.coefficients, point)
        if self.operator == ">=":
            return result >= self.rhs
        elif self.operator == "<=":
            return result <= self.rhs
        elif self.operator == "=":
            return result == self.rhs
        
    def __str__(self):
        terms = []
        for idx, coef in enumerate(self.coefficients):
            terms.append(f"{coef}x_{idx + 1}")
        lhs = " + ".join(terms)
        return f"{lhs} {self.operator} {self.rhs}"


def solve_lp(problem_type, objective_coefficients, constraints):
    num_variables = len(objective_coefficients)
    intersection_points = []

    for constraint_combo in combinations(constraints, num_variables):
        coeffs = np.array([constraint.coefficients for constraint in constraint_combo])
        rhs_values = np.array([constraint.rhs for constraint in constraint_combo])
        try:
            intersection_point = np.linalg.solve(coeffs, rhs_values)
            intersection_points.append(intersection_point)
        except np.linalg.LinAlgError:
            continue

    feasible_points = []
    for point in intersection_points:
        if all(constraint.satisfies(point) for constraint in constraints):
            feasible_points.append(point)
    
    feasible_points = np.array(feasible_points)
    if feasible_points.size == 0:
        return None, None, None, constraints

    hull = ConvexHull(feasible_points)
    feasible_region_vertices = feasible_points[hull.vertices]

    objective_values = np.dot(feasible_region_vertices, objective_coefficients)
    if problem_type == "max":
        optimal_solution_index = np.argmax(objective_values)
    elif problem_type == "min":
        optimal_solution_index = np.argmin(objective_values)
    
    optimal_solution = feasible_region_vertices[optimal_solution_index]
    optimal_value = objective_values[optimal_solution_index]
    return optimal_solution, optimal_value, feasible_region_vertices, constraints


def plot_feasible_region(feasible_region_vertices, constraints, optimal_solution, title="Feasible Region and Optimal Solution"):
    points = np.array(feasible_region_vertices)
    hull = ConvexHull(points)
    points_plot = points[hull.vertices.astype(int), :]
    x, y = points_plot[:, 0], points_plot[:, 1]
    
    plt.figure(figsize=(12, 12))
    plt.axis('equal')
    plt.fill(x, y, hatch="\\", alpha=0.5)
    plt.plot(x, y, "bo", label="Feasible Region")
    
    plt.grid(True)
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    
    optimal_point = optimal_solution
    plt.plot(optimal_point[0], optimal_point[1], "ro", markersize=10, label="Optimal Solution")
    
    for point, marker in zip(points_plot, string.ascii_uppercase[:len(points_plot)]):
        label = f"{marker} ({point[0]:.1f}, {point[1]:.1f})"
        plt.text(point[0], point[1], label, fontsize="medium", position=(point[0]+0.2, point[1]+0.2))
    
    axes = plt.gca()
    for constraint in constraints:
        coeffs = constraint.coefficients
        rhs = constraint.rhs
        if coeffs[1] != 0:
            slope = -coeffs[0] / coeffs[1]
            intercept = rhs / coeffs[1]
            x_vals = np.array(axes.get_xlim())
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, label=f"{coeffs[0]}x + {coeffs[1]}y {constraint.operator} {rhs}")
    
    plt.title(title)
    plt.legend()
    
    # Save the plot as an image file
    filename = "feasible_region_plot.png"
    plt.savefig(filename)
    plt.close()  # Close the plot to prevent displaying it
    
    return filename
