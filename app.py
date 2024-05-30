import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from geometry import Constraint, solve_lp, plot_feasible_region
from PIL import Image
import io
import pandas as pd
import contextlib
from simplex import (
    InputObjectiveFunction,
    InputConstraints,
    InputObjectiveFunctionConditions,
    InputProblemType,
    PrintObjectiveFunction,
    PrintConstraints,
    convert_to_standard_form,
    convert_to_equations,
    SolveEquation
)
from two_phase import (
    InputObjectiveFunction,
    InputConstraints,
    InputObjectiveFunctionConditions,
    InputProblemType,
    PrintObjectiveFunction,
    PrintConstraints,
    reset_global_state,
    print_phase1_problem,
    convert_to_equations_x0,
    convert_to_phase1_form_x0,
    Solve
)
from dual import (
    InputObjectiveFunction,
    InputConstraints,
    InputObjectiveFunctionConditions,
    InputProblemType,
    PrintObjectiveFunction,
    PrintConstraints,
    convert_to_standard_form,
    convert_to_equations,
    SolveEquation
)
def capture_output(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()

def main():
    st.write(
        """
        <style>
            .full-width {
                width: 100%;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Linear Programming Solver")

    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Simplex Method", "Geometric Method","Two-Phase","Dual simplex Method"])
    if app_mode == "Simplex Method":
        st.header("Simplex Method")
        m = st.number_input("Enter the number of variables in the objective function:", min_value=1, value=2)
        c = []
        for i in range(m):
            coefficient = st.number_input(f"Enter the coefficient for x{i+1}:", value=0.0)
            c.append(coefficient)

        st.header("Input Constraints")
        n = st.number_input("Enter the number of constraints:", min_value=1, value=2)
        a = []
        b = []
        operators = []
        for i in range(n):
            constraint = []
            st.write(f"Constraint {i+1}:")
            for j in range(m):
                coefficient = st.number_input(f"Enter the coefficient for x{j+1} in constraint {i+1}:", value=0.0)
                constraint.append(coefficient)
            a.append(constraint)
            operator = st.selectbox(f"Select the operator for constraint {i+1}:", ["<=", ">=", "="], key=f"op{i}")
            operators.append(operator)
            value = st.number_input(f"Enter the value for constraint {i+1}:", value=0.0)
            b.append(value)

        st.header("Input Variable Conditions")
        conditions = []
        for i in range(m):
            condition = st.selectbox(f"Enter the condition for x{i+1}:", ["<= 0", ">= 0", "tùy ý"], key=f"cond{i}")
            conditions.append(condition)

        st.header("Input Problem Type")
        problem_type = st.selectbox("Do you want to maximize or minimize?", ["max", "min"])

        if st.button("Solve"):
            st.write("Objective Function:")
            obj_func_output = capture_output(PrintObjectiveFunction, c, problem_type)
            st.text(obj_func_output)

            st.write("Constraints:")
            constraints_output = capture_output(PrintConstraints, a, operators, b, conditions)
            st.text(constraints_output)

            st.write("Standard Form:")
            a, b, c, n, m, conditions = convert_to_standard_form(a, b, c, n, m, problem_type, operators, conditions)
            st.write("Converted Equations:")
            equations_output = capture_output(convert_to_equations, a, b, c, n, m, conditions)
            st.text(equations_output)

            st.write("Solving the Equation:")
            solution_output = capture_output(SolveEquation, a, b, c, n, m, conditions, problem_type)
            st.text("Solution:")
            st.text(solution_output)
    elif app_mode == "Geometric Method":
        st.header("Geometric Method")
        st.header("Input Problem Type")
        problem_type = st.selectbox("Select the problem type:", ["max", "min"])
        num_variables = st.number_input("Enter the number of variables in the objective function:", min_value=1, step=1)
        objective_coefficients = []
        for i in range(num_variables):
            coefficient = st.number_input(f"Enter the coefficient of variable x_{i + 1}:", step=0.1, format="%.2f")
            objective_coefficients.append(coefficient)
        st.header("Input Constraints")
        num_constraints = st.number_input("Enter the number of constraints:", min_value=1, step=1)
        constraints = []
        for i in range(num_constraints):
            st.write(f"Constraint {i + 1}:")
            coefficients = []
            for j in range(num_variables):
                coefficient = st.number_input(f"Enter the coefficient of variable x_{j + 1} for constraint {i + 1}:", step=0.1, format="%.2f")
                coefficients.append(coefficient)
            operator = st.selectbox(f"Select the operator for constraint {i + 1}:", ["<=", ">=", "="])
            rhs = st.number_input(f"Enter the right-hand side value for constraint {i + 1}:", step=0.1, format="%.2f")
            constraints.append(Constraint(coefficients, operator, rhs))

        if st.button("Solve"):
            optimal_solution, optimal_value, feasible_region_vertices, constraints = solve_lp(problem_type, objective_coefficients, constraints)
            if optimal_solution is None:
                st.write("No feasible solution found.")
            else:
                st.write("Optimal Solution:")
                for i, value in enumerate(optimal_solution):
                    st.write(f"x_{i + 1} = {value:.2f}")
                st.write(f"Optimal Value: {optimal_value:.2f}")
                
                st.write("Feasible Region and Optimal Solution:")
                filename = plot_feasible_region(feasible_region_vertices, constraints, optimal_solution)
                
                # Display the image in Streamlit
                image = Image.open(filename)
                st.image(image, caption="Feasible Region and Optimal Solution", use_column_width=True)
    elif app_mode == "Two-Phase":
        # Xóa nội dung trước đó trước khi hiển thị kết quả mới
        st.empty()
        st.header("Two-Phase Method")
        m = st.number_input("Enter the number of variables in the objective function:", min_value=1, value=2)
        c = []
        for i in range(m):
            coefficient = st.number_input(f"Enter the coefficient for x{i+1}:", value=0.0)
            c.append(coefficient)

        st.header("Input Constraints")
        n = st.number_input("Enter the number of constraints:", min_value=1, value=2)
        a = []
        b = []
        operators = []
        for i in range(n):
            constraint = []
            st.write(f"Constraint {i+1}:")
            for j in range(m):
                coefficient = st.number_input(f"Enter the coefficient for x{j+1} in constraint {i+1}:", value=0.0)
                constraint.append(coefficient)
            a.append(constraint)
            operator = st.selectbox(f"Select the operator for constraint {i+1}:", ["<=", ">=", "="], key=f"op{i}")
            operators.append(operator)
            value = st.number_input(f"Enter the value for constraint {i+1}:", value=0.0)
            b.append(value)

        st.header("Input Variable Conditions")
        conditions = []
        for i in range(m):
            condition = st.selectbox(f"Enter the condition for x{i+1}:", ["<= 0", ">= 0", "tùy ý"], key=f"cond{i}")
            conditions.append(condition)

        st.header("Input Problem Type")
        problem_type = st.selectbox("Do you want to maximize or minimize?", ["max", "min"])
        
        if st.button("Solve"):
            reset_global_state()
            
            st.write("Objective Function:")
            obj_func_output = capture_output(PrintObjectiveFunction, c, problem_type)
            st.text(obj_func_output)

            st.write("Constraints:")
            constraints_output = capture_output(PrintConstraints, a, operators, b, conditions)
            st.text(constraints_output)

            st.write("Standard Form:")
            a, b, c, n, m, conditions = convert_to_standard_form(a, b, c, n, m, problem_type, operators, conditions)
            
            st.write("Converted Equations:")
            current_c = c.copy()
            current_c  =  [0.0] * (1) +[0.0] * (n - 1) 
            phase1_problem_output = capture_output(print_phase1_problem, a, b,current_c,n, m)
            st.text(phase1_problem_output)
            
            equations_output = capture_output(convert_to_equations_x0, a, b,current_c , n, m, conditions)
            st.text(equations_output)

            st.write("Solving the Equation:")
            current_c  =  [0.0] * (1) +[0.0] * (n - 1)
            tableau, num_variables = convert_to_phase1_form_x0(a, b, current_c, n, m)
            solution_output = capture_output(Solve, tableau, m, n)
            st.text("Solution:")
            st.text(solution_output)
            
    elif app_mode == "Dual simplex Method":
        st.header("Dual simplex Method")
        m = st.number_input("Enter the number of variables in the objective function:", min_value=1, value=2)
        c = []
        for i in range(m):
            coefficient = st.number_input(f"Enter the coefficient for x{i+1}:", value=0.0)
            c.append(coefficient)

        st.header("Input Constraints")
        n = st.number_input("Enter the number of constraints:", min_value=1, value=2)
        a = []
        b = []
        operators = []
        for i in range(n):
            constraint = []
            st.write(f"Constraint {i+1}:")
            for j in range(m):
                coefficient = st.number_input(f"Enter the coefficient for x{j+1} in constraint {i+1}:", value=0.0)
                constraint.append(coefficient)
            a.append(constraint)
            operator = st.selectbox(f"Select the operator for constraint {i+1}:", ["<=", ">=", "="], key=f"op{i}")
            operators.append(operator)
            value = st.number_input(f"Enter the value for constraint {i+1}:", value=0.0)
            b.append(value)

        st.header("Input Variable Conditions")
        conditions = []
        for i in range(m):
            condition = st.selectbox(f"Enter the condition for x{i+1}:", ["<= 0", ">= 0", "tùy ý"], key=f"cond{i}")
            conditions.append(condition)

        st.header("Input Problem Type")
        problem_type = st.selectbox("Do you want to maximize or minimize?", ["max", "min"])

        if st.button("Solve"):
            st.write("Objective Function:")
            obj_func_output = capture_output(PrintObjectiveFunction, c, problem_type)
            st.text(obj_func_output)

            st.write("Constraints:")
            constraints_output = capture_output(PrintConstraints, a, operators, b, conditions)
            st.text(constraints_output)

            st.write("Standard Form:")
            a, b, c, n, m, conditions = convert_to_standard_form(a, b, c, n, m, problem_type, operators, conditions)
            st.write("Converted Equations:")
            equations_output = capture_output(convert_to_equations, a, b, c, n, m, conditions)
            st.text(equations_output)

            st.write("Solving the Equation:")
            solution_output = capture_output(SolveEquation, a, b, c, n, m, conditions, problem_type)
            st.text("Solution:")
            st.text(solution_output)
if __name__ == "__main__":
    main()
