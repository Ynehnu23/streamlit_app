import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from geometry import Constraint, solve_lp, plot_feasible_region
from PIL import Image
import io
import pandas as pd
import contextlib

# Import from simplex module
from simplex import (
    InputObjectiveFunction as SimplexInputObjectiveFunction,
    InputConstraints as SimplexInputConstraints,
    InputObjectiveFunctionConditions as SimplexInputObjectiveFunctionConditions,
    InputProblemType as SimplexInputProblemType,
    PrintObjectiveFunction as SimplexPrintObjectiveFunction,
    PrintConstraints as SimplexPrintConstraints,
    convert_to_standard_form as SimplexConvertToStandardForm,
    convert_to_equations as SimplexConvertToEquations,
    SolveEquation as SimplexSolveEquation
)

# Import from two_phase module
from two_phase import (
    InputObjectiveFunction as TwoPhaseInputObjectiveFunction,
    InputConstraints as TwoPhaseInputConstraints,
    InputObjectiveFunctionConditions as TwoPhaseInputObjectiveFunctionConditions,
    InputProblemType as TwoPhaseInputProblemType,
    PrintObjectiveFunction as TwoPhasePrintObjectiveFunction,
    PrintConstraints as TwoPhasePrintConstraints,
    reset_global_state as TwoPhaseResetGlobalState,
    print_phase1_problem as TwoPhasePrintPhase1Problem,
    convert_to_equations_x0 as TwoPhaseConvertToEquationsX0,
    convert_to_phase1_form_x0 as TwoPhaseConvertToPhase1FormX0,
    Solve as TwoPhaseSolve
)

# Import from dual module
from dual import (
    InputObjectiveFunction as DualInputObjectiveFunction,
    InputConstraints as DualInputConstraints,
    InputObjectiveFunctionConditions as DualInputObjectiveFunctionConditions,
    InputProblemType as DualInputProblemType,
    PrintObjectiveFunction as DualPrintObjectiveFunction,
    PrintConstraints as DualPrintConstraints,
    convert_to_standard_form as DualConvertToStandardForm,
    convert_to_equations as DualConvertToEquations,
    SolveEquation as DualSolveEquation
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
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Simplex Method", "Geometric Method", "Two-Phase Method", "Dual Simplex Method"])

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
            obj_func_output = capture_output(SimplexPrintObjectiveFunction, c, problem_type)
            st.text(obj_func_output)

            st.write("Constraints:")
            constraints_output = capture_output(SimplexPrintConstraints, a, operators, b, conditions)
            st.text(constraints_output)

            st.write("Standard Form:")
            a, b, c, n, m, conditions = SimplexConvertToStandardForm(a, b, c, n, m, problem_type, operators, conditions)
            st.write("Converted Equations:")
            equations_output = capture_output(SimplexConvertToEquations, a, b, c, n, m, conditions)
            st.text(equations_output)

            st.write("Solving the Equation:")
            solution_output = capture_output(SimplexSolveEquation, a, b, c, n, m, conditions, problem_type)
            st.text("Solution:")
            st.text(solution_output)
    
    elif app_mode == "Geometric Method":
        st.header("Geometric Method")
        st.header("Input Problem Type")
        problem_type = st.selectbox("Select the problem type:", ["max", "min"])
        num_variables = st.number_input("Enter the number of variables in the objective function:", min_value=1, step=1)
        objective_coefficients = []
        for i in range(num_variables):
            coefficient = st.number_input(f"Enter the coefficient of variable x_{i + 1}:", value=0.0, step=0.1, format="%.2f")
            objective_coefficients.append(coefficient)

        st.header("Input Constraints")
        num_constraints = st.number_input("Enter the number of constraints:", min_value=1, step=1)
        constraints = []
        for i in range(num_constraints):
            st.write(f"Constraint {i + 1}:")
            coefficients = []
            for j in range(num_variables):
                coefficient = st.number_input(f"Enter the coefficient of variable x_{j + 1} for constraint {i + 1}:", value=0.0, step=0.1, format="%.2f")
                coefficients.append(coefficient)
            operator = st.selectbox(f"Select the operator for constraint {i + 1}:", ["<=", ">=", "="])
            rhs = st.number_input(f"Enter the right-hand side value for constraint {i + 1}:", value=0.0, step=0.1, format="%.2f")
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
    
    elif app_mode == "Two-Phase Method":
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
            TwoPhaseResetGlobalState()
            
            st.write("Objective Function:")
            obj_func_output = capture_output(TwoPhasePrintObjectiveFunction, c, problem_type)
            st.text(obj_func_output)

            st.write("Constraints:")
            constraints_output = capture_output(TwoPhasePrintConstraints, a, operators, b, conditions)
            st.text(constraints_output)

            st.write("Standard Form:")
            a, b, c, n, m, conditions = SimplexConvertToStandardForm(a, b, c, n, m, problem_type, operators, conditions)
            
            st.write("Converted Equations:")
            current_c = c.copy()
            current_c  = [0.0] * (1) + [0.0] * (n - 1)
            phase1_problem_output = capture_output(TwoPhasePrintPhase1Problem, a, b, current_c, n, m)
            st.text(phase1_problem_output)
            
            equations_output = capture_output(TwoPhaseConvertToEquationsX0, a, b, current_c, n, m, conditions)
            st.text(equations_output)

            st.write("Solving the Equation:")
            current_c  = [0.0] * (1) + [0.0] * (n - 1)
            tableau, num_variables = TwoPhaseConvertToPhase1FormX0(a, b, current_c, n, m)
            solution_output = capture_output(TwoPhaseSolve, tableau, m, n)
            st.text("Solution:")
            st.text(solution_output)
    
    elif app_mode == "Dual Simplex Method":
        st.header("Dual Simplex Method")
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
            obj_func_output = capture_output(DualPrintObjectiveFunction, c, problem_type)
            st.text(obj_func_output)

            st.write("Constraints:")
            constraints_output = capture_output(DualPrintConstraints, a, operators, b, conditions)
            st.text(constraints_output)

            st.write("Standard Form:")
            a, b, c, n, m, conditions = DualConvertToStandardForm(a, b, c, n, m, problem_type, operators, conditions)
            st.write("Converted Equations:")
            equations_output = capture_output(DualConvertToEquations, a, b, c, n, m, conditions)
            st.text(equations_output)

            st.write("Solving the Equation:")
            solution_output = capture_output(DualSolveEquation, a, b, c, n, m, conditions, problem_type)
            st.text("Solution:")
            st.text(solution_output)

    with open("app/style.css") as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
