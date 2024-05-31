import sympy as sp
import numpy as np
import streamlit as st
# -*- coding: utf-8 -*-
# Định nghĩa class Position
from rich.console import Console
from rich.table import Table
EPS = 1e-4
def reset_global_state():
    global GLOBAL_STATE
    GLOBAL_STATE = {}

class Position:
    def __init__(self, row, column):
        self.row = row
        self.column = column

def ReadEquation():
    n, m = map(int, st.text_input("Nhập số lượng ràng buộc (n) và biến (m), cách nhau bằng dấu cách:").split())
    a = []
    for row in range(n):
        row_input = st.text_input(f"Nhập hệ số của các biến trong ràng buộc thứ {row + 1}, cách nhau bằng dấu cách:")
        a.append(list(map(float, row_input.split())))
    b_input = st.text_input("Nhập hệ số sau toán tử ràng buộc, cách nhau bằng dấu cách:")
    c_input = st.text_input("Nhập hệ số của hàm mục tiêu, cách nhau bằng dấu cách:")
    b = list(map(float, b_input.split()))
    c = list(map(float, c_input.split()))
    return a, b, c, n, m

def InputObjectiveFunction():
    m = st.number_input("Nhập số lượng biến của hàm mục tiêu: ")
    st.write("Nhập hệ số của các biến trong hàm mục tiêu:")
    c = []
    if m is not None and m > 0:
        for i in range(int(m)):
            coefficient = float(st.text_input(f"Nhập hệ số của x{i+1}: "))
            c.append(coefficient)
    else:
        st.error("Số lượng biến phải là một số nguyên dương.")
    return m, c

def InputConstraints():
    n = st.number_input("Nhập số lượng ràng buộc:")
    a = []
    b = []
    operators = []
    st.write("Nhập ràng buộc:")
    if n is not None and n > 0:
        for i in range(int(n)):
            constraint = []
            st.write(f"Nhập hệ số của các biến trong ràng buộc thứ {i+1}:")
            for j in range(m):  # Assuming 'm' is defined globally
                coefficient = float(st.text_input(f"Nhập hệ số của x{j+1}: "))
                constraint.append(coefficient)
            a.append(constraint)
            operator = st.text_input("Nhập toán tử ràng buộc (>=, <=, =): ")
            operators.append(operator)
            value = float(st.text_input("Nhập hệ số sau toán tử ràng buộc: "))
            b.append(value)
    else:
        st.error("Số lượng ràng buộc phải là một số nguyên dương.")
    return n, a, operators, b


def InputProblemType():
    problem_type = st.text_input("Bạn muốn tìm max hay min (max/min): ")
    return problem_type

def InputObjectiveFunctionConditions(m):
    conditions = []
    if m is not None and m > 0:
        for i in range(int(m)):
            condition = st.text_input(f"Nhập điều kiện của x{i+1} (<= 0, >= 0, tùy ý): ")
            conditions.append(condition)
    else:
        st.error("Số lượng biến của hàm mục tiêu phải là một số nguyên dương.")
    return conditions
def PrintObjectiveFunction(c, problem_type):
    global GLOBAL_STATE
    GLOBAL_STATE = {"c": None}

    if m is None or c is None:
        st.error("Số lượng biến không được để trống.")
        return
    print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print(" Tìm giá trị", end=" ",)
    if problem_type == "max":
        print("lớn nhất" if problem_type == "max" else "nhỏ nhất", "của hàm:")
    elif problem_type == "min":
        print("nhỏ nhất" if problem_type == "min" else "lớn nhất", "của hàm:")
    print(" z =", end=" ")
    for i in range(len(c)):
        print(f"{c[i]}x{i+1}", end=" ")
    print()
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")


def PrintConstraints(a, operators, b, conditions):
    print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print(" Thỏa mãn các ràng buộc sau:")
    for i in range(len(a)):
        constraint_string = " + ".join([f"{a[i][j]}x{j+1}" for j in range(len(a[i]))])
        constraint_string += f" {operators[i]} {b[i]}"
        print("┃", constraint_string)
    for i, condition in enumerate(conditions):
        print("┃", f"x{i+1} {condition}")
    print()
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

def convert_to_standard_form(a, b, c, n, m, problem_type, operators, conditions):
     # Kiểm tra đầu vào hợp lệ
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Số lượng ràng buộc 'n' phải là một số nguyên dương.")
    if not isinstance(m, int) or m <= 0:
        raise ValueError("Số lượng biến 'm' phải là một số nguyên dương.")
    print("Bài toán có sau khi chuyển đổi:")
    new_c = []
    new_a = []
    new_b = []
    new_conditions = []
    print("Hàm mục tiêu sau khi chuyển đổi:")
    print("z =", end=" ")
    
    for i in range(len(c)):
      
        if conditions[i] == "<= 0":
            new_conditions.append(conditions[i])
            new_c.append(-c[i])
            if problem_type == "max":
               new_c[i] = -new_c[i]
            else:
               new_c[i] = new_c[i]
            print(f"-{c[i]}x{i+1}", end="")
        elif conditions[i] == "tùy ý":
            new_conditions.append(conditions[i])
            new_conditions.append(conditions[i]) 
            new_c.append(c[i])
            new_c.append(-c[i])
            if problem_type == "max":
               new_c[i] = -new_c[i]
            else:
               new_c[i] = new_c[i]
            print(f"{new_c[i]}x{i+1} - {new_c[i]}x{i+1}_t", end="")
        elif conditions[i] == ">= 0":
            new_conditions.append(conditions[i])
            new_c.append(c[i])
            if problem_type == "max":
              new_c[i] = -new_c[i]
            else:
              new_c[i] = new_c[i]
            print(f"{c[i]}x{i+1}", end="")
          else:
            raise ValueError(f"Toán tử '{operators[i]}' trong ràng buộc không hợp lệ.")
        if i < len(c) - 1:
            print(" +", end=" ")
    print()
    result_strings = []
    print("Thỏa mãn các ràng buộc sau:")
    for i in range(n):
        new_constraint = ""
        new_constraint_1 = ""
        temp_constraint = []
        for j in range(m):
            if conditions[j] == "<= 0":
                temp_constraint.append(-a[i][j])  # Đảo dấu của hệ số
                if -a[i][j] > 0 and -a[i][j] != a[i][0]:
                    new_constraint += f" + {-a[i][j]}x{j+1}"
                else:
                    new_constraint += f" {-a[i][j]}x{j+1}"
            elif conditions[j] == "tùy ý":
                # Lưu trữ giá trị biến hiện tại để sử dụng sau này
                current_variable = a[i][j]
                # Thêm biến mới vào danh sách ràng buộc
                temp_constraint.append(current_variable)
                temp_constraint.append(-current_variable)
                if current_variable > 0 and current_variable != a[i][0]:
                    new_constraint += f" + {current_variable}x{j+1} - {current_variable}x{j+1}_t"
                else:
                    new_constraint += f" {current_variable}x{j+1} - {current_variable}x{j+1}_t"
            else:
                temp_constraint.append(a[i][j])  # Giữ nguyên hệ số
                if a[i][j] > 0 and a[i][j] != a[i][0]:
                    new_constraint += f" + {a[i][j]}x{j+1}"
                else:
                    new_constraint += f" -{a[i][j]}x{j+1}"
        new_a.append(temp_constraint) 
        # Append temp_constraint to new_a
        if operators[i] == ">=":   
            new_a[i] = [-coeff for coeff in new_a[i]]  # Đảo dấu của từng phần tử trong danh sách new_a
            new_constraint += f" <= {-b[i]}"
            new_b.append(-b[i])
        elif operators[i] == "=":
            new_a.append([-coeff for coeff in temp_constraint])  # Negate the constraint
            for j in range(m):
                if conditions[j] == "<= 0":
                    new_constraint_1 += f" {-temp_constraint[j]}x{j+1}"
                elif conditions[j] == "tùy ý":
                    current_variable = -temp_constraint[j]
                    new_constraint_1 += f" {current_variable}x{j+1} - {current_variable}x{j+1}_t"
                else:
                    new_constraint_1 += f" {-temp_constraint[j]}x{j+1}"
            new_constraint_1 += f" <= {-b[i]}"
            new_b.append(-b[i])
            new_constraint = ""
            for j in range(m):
                if conditions[j] == "<= 0":
                    new_constraint_1 += f" {temp_constraint[j]}x{j+1}"
                elif conditions[j] == "tùy ý":
                    current_variable = temp_constraint[j]
                    new_constraint += f" {current_variable}x{j+1} - {current_variable}x{j+1}_t"
                else:
                    new_constraint += f" {temp_constraint[j]}x{j+1}"
            new_constraint += f" <= {b[i]}"
            new_b.append(b[i])
        elif operators[i] == "<=":
            new_constraint += f" <= {b[i]}"
            new_a[i] = [coeff for coeff in new_a[i]] 
            new_b.append(b[i])
        result_strings.append(new_constraint_1.strip())
        result_strings.append(new_constraint.strip())
    for constraint in result_strings:
        print(constraint)

    # Xóa các dòng trong a có số lượng phần tử khác với new_m
    new_a = [row for row in new_a if len(row) == len(new_c)]
    return new_a, new_b, new_c, len(new_b), len(new_c), new_conditions
def CreateTableau_x0(a, b, c, n, m):
    tableau = []
    c = [0.0] * m  + [0.0] * n +  [1.0]
    for i in range(n):
        slack_variables = [0] * n
        slack_variables[i] = 1.0
        tableau_row = [-x for x in a[i]] + slack_variables + [1] + [b[i]]  # Chỉnh sửa giá trị của cột x0 thành 1
        tableau.append(tableau_row)
        # Thêm cột biến phụ
        tableau[i][m + i] = 1.0
    # Thêm hàng cho ràng buộc -x0
    tableau_row_x0 = [0.0] * (m + n) + [1.0] + [0.0]
    tableau.append(tableau_row_x0)
    # Thêm cột hệ số của biến phụ trong hàng mục tiêu
    final_row = [x for x in c]  +[0.0] # Giá trị của cột x0 trong hàng mục tiêu là 0
    tableau.append(final_row)
    return tableau

def print_tableau_x0(tableau, m, n, conditions):
    # Tạo danh sách các biến và hằng số
    variables = []
    variable_index = 1
    is_last_arbitrary = False

    # Thêm các biến tối ưu vào danh sách biến
    for i in range(len(conditions)):
        if conditions[i] == "tùy ý":
            if not is_last_arbitrary:
                variables.append('x{}'.format(variable_index))
                is_last_arbitrary = True
            else:
                variables.append('x{}_t'.format(variable_index))
                variable_index += 1
                is_last_arbitrary = False
        else:
            variables.append('x{}'.format(variable_index))
            variable_index += 1
            is_last_arbitrary = False


    # Thêm các biến slack vào danh sách biến
    for i in range(n):
        variables.append('w{}'.format(i + 1))
    # Thêm biến x0 vào danh sách biến
    variables.append('x0')
    # Thêm hằng số vào danh sách biến
    variables.append('const.')
    
    # Thêm các cột cho biến và hằng số vào bảng Rich
    table = Table(show_header=True, header_style="bold")
    for variable in variables:
        table.add_column(variable)
    
    # Thêm hàng vào bảng Rich
    for row in tableau:
        # Lấy giá trị của hàng trừ cột cuối cùng
        row_values = ['%.2f' % x for x in row[:-1]]
        # Thêm giá trị của cột cuối cùng vào hàng
        row_values.append('%.2f' % row[-1])
        table.add_row(*row_values)
    
    # In bảng Rich
    console = Console()
    console.print(table)

        
def SelectPivotElement_x0(tableau):
    # Tìm cột của phần tử pivot (cột của biến có hệ số 1 ở hàng cuối cùng)
    pivot_column = -1
    for c in range(len(tableau[0]) - 1):
        if tableau[-1][c] == 1:
            pivot_column = c
            break

    # Tìm hàng của phần tử pivot (hàng có giá trị const âm nhất trừ hàng cuối)
    min_const = float('inf')
    pivot_row = -1
    for r in range(len(tableau) - 1):
        if tableau[r][-1] < min_const:
            min_const = tableau[r][-1]
            pivot_row = r

    return Position(pivot_row, pivot_column)

def ProcessPivotElement_x0(tableau, pivot_element):
    pivot_row = pivot_element.row
    pivot_column = pivot_element.column
    pivot_value = tableau[pivot_row][pivot_column]

    # Đảo dấu các phần tử trừ phần tử pivot trong hàng pivot
    tableau[pivot_row] = [-element / pivot_value if i != pivot_column and i != m + pivot_row else element for i, element in enumerate(tableau[pivot_row])]

    # Áp dụng phép biến đổi hàng cho các hàng còn lại
    for i in range(len(tableau)-1):  # Loại bỏ hàng cuối cùng
        if i != pivot_row:
            multiplier = tableau[i][pivot_column]
            tableau[i] = [a + b * multiplier if j != pivot_column else b for j, (a, b) in enumerate(zip(tableau[i], tableau[pivot_row]))]
    
    # Đảo dấu các phần tử trong cột chứa pivot mà không phải pivot
    for i in range(len(tableau)):
      if i != pivot_row:  # Không thực hiện trên hàng pivot
        tableau[i][pivot_column] = tableau[i][pivot_column]-tableau[i][pivot_column]
    # Gán hàng cuối cùng bằng hàng pivot đã rút pivot trừ cột pivot
    tableau[-1] = [element if i != pivot_column else 0 for i, element in enumerate(tableau[pivot_row])]
    return tableau

def CreateTableau(a, b, c, n, m):
    tableau = []
    for i in range(n):
        slack_variables = [0] * n
        slack_variables[i] = 1.0
        tableau_row = [-x for x in a[i]] + slack_variables + [b[i]]
        tableau.append(tableau_row)
    # Thêm cột hệ số của biến phụ trong hàng mục tiêu
    final_row = [x for x in c] + [0] * n + [0]
    tableau.append(final_row)
    return tableau
def print_tableau(tableau, n, m, conditions):
    # Tạo bảng Rich
    console = Console()
    table = Table(show_header=True, header_style="bold")

    # Tạo danh sách các biến và hằng số
    variables = []
    variable_index = 1
    is_last_arbitrary = False
    for i in range(m):
        if conditions[i] == "tùy ý":
            if not is_last_arbitrary:
                variables.append('x{}'.format(variable_index))
                is_last_arbitrary = True
            else:
                variables.append('x{}_t'.format(variable_index))
                variable_index += 1
                is_last_arbitrary = False
        else:
            variables.append('x{}'.format(variable_index))
            variable_index += 1
            is_last_arbitrary = False

    # Thêm các biến w và hằng số vào danh sách biến
    for i in range(n):
        variables.append('w{}'.format(i + 1))
    variables.append('const.')
    
    # Thêm các cột cho biến và hằng số vào bảng Rich
    for variable in variables:
        table.add_column(variable)
    # Thêm hàng vào bảng Rich
    for row in tableau:
        table.add_row(*["{:.2f}".format(cell) for cell in row])
   
    # In bảng Rich
    console.print(table)
def SelectPivotElement(tableau):
    # Xác định số cột và số hàng trong bảng
    num_columns = len(tableau[0])
    num_rows = len(tableau) - 1
    
    # Kiểm tra nếu các phần tử ở cột cuối cùng, trừ phần tử ở hàng cuối cùng, đều bằng 0
    if all(row[-1] == 0 for row in tableau[:-1]):
        # Tìm vị trí của phần tử âm đầu tiên trong hàng cuối cùng
        pivot_column = -1
        for j in range(num_columns - 1):
            if tableau[-1][j] < 0:
                pivot_column = j
                break
    else:
        # Tìm vị trí của phần tử âm đầu tiên trong hàng cuối cùng (trừ cột cuối cùng)
        min_value = min(tableau[-1][:-1])
        pivot_column = tableau[-1].index(min_value)
    
    # Tìm phần tử dưới pivot_column là số âm đầu tiên trong cột
    min_ratio = float('inf')
    pivot_row = -1
    for r in range(num_rows):
        if tableau[r][pivot_column] < 0:  # Chỉ xét các hệ số âm
            ratio = tableau[r][-1] / abs(tableau[r][pivot_column])  # Sử dụng giá trị tuyệt đối
            if ratio < min_ratio:
                min_ratio = ratio
                pivot_row = r
    if pivot_row == -1:
        return None 
    return Position(pivot_row, pivot_column)

def ProcessPivotElement(a, pivot_element):
    # Get the absolute value of the pivot element
    pivot_element_value = abs(a[pivot_element.row][pivot_element.column])

    if any(coefficient == 1 for coefficient in a[pivot_element.row][:-1]):
     pivot_row = a[pivot_element.row]
     pivot_column_index = pivot_element.column
     for j, coefficient in enumerate(pivot_row):
        if coefficient == 1 and j != pivot_column_index:  # Chỉ quan tâm đến cột của pivot
            column_values = [column[j] for column in a]
            contains_zero = any(value == 0 for value in column_values)
            contains_one = any(value == 1 for value in column_values)
            if contains_zero and contains_one:
                if pivot_row[j] == 1:
                    pivot_row[j] *= -1
                    print(j)
                break
    # Apply division by the absolute value of the pivot element
    a[pivot_element.row] = [n / pivot_element_value for n in a[pivot_element.row]]

    # Mark the pivot element
    a[pivot_element.row][pivot_element.column] = 1.0
   
    # Apply row operations to other rows
    for i in range(len(a)):
        if i != pivot_element.row:
            sec_mult = a[i][pivot_element.column]
            pri_row = [j * sec_mult for j in a[pivot_element.row]]
            if a[i] == a[-1]:
                a[-1] = [a + b for a, b in zip(a[i], pri_row)]
                a[-1][pivot_element.column] = 0.0
            else: 
                a[i] = [a + b for a, b in zip(a[i], pri_row)]
    for i in range(len(a)):
        if i != pivot_element.row:
           a[i][pivot_element.column] = 0.0            
    return a
def print_phase1_problem(a, b, current_c, n, m):
    if current_c is None:
        st.error("Hệ số của biến mục tiêu không được để trống.")
        return
    print("Bài toán đã chuyển đổi:")
    print("Hàm mục tiêu sau khi chuyển đổi:")
    print("z =", " + ".join([f"{current_c[i]}x{i+1}" for i in range(len(current_c))]), "+ x0") # Thêm biến x0 vào hàm mục tiêu
    print("Thỏa mãn các ràng buộc sau:")
    for i in range(n):
        print(f"{a[i][0]}*x1 + {a[i][1]}*x2 {-1}*x0 <= {b[i]}")
    print()           
def SolvePhase1(tableau, m):
    iteration = 1
    # Bước 1: Sử dụng hàm CreateTableau_x0 và in bảng ban đầu
    tableau = CreateTableau_x0(a, b, c, n, m)
    print("Bước №", iteration)
    print_tableau_x0(tableau, m, n,conditions)
    print("---------------------------------------------")
    # Chọn phần tử pivot
    pivot_element = SelectPivotElement_x0(tableau)
    print("Phần tử chốt đã chọn:", pivot_element.row, ",", pivot_element.column, " có giá trị là:",
            tableau[pivot_element.row][pivot_element.column])
    print("---------------------------------------------")
        # Xử lý phần tử pivot trong bảng
    print("Bảng sau khi xoay từ vựng:")
    tableau = ProcessPivotElement_x0(tableau, pivot_element)
    print_tableau_x0(tableau, m, n,conditions)
    print("---------------------------------------------")
    while not all(num >= 0 for num in tableau[-1][:-1]):
        iteration += 1
        print("Bước №", iteration)
        print_tableau_x0(tableau, m, n,conditions)
        # Chọn phần tử pivot
        pivot_element = SelectPivotElement(tableau)

        print("Phần tử chốt đã chọn:", pivot_element.row, ",", pivot_element.column, " có giá trị là:",
              f"{tableau[pivot_element.row][pivot_element.column]:.2f}") 
        print("---------------------------------------------")
        # Xử lý phần tử pivot trong bảng
        print("Bảng sau khi xoay từ vựng:")
        tableau = ProcessPivotElement(tableau, pivot_element)
        print_tableau_x0(tableau, m, n,conditions)
        print("---------------------------------------------")
    return tableau
    
def phase_two_problem(c, tableau):
    num_variables = len(c)
    num_rows = len(tableau)
    num_columns = len(tableau[0])
    
    for row in tableau:
        row[-2] = 0
    
    # Khởi tạo z_temp là một hàng có độ dài bằng số cột của tableau và giá trị ban đầu là 0
    z_temp = [0] * num_columns

    # Duyệt qua từng biến trong hàm mục tiêu
    for i in range(num_variables):
        # Xét cột thứ i
        column = [row[i] for row in tableau]
        
        # Tìm dòng có giá trị 1 trong cột i và các giá trị khác bằng 0
        row_with_one = -1
        for j in range(num_rows):
            if column[j] == 1 and all(column[k] == 0 for k in range(num_rows) if k != j):
                row_with_one = j
                break
        
        if row_with_one != -1:
            # Nếu tìm thấy dòng thỏa mãn điều kiện
            row = tableau[row_with_one]
            z_exp = [c[i] * value for value in row]
            z_exp[i] = 0  # Đặt giá trị tại [dòng đó][cột i] = 0
        else:
            # Nếu không tìm thấy dòng thỏa mãn điều kiện
            z_exp = [0] * num_columns
            z_exp[i] = c[i]  # Đặt giá trị tại [cột i] trong z_temp
        
        # Cộng dồn kết quả vào z_temp
        z_temp = [x + y for x, y in zip(z_temp, z_exp)]
        
    # Cập nhật hàm mục tiêu trong tableau
    tableau[-1] = z_temp
    tableau = [row[:-2] + [row[-1]] for row in tableau]
    return tableau

def ProcessPivotElement(a, pivot_element):
    # Get the absolute value of the pivot element
    pivot_element_value = abs(a[pivot_element.row][pivot_element.column])

    if any(coefficient == 1 for coefficient in a[pivot_element.row][:-1]):
        pivot_row = a[pivot_element.row]
        for j in range(len(a[pivot_element.row])):
         coefficient = a[pivot_element.row][j]
         if coefficient == 1:  # If the coefficient is 1
            column_values = [column[j] for column in a]
            contains_zero = False
            contains_one = False
            for value in column_values:
                if value == 0.0:
                    contains_zero = True
                elif value == 1.0:
                    contains_one = True
                if contains_zero and contains_one:
                    if pivot_row[j] == 1:
                        pivot_row[j] *= -1
                    break
    # Apply division by the absolute value of the pivot element
    a[pivot_element.row] = [n / pivot_element_value for n in a[pivot_element.row]]

    # Mark the pivot element
    a[pivot_element.row][pivot_element.column] = 1.0
   
    # Apply row operations to other rows
    for i in range(len(a)):
        if i != pivot_element.row:
            sec_mult = a[i][pivot_element.column]
            pri_row = [j * sec_mult for j in a[pivot_element.row]]
            if a[i] == a[-1]:
                a[-1] = [a + b for a, b in zip(a[i], pri_row)]
                a[-1][pivot_element.column] = 0.0
            else: 
                a[i] = [a + b for a, b in zip(a[i], pri_row)]
    for i in range(len(a)):
        if i != pivot_element.row:
           a[i][pivot_element.column] = 0.0            
    return a
def check_unbounded(tableau):
    num_columns = len(tableau[0]) - 1  # Trừ đi cột chứa giá trị hằng số
    num_rows = len(tableau) - 1  # Trừ đi hàng mục tiêu
    
    for j in range(num_columns):
        if tableau[-1][j] < 0:  # Nếu phần tử trong hàng mục tiêu là âm
            if all(tableau[i][j] > 0 for i in range(num_rows)):  # Và tất cả các phần tử khác trong cột đó là dương
                return True
    return False

def SolvePhase2(tableau, m):
    iteration = 1
    tableau = phase_two_problem(c, tableau)
    while not all(num >= 0 for num in tableau[-1][:-1]):
        print("Bước №", iteration)
        print_tableau(tableau,n, m,conditions)
        # Chọn phần tử pivot
        pivot_element = SelectPivotElement(tableau)
        print("Phần tử chốt đã chọn:", pivot_element.row, ",", pivot_element.column, " có giá trị là:",
             f"{tableau[pivot_element.row][pivot_element.column]:.2f}") 
        print("---------------------------------------------")
        # Xử lý phần tử pivot trong bảng
        print("Bảng sau khi xoay từ vựng:")
        tableau = ProcessPivotElement(tableau, pivot_element)
        print_tableau(tableau, n,m,conditions)
        print("---------------------------------------------")
        iteration += 1
    return tableau
def Solve(tableau, m, n):
    # Solve Phase 1 problem
    reset_global_state()
    tableau = SolvePhase1(tableau, m)  
    
    # Check condition to switch to Phase 2
    if all(num == 0 for num in tableau[-1][:-2]) and tableau[-1][-2] == 1:
        print("Chuyển qua pha 2.")
        tableau = SolvePhase2(tableau, m)
        
        # Check if all elements in the last row of Phase 2 tableau are non-negative
        if all(num >= 0 for num in tableau[-1][:-1]):
            print("Nghiệm của bài toán:")
            for i in range(n + m):  
                found = False
                for j in range(len(tableau) - 1):
                    if tableau[j][i] == 1:
                        found = True
                        if i < m:  
                            print(f"x{i+1} = {tableau[j][-1]}")  
                            break
                if not found:
                    if i < m:  
                        print(f"x{i+1} = 0")  
                
            # Calculate initial objective function value
            if problem_type == 'max':
               z_value = -1*tableau[-1][-1]
            else:
               z_value = 1*tableau[-1][-1]
            print(f"Giá trị hàm mục tiêu tối ưu: {z_value:.2f}")
        else:
            print("Bài toán vô nghiệm")
    else:
        print("Bài toán vô nghiệm")

# Dữ liệu test
m, c = InputObjectiveFunction()
# Input coefficients of constraints and constraint operators
n, a, operators, b = InputConstraints()
print()
# Input problem type (maximize or minimize)
problem_type = InputProblemType()
conditions= InputObjectiveFunctionConditions(m)
PrintObjectiveFunction(c, problem_type)
PrintConstraints(a, operators, b,conditions)
a, b, c, n, m,conditions  = convert_to_standard_form(a, b, c, n, m, problem_type, operators,conditions)
current_c = None
if n is not None and n > 0:
    current_c = [0.0] * (1) + [0.0] * (n - 1)
else:
    st.error("Số lượng ràng buộc phải là một số nguyên dương.")
print_phase1_problem(a, b, current_c, n, m)
 # Chuyển đổi sang dạng bài toán bổ trợ
tableau, num_variables = convert_to_phase1_form_x0(a, b, c, n, m)
 # In bài toán đã chuyển đổi
 #Cập nhật bảng đơn hình và giải quyết bài toán pha 1
convert_to_equations_x0(a, b, c, n, m,conditions)
Solve(tableau, m,n)
