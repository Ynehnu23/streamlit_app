from rich.console import Console
from rich.table import Table
EPS = 1e-4
class Position:
    def __init__(self, row, column):
        self.row = row
        self.column = column
def ReadEquation():
    n, m = map(int, input().split())
    a = []
    for row in range(n):
        a.append(list(map(float, input().split())))
    b = list(map(float, input().split()))
    c = list(map(float, input().split()))
    return a, b, c, n, m

def InputObjectiveFunction():
    m = int(input("Nhập số lượng biến của hàm mục tiêu: "))
    print("Nhập hệ số của các biến trong hàm mục tiêu:")
    c = []
    for i in range(m):
        coefficient = float(input(f"Nhập hệ số của x{i+1}: "))
        c.append(coefficient)
    return m, c

def InputConstraints():
    print("Nhập số lượng ràng buộc:")
    n = int(input("Nhập số lượng ràng buộc:"))
    a = []
    b = []
    operators = []
    print("Nhập ràng buộc:")
    for i in range(n):
        constraint = []
        print(f"Nhập hệ số của các biến trong ràng buộc thứ {i+1}:")
        for j in range(m):
            coefficient = float(input(f"Nhập hệ số của x{j+1}: "))
            constraint.append(coefficient)
        a.append(constraint)
        print(a)
        operator = input("Nhập toán tử ràng buộc (>=, <=, =): ")
        operators.append(operator)
        print(operators)
        value = float(input("Nhập hệ số sau toán tử ràng buộc: "))
        b.append(value)
        print(b)
    return n, a, operators, b

def InputProblemType():
    problem_type = input("Bạn muốn tìm max hay min (max/min): ")
    return problem_type
def InputObjectiveFunctionConditions(m):
    conditions = []
    for i in range(m):
        condition = input(f"Nhập điều kiện của x{i+1} (<= 0, >= 0, tùy ý): ")
        conditions.append(condition)
    return conditions
def PrintObjectiveFunction(c,problem_type):
    print("Tìm giá trị", end=" ")
    if problem_type == "max":
        print("lớn nhất" if problem_type == "max" else "nhỏ nhất", "của hàm:")
    elif problem_type == "min":
        print("nhỏ nhất" if problem_type == "min" else "lớn nhất", "của hàm:")
    print("z =", end=" ")
    for i in range(len(c)):
        print(f"{c[i]}x{i+1}", end="")
    print()

def PrintConstraints(a, operators, b, conditions):
    print("Thỏa mãn các ràng buộc sau:")
    for i in range(len(a)):
        constraint_string = " + ".join([f"{a[i][j]}x{j+1}" for j in range(len(a[i]))])
        constraint_string += f" {operators[i]} {b[i]}"
        print(constraint_string)
    for i, condition in enumerate(conditions):
        print(f"x{i+1} {condition}")
    print("---------------------------------------------")
def PrintProblem(c, problem_type, a, operators, b, conditions):
    print("Tìm giá trị", end=" ")
    if problem_type == "max":
        print("lớn nhất" if problem_type == "max" else "nhỏ nhất", "của hàm:")
    elif problem_type == "min":
        print("nhỏ nhất" if problem_type == "min" else "lớn nhất", "của hàm:")
    print("z =", end=" ")
    for i in range(len(c)):
        print(f"{c[i]}x{i+1}", end="")
    print()

    print("Thỏa mãn các ràng buộc sau:")
    for i in range(len(a)):
        constraint_string = " + ".join([f"{a[i][j]}x{j+1}" for j in range(len(a[i]))])
        constraint_string += f" {operators[i]} {b[i]}"
        print(constraint_string)
    for i, condition in enumerate(conditions):
        print(f"x{i+1} {condition}")
    print("---------------------------------------------")
def convert_to_standard_form(a, b, c, n, m, problem_type, operators, conditions):
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
    print(new_c)
    for constraint in result_strings:
        print(constraint)

    # Xóa các dòng trong a có số lượng phần tử khác với new_m
    new_a = [row for row in new_a if len(row) == len(new_c)]
    return new_a, new_b, new_c, len(new_b), len(new_c), new_conditions
def convert_to_equations(a, b, c, n, m, conditions):
    print("Lập từ vựng xuất phát:")
    print("z =", end=" ")
    first_arbitrary_variable = -1
    # Khởi tạo hàm mục tiêu dựa trên conditions
    for k in range(len(c)):
        # Nếu biến hiện tại là biến tùy ý
        if conditions[k] == "tùy ý":
            if first_arbitrary_variable == -1:  # Nếu đây là biến tùy ý đầu tiên
                first_arbitrary_variable = k
                print(f"{c[k]}*x{k+1}", end=" ")
                if k + 1 < len(conditions) and conditions[k + 1] == "tùy ý":  # Kiểm tra xem cặp hiện tại có chứa điều kiện "tùy ý" không
                    temp = first_arbitrary_variable % 2
                    if (k + 1) % 2 == temp:  # Kiểm tra vị trí của biến tùy ý
                        print(f" + {c[k]}*x{k+1}", end=" ")
                    else:
                        print(f" - {c[k]}*x{k+1}_t", end=" ")
        elif k == 0:
            print(f"{c[k]}*x{k+1}", end=" ")
        else:
            print(f"{c[k]}*x{k}", end=" ")
        
        if k < len(c) - 1:
            print("+", end=" ")
    
    print()
    equations = []
    
    for i in range(len(a)):
        first_arbitrary_variable = -1
        if i < len(b):
            equation = f"w{i+1} = {b[i]}"
            for j in range(len(a[i])):
                if a[i][j] != 0:
                    if conditions[j] == "tùy ý":
                        # Kiểm tra xem cặp hiện tại có chứa điều kiện "tùy ý" không
                        if first_arbitrary_variable == -1:  # Nếu đây là biến tùy ý đầu tiên
                            first_arbitrary_variable = j                     
                            equation += f" {'-' if a[i][j] > 0 else '+'} {abs(a[i][j])}x{j+1}"
                            if j + 1 < len(conditions) and conditions[j + 1] == "tùy ý":
                                temp = first_arbitrary_variable % 2
                                if (k + 1) % 2 == temp:  # Kiểm tra vị trí của biến tùy ý
                                   equation += f" {'-' if a[i][j] > 0 else '+'} {abs(a[i][j])}x{j+1}"
                                
                                else:
                                   equation += f" {'-' if a[i][j] > 0 else '+'} {abs(a[i][j])}x{j+1}_t"
                    elif j == 0 or j == 1:
                        equation += f" {'-' if a[i][j] > 0 else '+'} {abs(a[i][j])}x{j+1}"
                    else:
                        equation += f" {'-' if a[i][j] > 0 else '+'} {abs(a[i][j])}x{j}"
            equations.append(equation)
    
    # In ra các phương trình đã được chuyển đổi
    for equation in equations:
        print(equation)
    print("---------------------------------------------")
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

def print_tableau(tableau, m, n, conditions):
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
def SelectPivotElement_dual(tableau):
    num_columns = len(tableau[0])
    num_rows = len(tableau) - 1

    # Step 1: Find the pivot row
    pivot_row = None
    min_value = float('inf')
    for i in range(num_rows):
        if tableau[i][-1] < 0 and tableau[i][-1] < min_value:
            min_value = tableau[i][-1]
            pivot_row = i
    
    if pivot_row is None:
        raise ValueError("No negative value found in the last column. No pivot row can be selected.")

    # Step 2: Find the pivot column
    pivot_column = None
    min_ratio = float('inf')
    for j in range(num_columns - 1):
        if tableau[pivot_row][j] > 0 and tableau[-1][j] > 0:
            ratio = tableau[-1][j] / tableau[pivot_row][j]
            if ratio < min_ratio:
                min_ratio = ratio
                pivot_column = j
    
    if pivot_column is None:
        raise ValueError("No valid pivot column found. No pivot element can be selected.")

    return Position(pivot_row, pivot_column)
def ProcessPivotElement_dual(a, pivot_element):
    # Get the absolute value of the pivot element
    pivot_element_value = a[pivot_element.row][pivot_element.column]

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
                break
    # Apply division by the absolute value of the pivot element
    a[pivot_element.row] = [abs(n) / pivot_element_value for n in a[pivot_element.row]]

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

def SolveEquation(a, b, c, n, m, conditions, problem_type):
    # Tạo bảng Simplex từ phương trình
    tableau = CreateTableau(a, b, c, n, m)
    iteration = 1

    while True:
        print("Bước №", iteration)
        print_tableau(tableau, m, n, conditions)
        print("---------------------------------------------")

        # Kiểm tra xem có giá trị âm nào trong cột cuối cùng không
        if any(row[-1] < 0 for row in tableau[:-1]):
            pivot_element = SelectPivotElement_dual(tableau)
            if pivot_element is None:
                print("Không tìm thấy phần tử chốt hợp lệ.")
                break
            print("Phần tử chốt đã chọn:", pivot_element.row, ",", pivot_element.column, " có giá trị là:",
             f"{tableau[pivot_element.row][pivot_element.column]:.2f}") 
            print("---------------------------------------------")
            # Xử lý phần tử pivot trong bảng
            print("Bảng sau khi xoay từ vựng:")
            tableau = ProcessPivotElement_dual(tableau, pivot_element)
        else:
            if all(value >= 0 for value in tableau[-1][:-1]) and all(row[-1] >= 0 for row in tableau[:-1]):
                break
            pivot_element = SelectPivotElement(tableau)
            print("Phần tử chốt đã chọn:", pivot_element.row, ",", pivot_element.column, " có giá trị là:",
             f"{tableau[pivot_element.row][pivot_element.column]:.2f}") 
            
            if pivot_element is None:
                print("Không tìm thấy phần tử chốt hợp lệ.")
                break
            # Xử lý phần tử pivot trong bảng
            print("Bảng sau khi xoay từ vựng:")
            tableau = ProcessPivotElement(tableau, pivot_element)

        # In ra bảng sau khi xử lý
        print_tableau(tableau, m, n, conditions)

        iteration += 1

    print("---------------------------------------------")
    print("Bảng cuối cùng:")
    # In ra bảng cuối cùng
    print_tableau(tableau, m, n, conditions)
    print("---------------------------------------------")
    
    # Xác định câu trả lời từ bảng cuối cùng
    ans = determine_answer(tableau,m,n)
    
    if check_unbounded(tableau):
    
        print("Bài toán không giới nội.")
        if problem_type == "max":
            print("Giá trị tối ưu là dương vô hạn (+∞).")
        elif problem_type == "min":
            print("Giá trị tối ưu là âm vô hạn (-∞).")
        return
    else:
        # Kiểm tra điều kiện cho nghiệm
        valid_solution = True
        for i in range(m):
            if conditions[i] == ">= 0" and ans[i] < 0:
                valid_solution = False
                print(f"Giá trị x{i+1} không thỏa điều kiện >= 0.")
            elif conditions[i] == "<= 0" and ans[i] > 0:
                valid_solution = False
                print(f"Giá trị x{i+1} không thỏa điều kiện <= 0.")
            elif conditions[i] == "tùy ý":
                print(f"Giá trị x{i+1} có thể nhận bất kỳ giá trị nào.")
        
        if valid_solution:
            print("Giải pháp có giới hạn")
            print("Giải pháp tối ưu:")
            for i in range(m):
                print(f"x{i+1} = {ans[i]}")
            
            optimal_solution = [ans[i] for i in range(m)]
            if problem_type == "max":
                optimal_value = -1 * sum(optimal_solution[i] * c[i] for i in range(m))
            else:
                optimal_value = sum(optimal_solution[i] * c[i] for i in range(m))
            
            optimal_value = round(optimal_value, 2)
            print("Giá trị tối ưu:", optimal_value)


def determine_answer(tableau,m,n):
    ans = [0] * m
    for i in range(n):
        count_ones = sum(1 for j in range(m) if tableau[i][j] == 1)
        if count_ones == 1:
            for j in range(m):
                if tableau[i][j] == 1:
                    ans[j] = tableau[i][-1]
        elif count_ones == 2:  # Nếu có hai cột chỉ chứa số 0 và số 1
            first_one_column = None
            for j in range(m):
                if tableau[i][j] == 1:
                    if first_one_column is None:
                        first_one_column = j
                    else:
                        # Lấy giá trị từ cột đầu tiên
                        ans[first_one_column] = tableau[i][-1]
                        break
    ans = [round(num, 2) for num in ans]
    # Kiểm tra nếu không có nghiệm thỏa mãn (không giới nội)
    if any(num < 0 for num in ans):
        return [-1] * m  # Trả về danh sách các giá trị -1 để biểu thị không có nghiệm
    else:
        return ans
if __name__ == "__main__":
    # Input coefficients of objective function
    m, c = InputObjectiveFunction()
    # Input coefficients of constraints and constraint operators
    n, a, operators, b = InputConstraints()
    conditions=InputObjectiveFunctionConditions(m)
    # Input problem type (maximize or minimize)
    problem_type = InputProblemType()
    PrintObjectiveFunction(c, problem_type)
    PrintConstraints(a, operators, b,conditions)
    a, b, c, n, m ,conditions =    convert_to_standard_form(a, b, c, n, m, problem_type, operators, conditions)
    convert_to_equations(a, b, c, n, m, conditions)
    SolveEquation(a, b, c, n, m,conditions,problem_type)
    