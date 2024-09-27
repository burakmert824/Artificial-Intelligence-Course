def read_sudoku_file(file_path):
    sudokus = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_sudoku = []
    
    for line in lines[4:-2]:
        line = line.strip()

        if line.startswith("SUDOKU"):
            if(len(current_sudoku) !=0 ):
                sudokus.append(current_sudoku)
            current_sudoku = []
        elif line:
            current_sudoku.append(list(map(int, line)))

    if current_sudoku is not None:
        sudokus.append(current_sudoku)

    return sudokus

def isValid(sudoku, row, col, num):
    #check row
    for i in range(9):
        if sudoku[row][i] == num:
            return False
    #check the column
    for i in range(9):
        if sudoku[i][col] == num:
            return False
        
    # Check 3x3
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if sudoku[start_row + i][start_col + j] == num:
                return False

    return True

def findEmpty(sudoku):
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                return (i, j)
    return (-1, -1)

def solve_sudoku(sudoku):
    #find empty place
    (row, col) = findEmpty(sudoku)

    # check if it is finished (no place to go)
    if row is -1 and col is -1:
        return True

    for num in range(1, 10):
        if isValid(sudoku, row, col, num):
            sudoku[row][col] = num

            # try new move with recursive
            if solve_sudoku(sudoku):
                return True

            # If it couldn't find any solution go back, backtrack
            sudoku[row][col] = 0

    # if couldn't fiind solution with given sudoku return false
    return False

file_path = 'Assignment_2_sudoku.txt'  
sudoku_puzzles = read_sudoku_file(file_path)

# Display the extracted Sudoku puzzles
for i, sudoku in enumerate(sudoku_puzzles, start=1):
    print(f"Sudoku {i}:")
    for row in sudoku:
        print(' '.join(map(str, row)))
    print("\n")
    
    print(f"Solution {i}:")
    solve_sudoku(sudoku)
    check = True
    #check every piece
    for i in range (9):
        for j in range(9):
            isValid(sudoku,i,j,sudoku[i][j])
    if(check):
        for row in sudoku:
            print(' '.join(map(str, row)))
    else:
        print("Answer is wrong")
    print("\n")
