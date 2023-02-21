import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

#Question 1

def neville(x, y, x0):

    n = len(x)
    p = [0] * n
    
    for k in range(n):
        for i in range(n-k):
            if k == 0:
                p[i] = y[i]
            else:
                p[i] = ((x0 - x[i+k]) * p[i] + (x[i] - x0) * p[i+1]) / (x[i] - x[i+k])
                
    return p[0]

x = np.array([3.6,3.8,3.9])
y = np.array([1.675,1.436,1.318])

p = neville(x, y, 3.7)

print(p) 
print("")

#Question 2

def divided_difference_table(x_points, y_points):
    # set up the matrix
    size: int = len(x_points)
    matrix: np.array = np.zeros((size,size))

    # fill the matrix
    for index, row in enumerate(matrix):
        row[0] = y_points[index]

    # populate the matrix (end points are based on matrix size and max operations we're using)
    for i in range(1, size):
        for j in range(1, i+1):
            # the numerator are the immediate left and diagonal left indices...
            numerator = matrix[i][j-1] - matrix[i-1][j-1]

            # the denominator is the X-SPAN...
            denominator = x_points[i] - x_points[i-j]

            operation = numerator / denominator

            # cut it off to view it more simpler
            matrix[i][j] = '{0:.17g}'.format(operation)
    return matrix


def get_approximate_result(matrix, x_points, value):
    # p0 is always y0 and we use a reoccuring x to avoid having to recalculate x 
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0][0]
    
    # we only need the diagonals...and that starts at the first row...
    for index in range(1, len(x_points)):
        polynomial_coefficient = matrix[index][index]

        # we use the previous index for x_points....
        reoccuring_x_span *= (value - x_points[index-1])
        
        # get a_of_x * the x_span
        mult_operation = polynomial_coefficient * reoccuring_x_span

        # add the reoccuring px result
        reoccuring_px_result += mult_operation

    
    # final result
    return reoccuring_px_result
    
x_points = [7.2, 7.4, 7.5, 7.6]
y_points = [23.5492, 25.3913, 26.8224 , 27.4589 ]
divided_table = divided_difference_table(x_points, y_points)
print('[', divided_table[1][1], ', ', divided_table[2][2], ', ', divided_table[3][3], ']', sep='')
    
# find approximation
approximating_x = 7.3
final_approximation = get_approximate_result(divided_table, x_points, approximating_x)
print("")
print(final_approximation)
print("")

#Question 3
np.set_printoptions(precision=7, suppress=True, linewidth=100)

def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            # skip if value is prefilled (we dont want to accidentally recalculate...)
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            
            # get left cell entry
            left: float = matrix[i][j-1]

            # get diagonal left entry
            diagonal_left: float = matrix[i-1][j-1]

            # order of numerator is SPECIFIC.
            numerator: float = (left - diagonal_left)

            # denominator is current i's x_val minus the starting i's x_val....
            denominator = matrix[i][0] - matrix[i-j+1][0]

            # something save into matrix
            operation = numerator / denominator
            matrix[i][j] = operation
    
    return matrix


def hermite_interpolation():
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436 , 1.318 ]
    slopes = [-1.195 , -1.188 , -1.182 ]

    # matrix size changes because of "doubling" up info for hermite 
    num_of_points = len(x_points)
    matrix = np.zeros((2*num_of_points, 2*num_of_points))

    # populate x values (make sure to fill every TWO rows)
    for i in range(num_of_points):
        matrix[2*i][0] = matrix[2*i+1][0] = x_points[i]
    
    # prepopulate y values (make sure to fill every TWO rows)
    for i in range(num_of_points):
        matrix[2*i][1] = y_points[i]
        matrix[2*i+1][1] = y_points[i]
    
    # prepopulate with derivates (make sure to fill every TWO rows. starting row CHANGES.)
    for i in range(num_of_points):
        matrix[2*i+1][2] = slopes[i]

    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix)



hermite_interpolation()
print("")


from scipy.interpolate import CubicSpline
# Input data 
x = np.array([2, 5,8, 10])
y = np.array([3, 5, 7, 9])

#find constants
h0 = x[1] - x[0]
h1 = x[2] - x[1]
h2 = x[3] - x[2]

a0 = y[0]
a1 = y[1]
a2 = y[2]
a3 = y[3]

size = len(x)
matrix: np.array = np.zeros((size,size))

for i in range (0,size):
  for j in range(0,size):
    if(i == 0 and j == 0 or i == size-1 and j == size-1):
      matrix[i][j] = 1
    elif(i == 1 and j == 0 or i == 1  and j == 2 or i == 2 and j == 1):
      matrix[i][j] = h0
    elif(i == 1 and j == 1 ):
      matrix[i][j] = 2*(h0+h1)
    elif(i == 2 and j== 2):
      matrix[i][j] = 2*(h1+h2)
    elif(i == 2 and j == 3):
      matrix[i][j] = h2

print(matrix)
print("")
vectorB: np.array = np.zeros(size)
vectorB[0] = 0
vectorB[1] =((3/h1)*(a2-a1))-((3/h0)*(a1-a0))
vectorB[2] =((3/h2)*(a3-a2))-((3/h1)*(a2-a1))
vectorB[0] = 0

print(vectorB)
print("")


vectorX: np.array = np.zeros(size)
vectorX = np.linalg.inv(matrix) * vectorB

newVectorX : np.array = np.zeros(size)


newVectorX[0] = vectorX[0][2]
newVectorX[1] = vectorX[1][2]
newVectorX[2] = vectorX[2][2]
newVectorX[3] = vectorX[3][2]

print(newVectorX)
print("")
