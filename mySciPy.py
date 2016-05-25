__author__ = "Phillips0616"

####Description####
#matrices are of the form [[row1],[row2],...,[rowN]]
#vectors are of the form [[vector]], they are treated as a matrix with one row
####################


#takes a vector and a scalar value to scale the vector by
#returns the resulting scaled vector
def scalar_mult_vector(scale, vector):

    scaled = [[]]
    for entry in vector[0]:
        scaled[0].append(scale * entry)

    return scaled

#takes a matrix and a scalar value to scale the matrix by
#returns the resulting scaled matrix
def scale_matrix(scale, matrix):

    scaled = [[0 for a in range(len(matrix[0]))] for x in range(len(matrix))]

    for column_num in range(len(scaled)):
        for row_num in range(len(scaled[0])):
            scaled[column_num][row_num] = scale*matrix[column_num][row_num]

    return scaled

#takes two vectors that are to be summed
#returns the result of the vector addition
def vector_add(vectora, vectorb):
    sum = [[]]
    for r in range(len(vectora[0])):
        sum[0].append(vectora[0][r] + vectorb[0][r])
    return sum

#takes two vectors, vectorb is subtracted from vectora
#returns the resulting vectorc
def vector_subtract(vectora, vectorb):
    result = [[]]
    for r in range(len(vectora[0])):
        result[0].append(vectora[0][r] - vectorb[0][r])
    return result

#takes number of rows and columns of desired identity matrix
#returns identity matrix of indicated size
def create_identity(num_rows, num_columns):

    identity = [[0 for a in range(num_rows)] for x in range(num_columns)]

    for column_num in range(num_columns):
        for row_num in range(num_rows):
            if column_num == row_num:
                identity[column_num][row_num] = 1

    return identity

#takes two matrices to be added together
#returns the result of the matrix addition
def add_matricies(matrixa, matrixb):
    result = [[0 for a in range(len(matrixa[1]))] for x in range(len(matrixa))]

    for column_num in range(len(matrixa)):
        for row_num in range(len(matrixa[1])):
            result[column_num][row_num] = matrixa[column_num][row_num] + matrixb[column_num][row_num]
    return result

#takes two matrices to be subtracted
#subtracts matrixb from matrixa
#returns the result of the matrix subtraction
def subtract_matricies(matrixa, matrixb):
    result = [[0 for a in range(len(matrixa[1]))] for x in range(len(matrixa))]

    for column_num in range(len(matrixa)):
        for row_num in range(len(matrixa[1])):
            result[column_num][row_num] = matrixa[column_num][row_num] - matrixb[column_num][row_num]
    return result

#takes a matrix to compute the transpose of
#returns the transpose of the matrix
def matrix_transpose(matrix):

    result = [[0 for a in range(len(matrix))] for x in range(len(matrix[0]))]
    for column_num in range(len(matrix[0])):
        for row_num in range(len(matrix)):
            result[column_num][row_num] = matrix[row_num][column_num]
    return result

#takes a vector to compute the transpose of
#returns the transpose of the vector
def vector_transpose(vector):
    if len(vector) == 1:
        result = []
        for entry in vector[0]:
            result.append([entry])
    else:
        result = [[]]
        for entry in vector:
            result[0].append(entry[0])
    return result

#takes two vectors to compute the dot product of
#returns the resulting dot product
def dot_product(vectora, vectorb):
    result = 0
    for r in range(len(vectora)):
        result += vectora[r] * vectorb[r]
    return result

#takes a 2x2 matrix
#returns the determinant of the 2x2 matrix
def determinant_2x2(matrix):
    if len(matrix) > 2 or len(matrix[0]) > 2:
        print("Improper dimensions for calculating 2x2 determinant")
    else:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]

        return a*d - b*c

#accepts two matrices to be mutiplied
#returns the result of matrix multiplication
def mult_matrices(matrixa, matrixb):

    result = [[0 for a in range(len(matrixb[0]))] for x in range(len(matrixa))]

    for row_num in range(len(matrixa)):
        for col_num in range(len(matrixb[0])):
            column = [] #create current column of matrixb
            for entry in matrixb:
                column.append(entry[col_num])
            result[row_num][col_num] = dot_product(matrixa[row_num],column)
    return result

#takes a matrix and finds covariance between variables
#returns covariance matrix
def covariance(matrix):
    sum = [[1 for a in range(len(matrix))] for x in range(len(matrix))]
    difference = subtract_matricies(matrix, scale_matrix(1/len(matrix),mult_matrices(sum,matrix)))
    product = mult_matrices(matrix_transpose(difference), difference)
    scaled = scale_matrix(1 / len(matrix), product)
    return scaled
    
#finds pivot placement for Gauss-Jordan elimination
#checks for largest absolute value in a column which will be the pivot index
#returns pivot index
def find_pivot(matrix):
    largest = 0
    pivot = 0
    for r in range(len(matrix)):
        if abs(matrix[r][0]) > abs(largest):
            largest = matrix[r][0]
            pivot = r
    return pivot

#interchanges two rows in a matrix for Gauss-Jordan elimination
#returns resulting matrix after row interchange
def interchange_rows(matrix, row_to_switch, row ):
    temp = matrix[row_to_switch]
    matrix[row_to_switch] = matrix[row]
    matrix[row] = temp
    return matrix

#returns the diagional entries of a matrix as a list
def get_diag(matrix):
    diag = []
    for r in range(len(matrix)):
        for s in range(len(matrix[r])):
            if r == s:
                diag.append(matrix[r][s])
                break
    return diag

#calculates the trace of a matrix
#returns the trace as an integer or float
def trace(matrix):
    diag = get_diag(matrix)
    sum = 0
    for entry in diag:
        sum += entry
    return sum
