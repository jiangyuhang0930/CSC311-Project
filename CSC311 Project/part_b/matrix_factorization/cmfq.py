## TODO Implement Coupled Matrix Factorization by Question

from utils import *
import numpy as np
import matplotlib.pyplot as plt

def create_sparse_matrix(data, shape, correctness=1):
    """ Create a sparse matrix representation of data with specified shape
    and axes. If col_label is None, create the correctness
    matrix.
    Note that the shape is (x, num_questions)

    """

    matrix = np.ones(shape) * np.nan

    if correctness: ## Correctness matrix
        for i, q in enumerate(data["question_id"]):
            u = data["user_id"][i]
            c = data["is_correct"][i]
            matrix[u][q] = c
    else:
        for i, q in enumerate(data["question_id"]):
            s_lst = data["subject_id"][i]
            ## Create vector
            col = np.zeros(shape[0])
            col[s_lst] = 1
            matrix.T[q] = col

    return matrix

def single_squared_error_loss(u, z, matrix):
    """ Return the squared-error-loss given the matrix and factors.
    :param matrix: Sparse matrix representation for data
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    ## Replace nan with 0 in matrix
    matrix_cpy = np.nan_to_num(matrix)

    ## Get matrix prediction and delete all the nans
    pred_matrix = u.dot(z.T)
    pred_matrix = np.where(np.isnan(matrix), 0, pred_matrix)

    loss = np.sum((matrix_cpy - pred_matrix) ** 2)

    return 0.5 * loss


def total_squared_error_loss(z, d_1, d_2, matrix, subject_matrix):
    """Calculate the total squared error loss
    """
    loss = 0
    D = [d_1, d_2]
    matrices = [matrix].append(subject_matrix)
    for i in range(len(D)):
        loss += single_squared_error_loss(D[i], z, matrices[i])
    return loss

def update_d_z_partial(data, matrix, d, z, lr, correctness=0):
    """ Return the partially updated d and z from stochastic gradient descent
    for matrix completion (single sum)

    :param data: A dictionary
    :param matrix: 2D matrix of data
    :param d: 2D matrix
    :param z: 2D matrix
    :param lr: float
    :param correctness: int
    :return: d, z
    """
    # Randomly select a pair (x, question_id).
    i = np.random.choice(len(data["question_id"]), 1)[0]

    ## Get corresponding indices
    m = data["question_id"][i]
    if correctness:
        n = data["user_id"][i]
    else:
        n = np.random.choice(data["subject_id"][i], 1)[0]

    ## Get correct value
    x = matrix[n][m]

    ## Stochastic gradient descent
    dd = -1 * (x - np.dot(d[n], z[m])) * z[m]
    d[n] -= lr * dd
    dz = -1 * (x - np.dot(d[n], z[m])) * d[n]
    z[m] -= lr * dz

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return d, z


def update_d_z(train_data, train_matrix, question_data, subject_matrix, lr,
               z, d_1, d_2):
    """ Return the updated u and d_i's after applying stochastic gradient descent
        for matrix completion.

        :param train_data: A dictionary
        :param train_matrix: 2D matrix of train_data
        :param question_data: Dictionary of question metadata
        :param subject_matrix: Matrix of question metadata
        :param lr: float
        :param z: 2D matrix
        :param d_1: 2D matrix
        :param d_2: 2D matrix
        :return: (z, d_1, d_2)
        """

    ## Update based on correctness
    d_1, z = update_d_z_partial(data=train_data, matrix=train_matrix, d=d_1,
                                 z=z, lr=lr, correctness=1)
    d_2, z = update_d_z_partial(data=question_data, matrix=subject_matrix, d=d_2,
                          z=z, lr=lr)

    return z, d_1, d_2

def als(train_data, question_data, train_matrix, subject_matrix,
        k, lr, num_iteration, nums):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param question_data: A dictionary of question metadata
    :param train_matrix: Sparse matrix representation of training data
    :param subject_matrix: Sparse matrix for question metadata
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param nums: [num_questions, num_students, num_subjects]
    :return: 2D reconstructed Matrix.
    """
    # Initialize z, d_1, d_2
    ## Question
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(nums[0], k))
    ## User
    d_1 = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(nums[1], k))
    ## Subject
    d_2 = np.random.uniform(low=0, high=1 / np.sqrt(k),
                            size=(nums[2], k))
    #####################################################################
    for i in range(num_iteration):
        z, d_1, d_2 = update_d_z(train_data=train_data, train_matrix=train_matrix,
                                question_data=question_data, subject_matrix=subject_matrix,
                                lr=lr, z=z, d_1=d_1, d_2=d_2)

    mat = d_1.dot(z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def main():
    ## Load data
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    question_data = load_question_meta("../data")
    subject_data = load_subject_meta("../data")

    num_students = train_matrix.shape[0]
    num_questions = train_matrix.shape[1]
    num_subjects = len(subject_data["subject_id"])
    nums = [num_questions, num_students, num_subjects]

    ## Create sparse matrix representations of the data
    val_matrix = create_sparse_matrix(data=val_data,
                                      shape=(nums[1], nums[0]))
    test_matrix = create_sparse_matrix(data=test_data,
                                      shape=(nums[1], nums[0]))
    subject_matrix = create_sparse_matrix(data=question_data,
                                      shape=(nums[2], nums[0]), correctness=0)

    ## Hyperparameters -- need to tune!!!
    k = 10
    num_iteration = 2000
    lr = 0.05

    reconst_matrix = als(train_data=train_data, question_data=question_data,
                         train_matrix=train_matrix, subject_matrix=subject_matrix,
                        k=k, lr=lr, num_iteration=num_iteration, nums=nums)

    val_acc = evaluate_matrix(data_matrix=val_matrix,
                              reconst_matrix=reconst_matrix)
    test_acc = evaluate_matrix(data_matrix=test_matrix,
                               reconst_matrix=reconst_matrix)

    print("Final Validation Accuracy: ", val_acc)
    print("Final Test Accuracy: ", test_acc)

if __name__ == '__main__':
    main()

