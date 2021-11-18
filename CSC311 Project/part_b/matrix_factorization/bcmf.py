'''
Biased coupled matrix factorization (with context; no global bias)
BAD
'''

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.linalg import sqrtm

def concatenate_student_meta(student_data, num_students):
    """Concatenate student metadata to the correctness matrix
    :param student_data: tuple of student metadata dictionaries
            [age, gender, premium_pupil]
    """
    student_matrix = np.ones((num_students, 3)) * np.nan
    ## Age
    for i, u in enumerate(student_data[0]["user_id"]):
        student_matrix[u][0] = student_data[0]["age"][i]

    ## Gender
    for i, u in enumerate(student_data[1]["user_id"]):
        student_matrix[u][1] = student_data[1]["gender"][i]

    ## Premium_pupil
    for i, u in enumerate(student_data[2]["user_id"]):
        student_matrix[u][2] = student_data[2]["premium_pupil"][i]

    ## Concatenate
    return student_matrix

def impute_matrix_values(matrix):
    """Return a new matrix with filled values being the means
    """
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    return masked_matrix.filled(item_means)


def create_sparse_matrix(data, shape, create_subject=0):
    """ Create a sparse matrix representation of data with specified shape
    and axes. If col_label is None and student=1, create the correctness
    matrix.

    :param data: Dictionary of data
    :param shape: Tuple of matrix shape
    :param is_question: Whether we are creating the subject matrix
    """
    matrix = np.ones(shape) * np.nan

    if create_subject: ## Subject matrix
        for i, s_lst in enumerate(data["subject_id"]):
            q = data["question_id"][i]
            ## Create vector
            col = np.zeros(shape[0])
            col[s_lst] = 1
            matrix.T[q] = col
    else: ## Correctness
        for i, u in enumerate(data["user_id"]):
            q = data["question_id"][i]
            c = data["is_correct"][i]
            matrix[u][q] = c

    return matrix


def update_u_z(train_matrix, s, u, z, lr, biases):
    """ Return the updated u and z from stochastic gradient descent
    for matrix completion.

    :param train_data: train dictionary
    :param u: 2D matrix
    :param z: 2D matrix
    :param lr: float
    :param lambda_list: list of weight-penalizing coefficients
    :param biases: list of bias vectors
    :return the updated u and z
    """
    ## Randomly select (student, question) pair
    for i in range(10):
        n = np.random.choice(range(train_matrix.shape[0]), 1)[0]
        m = np.random.choice(range(train_matrix.shape[1]), 1)[0]
        c = train_matrix[n][m]

        ## Get biases
        b_s = biases[0][n]
        b_q = biases[1][m]

        ## Stochastic gradient descent
        du = -1 * (c - b_s - b_q - np.dot(u[n], z[m]) - np.dot(s[n], z[m])) * z[m]
        u[n] -= lr * du
        dz = -1 * (c - b_s - b_q - np.dot(u[n], z[m]) - np.dot(s[n], z[m])) * (u[n] + s[n])
        z[m] -= lr * dz

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def update_all(train_data, train_matrix, filled_train_matrix, filled_student_matrix,
               u, z, biases, lr):
    """ Update the prediction and biases

        :param train_data: dictionary
        :param train_matrix: original matrix from training data (with NaNs)
        :param filled_train_matrix: filled training matrix (no NaNs)
        :param biases: (student_bias, question_bias)
        :return: (filled_train_matrix, biases)
    """
    n = train_matrix.shape[0]
    m = train_matrix.shape[1]
    not_nan = 1 - np.isnan(train_matrix)
    full_mat = np.where(not_nan == 1, train_matrix, filled_train_matrix)

    ## Update u and z (user and question latent matrices)
    u, z = update_u_z(train_matrix=full_mat, s=filled_student_matrix,
                      u=u, z=z, lr=lr, biases=biases)
    avg_mat = u.dot(z.T)
    stu_mat = np.dot(filled_student_matrix, z.T)

    ## Update local biases
    biases[0] = full_mat - avg_mat - stu_mat - np.dot(np.ones((n, 1)), biases[1].T)
    biases[0] = biases[0].dot(np.ones((m, 1))) / m

    biases[1] = full_mat - avg_mat - stu_mat - np.dot(biases[0], np.ones((1, m)))
    biases[1] = biases[1].T.dot(np.ones((n, 1))) / n

    ## Update filled matrices
    filled_train_matrix = avg_mat + stu_mat + np.dot(biases[0], np.ones((1, m))) \
                          + np.dot(np.ones((n, 1)), biases[1].T)

    return filled_train_matrix, biases


def matrix_factorization(train_data, train_matrix, student_matrix, k, lr, num_iteration,
                         val_matrix=None):
    """ Perform matrix factorization algorithm. Return reconstructed matrix.

    :param train_matrix: Sparse matrix representation of training + student metadata
    :param k: int
    :param num_iteration: int
    :param val_matrix: validation matrix
    :return: 2D reconstructed Matrix.
    """

    # Impute missing values of correctness matrix
    filled_train_matrix = impute_matrix_values(train_matrix)

    # Initialize u, z, biases
    num_students = train_matrix.shape[0]
    num_questions = train_matrix.shape[1]

    ## Student
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(num_students, k))
    ## Question
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(num_questions, k))

    ## Student bias, question bias
    biases = [np.zeros((num_students, 1)), np.zeros((num_questions, 1))]

    #####################################################################
    val_accs = []
    for i in range(num_iteration):
        filled_train_matrix, biases = update_all(train_data=train_data,
                                                 train_matrix=train_matrix,
                                            filled_train_matrix=filled_train_matrix,
                                            filled_student_matrix=filled_student_matrix,
                                            u=u, z=z, biases=biases, lr=lr)

        if val_matrix is not None:
            not_nan = 1 - np.isnan(train_matrix)
            pred_mat = np.where(not_nan == 1, train_matrix, filled_train_matrix)
            val_acc = evaluate_matrix(data_matrix=val_matrix, reconst_matrix=pred_mat)
            val_accs.append(val_acc)

            print(i, val_acc)

    ## Final prediction (ensures original values in training set not changed)
    not_nan = 1 - np.isnan(train_matrix)
    mat = np.where(not_nan == 1, train_matrix, filled_train_matrix)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, val_accs


def main():
    ## Load data
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    student_data = load_student_meta("../data") ## student_data is a tuple of metadata dictionaries
    question_data = load_question_meta("../data")
    subject_type_data = load_subject_meta("../data")

    num_students = train_matrix.shape[0]
    num_questions = train_matrix.shape[1]
    num_subjects = len(subject_type_data["subject_id"])

    ## Create sparse matrix representations of the data
    val_matrix = create_sparse_matrix(data=val_data,
                                      shape=(num_students, num_questions))
    test_matrix = create_sparse_matrix(data=test_data,
                                      shape=(num_students, num_questions))
    subject_matrix = create_sparse_matrix(data=question_data,
                                           shape=(num_subjects, num_questions), create_subject=1)

    ## Student meta
    student_matrix = concatenate_student_meta(student_data, num_students)

    num_iteration = 300

    ## TODO: Grid search to tune hyperparameters
    ## TODO: Plots of loss and accuracy vs. iteration for training and validation
    ## TODO: Print final validation and test accuracies

    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")
    k = 3
    lr = 0.1
    ## Plot coupled mf with no subject metadata
    reconst_matrix, val_accs = matrix_factorization(train_data = train_data,
                                    train_matrix=train_matrix, student_matrix=student_matrix,
                                    k=k, lr=lr,
                                   num_iteration=num_iteration,
                                   val_matrix=val_matrix)

    ## Plot validation accuracy
    plt.plot(range(num_iteration), val_accs)

    val_acc = evaluate_matrix(data_matrix=val_matrix,
                              reconst_matrix=reconst_matrix)

    print("***************************************")
    print("Val Accuracy: {}".format(val_acc))


    # plt.legend()
    plt.savefig('graphs/biased_mf_svd')

if __name__ == '__main__':
    main()
