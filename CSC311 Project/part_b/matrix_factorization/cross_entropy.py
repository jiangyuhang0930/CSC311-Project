'''
Biased matrix factorization with L2 regularization (no context or global bias)

'''
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import itertools

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

    if create_subject:  ## Subject matrix
        for i, s_lst in enumerate(data["subject_id"]):
            q = data["question_id"][i]
            ## Create vector
            col = np.zeros(shape[0])
            col[s_lst] = 1
            matrix.T[q] = col
    else:  ## Correctness
        for i, u in enumerate(data["user_id"]):
            q = data["question_id"][i]
            c = data["is_correct"][i]
            matrix[u][q] = c

    return matrix


def update_u_z(train_data, u, z, lr, lambda_list, biases):
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
    ## Randomly select (student, question) pairs
    indices = np.random.choice(range(len(train_data["user_id"])), 10)
    for i in indices:
        n = train_data["user_id"][i]
        m = train_data["question_id"][i]
        c = train_data["is_correct"][i]

        ## Get biases
        b_s = biases[0][n]
        b_q = biases[1][m]

        ## Stochastic gradient descent
        du = -1 * (c - sigmoid(b_s + b_q + np.dot(u[n], z[m]))) * z[m]
        u[n] = (1 - lambda_list[0] * lr) * u[n] - lr * du       ## With L2 regularization
        dz = -1 * (c - sigmoid(b_s + b_q + np.dot(u[n], z[m]))) * u[n]
        z[m] = (1 - lambda_list[1] * lr) * z[m] - lr * dz      ## With L2 regularization

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def update_all(train_data, train_matrix, filled_train_matrix, lr, u, z, biases, lambda_list):
    """ Update all the latent factors, biases and such

        :param train_data: data dictionary
        :param train_matrix: original matrix from training data (with NaNs)
        :param filled_train_matrix: filled training matrix (no NaNs)
        :param lr: float
        :param u: 2D matrix (latent student matrix)
        :param z: 2D matrix (latent question matrix)
        :param biases: (student_bias, question_bias)
        :param lambda_list: list of weight-penalizing coefficients
            [lambda_U, lambda_Z, lambda_student_bias, lambda_question_bias]
        :return: (u, z, biases, filled_train_matrix)
    """
    n = u.shape[0]
    m = z.shape[0]
    not_nan = 1 - np.isnan(train_matrix)
    full_mat = np.where(not_nan == 1, train_matrix, filled_train_matrix)

    ## Update u and z (user and question latent matrices)
    u, z = update_u_z(train_data=train_data, u=u, z=z, lr=lr, lambda_list=lambda_list,
                      biases=biases)

    avg_mat = u.dot(z.T)

    ## Update local biases
    # biases[0] = full_mat - avg_mat - np.dot(np.ones((n, 1)), biases[1].T)
    # biases[0] = biases[0].dot(np.ones((m, 1))) / (lambda_list[2] + m)
    biases[0] = (1 - lambda_list[2] * lr) * biases[0] + lr * \
                (full_mat - sigmoid(avg_mat + np.dot(biases[0], np.ones((1, m))) + np.dot(np.ones((n, 1)), biases[1].T))).dot(np.ones((m, 1))) / m

    # biases[1] = full_mat - avg_mat - np.dot(biases[0], np.ones((1, m)))
    # biases[1] = biases[1].T.dot(np.ones((n, 1))) / (lambda_list[3] + n)
    biases[1] = (1 - lambda_list[3] * lr) * biases[1] + lr * \
                (full_mat - sigmoid(avg_mat + np.dot(biases[0], np.ones((1, m))) + np.dot(np.ones((n, 1)), biases[1].T))).T.dot(np.ones((n, 1))) / n

    ## Update filled matrices
    filled_train_matrix = sigmoid(avg_mat + np.dot(biases[0], np.ones((1, m))) \
                          + np.dot(np.ones((n, 1)), biases[1].T))

    return u, z, biases, filled_train_matrix


def matrix_factorization(train_data, train_matrix, k, lr, num_iteration, lambda_list, val_matrix=None):
    """ Performs the biased matrix factorization algorithm with L2 regularization.
     Return reconstructed matrix.

    :param train_data: dictionary
    :param train_matrix: Sparse matrix representation of training + student metadata
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param lambda_list: list of weight-penalizing coefficients
        [alpha_U, alpha_Z, alpha_global_bias, alpha_student_bias, alpha_question_bias]
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
        u, z, biases, filled_train_matrix = update_all(train_data=train_data,
            train_matrix=train_matrix,
            filled_train_matrix=filled_train_matrix,
            lr=lr, u=u, z=z, biases=biases, lambda_list=lambda_list)

        if val_matrix is not None:
            not_nan = 1 - np.isnan(train_matrix)
            pred_mat = np.where(not_nan == 1, train_matrix, filled_train_matrix)
            val_acc = evaluate_matrix(data_matrix=val_matrix,
                                      reconst_matrix=pred_mat)
            val_accs.append(val_acc)
            print(val_acc)

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
    student_data = load_student_meta(
        "../data")  ## student_data is a tuple of metadata dictionaries
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
                                          shape=(num_subjects, num_questions),
                                          create_subject=1)

    num_iteration = 3000

    ## TODO: Grid search to tune hyperparameters
    ## TODO: Plots of loss and accuracy vs. iteration for training and validation
    ## TODO: Print final validation and test accuracies

    ## Plot
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    k = 40
    lr = 0.5
    for lambd in [1]:
        lambda_list = [lambd] * 4
        ## Plot coupled mf with no subject metadata
        reconst_matrix, val_accs = matrix_factorization(train_data=train_data,
                                    train_matrix=train_matrix, k=k, lr=lr,
                                    num_iteration=num_iteration, lambda_list=lambda_list,
                                    val_matrix=val_matrix)

        ## Plot validation accuracy
        plt.plot(range(num_iteration), val_accs, label="lambda = {}".format(lambd))

        val_acc = evaluate_matrix(data_matrix=val_matrix,
                                  reconst_matrix=reconst_matrix)

        print("***************************************")
        print("lambda: {} \t Val Accuracy: {}".format(lambd, val_acc))

    print("***************************************")
    print("lambda: {} \t Test Accuracy: {}".format(lambd, evaluate_matrix(data_matrix=test_matrix,
                                  reconst_matrix=reconst_matrix)))

    plt.legend()
    plt.savefig('graphs/bmf_with_reg_2')


if __name__ == '__main__':
    main()
