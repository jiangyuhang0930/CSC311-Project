'''An offshoot of cmf3
Features:
- Actual stochastic gradient descent
- Varying learning rate for the matrices
- Biases for each matrix
- Combining student metadata into single matrix rather than separate

Note: this more closely resembles the algorithm from the paper
Note: because of matrix multiplication in bias updates, this takes a long time to run
'''

## TODO: HYPERPARAMETER TUNING!!!
## - k value
## - metadata to include (ie, the metadata_list)
## - number of iterations
## - learning rate(s)
## Plan: tune the lr and num_it before tuning k_val and metadata_list

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.linalg import sqrtm

def impute_matrix_values(matrix):
    """Return a new matrix with filled values being the means
    """
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    return masked_matrix.filled(item_means)

def svd(matrix, k):
    """ Given the matrix, perform singular value decomposition to get latent factors

    :param matrix: 2D sparse matrix
    :param k: int
    :return: u, z
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # # Next, compute the average and subtract it.
    # item_means = np.mean(new_matrix, axis=0)
    # mu = np.tile(item_means, (new_matrix.shape[0], 1))
    # new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    u = np.dot(Q, s_root)
    z = np.dot(s_root, Ut).T
    # reconst_matrix = reconst_matrix + mu
    return u, z


def create_sparse_matrix(data, shape, col_label=None, student=0):
    """ Create a sparse matrix representation of data with specified shape
    and axes. If col_label is None and student=1, create the correctness
    matrix.

    :param data: Dictionary of data
    :param shape: Tuple of matrix shape
    :param col_label: The column label specifying metadata matrix
    :param student: Whether or not this is a student metadata matrix
    """
    matrix = np.ones(shape) * np.nan

    if col_label is None and student: ## Correctness matrix
        for i, u in enumerate(data["user_id"]):
            q = data["question_id"][i]
            c = data["is_correct"][i]
            matrix[u][q] = c
    elif student: ## Student metadata matrices
        for i, x in enumerate(data[col_label]):
            u = data["user_id"][i]
            ## Create one-hot vector
            row = np.zeros(shape[1])
            row[x] = 1
            matrix[u] = row
    else: ## Question metadata matrix
        for i, s_lst in enumerate(data["subject_id"]):
            q = data["question_id"][i]
            ## Create vector
            col = np.zeros(shape[0])
            col[s_lst] = 1
            matrix.T[q] = col

    return matrix


def partial_update(matrix, u, z, lr, student=0, student_meta=0, deriv_u=0, data=None):
    """ Return the updated u and z from stochastic gradient descent
    for matrix completion (for a single sum).

    :param data: A dictionary
    :param matrix: 2D matrix of data
    :param u: 2D matrix
    :param z: 2D matrix
    :param: lr: float
    :param student: Whether the row label is student
    :param label: string (specifies the matrix column label)
    :param deriv_u: Whether to return derivative wrt u or z
    :return: the updated u and z
    """
    ## Randomly select pair and find corresponding indices
    if student:
        # (user_id, label).
        i = np.random.choice(len(data["user_id"]), 1)[0]
        n = data["user_id"][i]
        m = data["question_id"][i]
    elif student_meta:
        n = np.random.choice(u.shape[0], 1)[0]
        m = np.random.choice(z.shape[0], 1)[0]
    else:
        # (subject_id, question_id).
        i = np.random.choice(len(data["question_id"]), 1)[0]
        n = np.random.choice(data["subject_id"][i], 1)[0]
        m = data["question_id"][i]

    ## Get correct value
    x = matrix[n][m]

    ## Stochastic gradient descent
    if deriv_u:
        du = -1 * (x - np.dot(u[n], z[m])) * z[m]
        u[n] -= lr * du
    else:
        dz = -1 * (x - np.dot(u[n], z[m])) * u[n]
        z[m] -= lr * dz

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def update_u_z(all_data, all_matrices_old, all_matrices_new, learning_rates, u,
               z, s, d, B):
    """ Return the updated u and b_i's after applying stochastic gradient descent
        for matrix completion.

        :param learning_rates: list of float [correctness, age, gender, premium, subject]
        :param u: 2D matrix (latent student matrix)
        :param z: 2D matrix (latent question matrix)
        :param s: latent 2D matrix (corresponding to student metadata)
        :param d: 2D matrix (latent subject matrix)
        :param B: nested list of bias vectors
        :return: (u, z, B, d)
    """

    ## Put all these in lists to make for loops simpler
    u_list = [u, u, d]
    z_list = [z, s, z]

    ## Update u (user)
    u = partial_update(data=all_data[0], matrix=all_matrices_old[0],
                        u=u, z=z, student=1, lr=learning_rates[0], deriv_u=1)[0]
    u = partial_update(matrix=all_matrices_new[1],
                       u=u, z=s, student_meta=1, lr=learning_rates[1], deriv_u=1)[0]

    ## Update z (question)
    z = partial_update(data=all_data[0], matrix=all_matrices_old[0],
                               u=u, z=z, student=1, lr=learning_rates[0])[1]
    z = partial_update(data=all_data[1], matrix=all_matrices_old[2],
                       u=d, z=z, lr=learning_rates[2])[1]

    ## Update s
    s = partial_update(matrix=all_matrices_new[1],
                        u=u, z=s, student_meta=1, lr=learning_rates[1])[1]
    ## Update d
    d = partial_update(data=all_data[1], matrix=all_matrices_old[2],
                               u=d, z=z, lr=learning_rates[2], deriv_u=1)[0]

    ## Update biases and filled matrices
    for i in range(3):
        not_nan = 1 - np.isnan(all_matrices_old[i])
        full_mat = np.where(not_nan == 1, all_matrices_old[i], all_matrices_new[i])
        n = u_list[i].shape[0]
        m = z_list[i].shape[0]
        pred_mat = u_list[i].dot(z_list[i].T)

        ## Update biases
        B[i][0] = full_mat - pred_mat - np.dot(np.ones((n, 1)), B[i][1].T)
        B[i][0] = B[i][0].dot(np.ones((m, 1))) / m

        B[i][1] = full_mat - pred_mat - np.dot(B[i][0], np.ones((1, m)))
        B[i][1] = B[i][1].T.dot(np.ones((n, 1))) / n

        ## Update filled matrices
        all_matrices_new[i] = pred_mat + np.dot(B[i][0], np.ones((1, m))) \
                              + np.dot(np.ones((n, 1)), B[i][1].T)

    return u, z, s, d, B


def als(train_data, train_matrix, student_matrix,
        question_data, subject_matrix, k, learning_rates, num_iteration, nums,
        val_matrix=None, pretrained_matrix=None):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param train_matrix: Sparse matrix representation of training data
    :param student_data: A list of dictionaries of student metadata (age, gender, premium)
    :param student_matrix: Sparse matrix representation of student metadata
    :param question_data: Dictionary of question metadata
    :param subject_matrix: Matrix of question subjects
    :param k: int
    :param learning_rates: list of float [correctness, age, gender, premium, subject]
    :param num_iteration: int
    :param nums: [num_students, num_ages, num_genders, num_premium, num_questions, num_subjects]
    :param metadata_list: List of which "metadata" stuff to include
                0 = correctness, 1 = age, 2 = gender, 3 = premium_pupil, 4 = subject
                Note that the first element must be 0 if not None
    :param val_matrix: validation matrix
    :return: 2D reconstructed Matrix.
    """

    # Impute missing values of correctness, age, gender, premium, subject matrices
    all_matrices_old = [train_matrix, student_matrix, subject_matrix]
    all_matrices_new = [impute_matrix_values(m) for m in all_matrices_old]

    if pretrained_matrix is not None:
        all_matrices_new[0] = pretrained_matrix
        u, z = svd(matrix=pretrained_matrix, k=k)
    else:
        ## Student
        u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                              size=(nums[0], k))
        ## Question
        z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                              size=(nums[2], k))

    # Initialize u, z, S, d, B
    B = [] ## List of tuples (row bias, col bias)

    ## Correctness bias
    B.append([np.zeros((nums[0], 1)), np.zeros((nums[2], 1))])

    ## Student metadata
    s = np.random.uniform(low=0, high=1 / np.sqrt(k), size=(nums[1], k))
    ## Student metadata biases
    B.append([np.zeros((nums[0], 1)), np.zeros((nums[1], 1))])

    ## Question metadata (ie, subject)
    d = np.random.uniform(low=0, high=1 / np.sqrt(k),
                            size=(nums[3], k))
    B.append([np.zeros((nums[3], 1)), np.zeros((nums[2], 1))])

    all_data = [train_data, question_data]

    #####################################################################
    val_accs = []
    for i in range(num_iteration):
        u, z, s, d, B = update_u_z(all_data=all_data, all_matrices_old=all_matrices_old,
                                   all_matrices_new=all_matrices_new,
                                learning_rates=learning_rates, u=u, z=z, s=s,
                                d=d, B=B)

        if val_matrix is not None:
            not_nan = 1 - np.isnan(all_matrices_old[0])
            pred_mat = np.where(not_nan == 1, all_matrices_old[0],
                                all_matrices_new[0])
            val_acc = evaluate_matrix(data_matrix=val_matrix, reconst_matrix=pred_mat)
            val_accs.append(val_acc)
            print("Iteration: {} \t Val Acc: {}".format(i, val_acc))

    ## Final prediction
    not_nan = 1 - np.isnan(all_matrices_old[0])
    mat = np.where(not_nan == 1, all_matrices_old[0], all_matrices_new[0])
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
    subject_data = load_subject_meta("../data")

    num_students = train_matrix.shape[0]
    num_questions = train_matrix.shape[1]
    num_ages = max(student_data[0]["age"]) + 1
    num_genders = 3
    num_premium = 2
    num_subjects = len(subject_data["subject_id"])
    nums = [num_students, num_ages + num_genders + num_premium, num_questions, num_subjects]

    ## Create sparse matrix representations of the data
    val_matrix = create_sparse_matrix(data=val_data,
                                      shape=(nums[0], nums[2]), student=1)
    test_matrix = create_sparse_matrix(data=test_data,
                                      shape=(nums[0], nums[2]), student=1)
    subject_matrix = create_sparse_matrix(data=question_data,
                                           shape=(nums[3], nums[2]), col_label="question_id")
    age_matrix = create_sparse_matrix(data=student_data[0],
                                      shape=(nums[0], num_ages), col_label="age",
                                      student=1)
    gender_matrix = create_sparse_matrix(data=student_data[1],
                                         shape=(nums[0], num_genders), col_label="gender",
                                         student=1)
    premium_matrix = create_sparse_matrix(data=student_data[2],
                                            shape=(nums[0], num_premium), col_label="premium_pupil",
                                          student=1)

    student_meta_matrix = np.concatenate((age_matrix, gender_matrix, premium_matrix), axis=1)

    num_iteration = 500

    ## TODO: Grid search to tune hyperparameters
    ## TODO: Plots of loss and accuracy vs. iteration for training and validation
    ## TODO: Print final validation and test accuracies

    ## Can change this to experiment with models using different metadata
    ## Refer to als docstring for more details
    # metadata_list = None

    ## Plot with subject metadata
    # plt.figure()
    # plt.title("Validation Accuracy k = {} lr = {}".format(k, learning_rates))
    # plt.xlabel("Iteration")
    # plt.ylabel("Accuracy")
    best_acc = 0
    best_lr = 0 ## 0.1
    k = 3
    learning_rates = [0.1, 0.1, 0.1]
    ## Plot coupled mf with no subject metadata
    for lr in [0.1]:
        learning_rates[0] = lr
        # for size in range(4):
        #     for combo in itertools.combinations(range(1, 4), size):
        reconst_matrix, val_accs = als(train_data=train_data,
                                       train_matrix=train_matrix,
                                       student_data=student_data,
                                       student_matrix=student_meta_matrix,
                                       question_data=question_data,
                                       subject_matrix=subject_matrix,
                                       k=k,
                                       learning_rates=learning_rates,
                                       num_iteration=num_iteration,
                                       nums=nums, val_matrix=val_matrix)

        ## Plot validation accuracy
        plt.plot(range(num_iteration), val_accs)

        val_acc = evaluate_matrix(data_matrix=val_matrix,
                                  reconst_matrix=reconst_matrix)
        if val_acc > best_acc:
            best_lr = lr
            best_acc = val_acc

        print("***************************************")
        print("k = {} lr = {}".format(k, learning_rates))
        print("Final Validation Accuracy: ", val_acc)

    print("************************************")
    print("Best lr: ", best_lr)

    # plt.legend()
    plt.savefig('graphs/cmf5')

if __name__ == '__main__':
    main()
