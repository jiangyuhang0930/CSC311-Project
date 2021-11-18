from utils import *

import numpy as np
from matplotlib import pyplot as plt
from part_b.irt_sigmoid import irt
from part_b.biased_mf_with_l2_reg import matrix_factorization, \
    create_sparse_matrix, concatenate_meta_to_correctness
from part_b.cmf7 import als

def split_training_set(train_data, shape):
    """Split the training set so as to train on different models
    """
    len_data = len(train_data["user_id"])
    train_indices_1 = np.random.choice(range(len_data), len_data // 2, replace=False)
    train_indices_2 = np.array(list(set(range(len_data)) - set(train_indices_1)))

    train_dict_1 = {"user_id": [], "question_id": [], "is_correct": []}
    train_dict_2 = {"user_id": [], "question_id": [], "is_correct": []}

    ## Split dictionary
    for label in ["user_id", "question_id", "is_correct"]:
        full_list = np.array(train_data[label])
        train_dict_1[label] = list(np.take(full_list, train_indices_1))
        train_dict_2[label] = list(np.take(full_list, train_indices_2))

    train_matrix_1 = create_sparse_matrix(train_dict_1, shape)
    train_matrix_2 = create_sparse_matrix(train_dict_2, shape)

    return train_matrix_1, train_matrix_2, train_dict_1, train_dict_2


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    val_matrix = create_sparse_matrix(data=val_data,
                                      shape=train_matrix.shape)

    ## Split data
    train_matrix_1, train_matrix_2, train_data_1, train_data_2 = \
        split_training_set(train_data=train_data, shape=train_matrix.shape)

    ## Train using IRT for first half of data
    lr = 0.05
    num_it = 50
    print("Pretraining with IRT:")
    pretrained_matrix = irt(lr=lr, iterations=num_it, data_matrix=train_matrix_1, val_matrix=val_matrix,
                            eval_acc=1)[0]
    val_acc = evaluate_matrix(data_matrix=val_matrix, reconst_matrix=pretrained_matrix)
    print("Val Acc After Pretraining: ", val_acc)

    ## Train using MF for second half of data
    student_data = load_student_meta(
        "../data")  ## student_data is a tuple of metadata dictionaries
    question_data = load_question_meta("../data")
    subject_type_data = load_subject_meta("../data")

    num_students = train_matrix.shape[0]
    num_questions = train_matrix.shape[1]
    num_subjects = len(subject_type_data["subject_id"])

    subject_matrix = create_sparse_matrix(data=question_data,
                                          shape=(num_subjects, num_questions),
                                          create_subject=1)

    train_matrix = concatenate_meta_to_correctness(data_matrix=train_matrix,
                                                   student_data=student_data,
                                                   subject_matrix=subject_matrix)

    pretrained_matrix = np.concatenate((pretrained_matrix, np.zeros((num_students, 3))), axis=1)

    k = 20
    num_it = 50
    lr = 0.05
    lambda_list = [1, 1]
    print("***********************")
    print("Biased Matrix Factorization:") ## This one overfits
    pred_matrix_2 = matrix_factorization(train_data=train_data_2, train_matrix=train_matrix,
                                         k=k, lr=lr, num_iteration=num_it,
                                         lambda_list=lambda_list, pretrained_matrix=pretrained_matrix,
                                         val_matrix=val_matrix)[0]
    pred_matrix_2 = pred_matrix_2[:num_students].T[:num_questions].T
    ## Validation accuracy
    val_acc = evaluate_matrix(data_matrix=val_matrix, reconst_matrix=pred_matrix_2)

    print("Final Validation Accuracy: ", val_acc)
    return

    student_data = load_student_meta(
        "../data")  ## student_data is a tuple of metadata dictionaries
    question_data = load_question_meta("../data")
    subject_type_data = load_subject_meta("../data")

    num_students = train_matrix.shape[0]
    num_questions = train_matrix.shape[1]
    num_subjects = len(subject_type_data["subject_id"])

    subject_matrix = create_sparse_matrix(data=question_data,
                                          shape=(num_subjects, num_questions),
                                          create_subject=1)

    train_matrix = concatenate_meta_to_correctness(data_matrix=train_matrix,
                                                   student_data=student_data,
                                                   subject_matrix=subject_matrix)

    ## Create sparse matrix representations of the data
    val_matrix = create_sparse_matrix(data=val_data,
                                      shape=(num_students, num_questions))
    test_matrix = create_sparse_matrix(data=test_data,
                                       shape=(num_students, num_questions))

    num_iteration = 1000

if __name__ == '__main__':
    main()
