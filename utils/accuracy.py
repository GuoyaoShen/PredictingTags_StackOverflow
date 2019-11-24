'''
This script define functions for scoring predicted tags from network
(c) Guoyao Shen
'''
import numpy as np
import torch

def accuracy(tags_predict, tags_true, prob_threshold):
    '''
    tag_dim indicate the num of tags
    :param tags_predict: shape [batch_size, tag_dim]
    :param tags_true: shape [batch_size, tag_dim]
    :param prob_threshold: float from 0 to 1
    :return: acc: the average accuracy of predictions for batch size examples
    '''
    # make sure it's numpy
    tags_predict = np.array(tags_predict).astype(float)
    tags_true = np.array(tags_true).astype(float)
    if tags_predict.ndim>1:
        N = tags_predict.shape[0]
    else:
        N=1
    # print('N:', N)

    tags_predict_mix = np.where(tags_predict >= prob_threshold, 1, tags_predict)  # mix of prob and 1
    tags_predict_binary = np.where(tags_predict >= prob_threshold, 1, 0) # 0 or 1

    # right = np.sum(tags_predict_binary*tags_true)
    # wrong = np.sum(tags_predict_mix)-right

    right = np.sum(tags_predict_binary * tags_true)
    wrong = np.sum(tags_predict_binary) - right
    # print('right', right)
    # print('wrong', wrong)

    acc = (right - wrong)/N
    if acc < 0:
        acc = 0
    return acc


def evaluation(tags_predict, tags_true, prob_threshold, name_type):
    '''
    Give evaluation of the prediction for true tags
    :param tags_predict: shape [batch_size, tag_dim]
    :param tags_true: shape [batch_size, tag_dim]
    :param prob_threshold: float from 0 to 1
    :param name_type: str, evaluation type: 'precision', 'recall'('sensitivity')
    :return: eval: the average evaluation of predictions for batch size examples
    '''
    # make sure it's numpy
    tags_predict = np.array(tags_predict).astype(float)
    tags_true = np.array(tags_true).astype(float)
    if tags_predict.ndim > 1:
        N = tags_predict.shape[0]
    else:
        N = 1

    tags_predict_mix = np.where(tags_predict >= prob_threshold, 1, tags_predict)  # mix of prob and 1
    tags_predict_binary = np.where(tags_predict >= prob_threshold, 1, 0)  # 0 or 1

    if name_type == 'precision':
        TP = np.sum(tags_predict_binary * tags_true)
        TPFP = np.sum(tags_predict_binary)
        eval = TP / (TPFP*1.0)
    elif name_type == 'recall':
        TP = np.sum(tags_predict_binary * tags_true)
        TPFN = np.sum(tags_true)
        eval = TP / (TPFN*1.0)
    else:
        print('NOT supported, please set type(arg3) as: precision/ recall')
        eval = 0

    return eval


if __name__ == '__main__':
    # test accuracy_Lnorm
    # a = torch.tensor([[0.2, 0.4, 0.3], [0.3, 0.8, 0.1]])
    a = torch.tensor([[0.0, 0.1, 0.7], [0.0, 0.8, 0.0]])
    b = torch.tensor([[0, 0, 1], [0, 1, 0]])
    acc = accuracy(a, b, 0.5)
    print('ACC', acc)
