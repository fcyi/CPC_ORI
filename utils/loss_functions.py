"""
Library of loss functions.
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
"""

import numpy as np
import torch as th


def cpc_loss(encoder_samples, predictions, options):
    """
    :param encoder_samples: Output of CNN-based encoder
    :param predictions: Output generated from context, ct i.w. Wk*ct
    :param options: Dictionary that holds parameters
    :return:
    """
    # Number of time steps predicted
    time_steps = options["time_steps"]
    # Batch size
    bs = options["batch_size"]
    # Initialize softmax to compute accuracy
    softmax = th.nn.Softmax(dim=1)  # 默认沿着行方向，但是最好指定维度
    # Initialize log-Softmax to compute loss
    log_softmax = th.nn.LogSoftmax(dim=1)  # 默认沿着行方向，但是最好指定维度
    # Initialize loss
    InfoNCE = 0
    # Go through each time step, for which we made a prediction and accumulate loss and accuracy.
    for i in np.arange(0, time_steps):
        # Compute attention between encoder samples and predictions
        # torch.mm 是 PyTorch 中专门用于二维张量（矩阵）之间进行矩阵乘法的函数。
        # 与 torch.matmul 不同，torch.mm 仅适用于2D张量，并且不支持高维张量或广播操作。
        # torch.mm 进行标准的矩阵乘法操作，适用于两个2D张量（矩阵）之间的乘法。
        # 对于形状为 (m, n) 的张量 A 和形状为 (n, p) 的张量 B，torch.mm(A, B) 的结果是一个形状为 (m, p) 的张量。
        # 每一个预测编码和真实编码都是一个(B, F)的矩阵，B为一批数据的数目，F为编码特征的维数
        # 此处通过矩阵乘法直接计算，预测编码不同时刻信息与真实编码不同时刻信息的内积所组成的协方差矩阵attention
        # 即attention(i, j)表示第i个真实编码与第j个预测编码的信息相关性，该数值越大越好
        # 根据CPC的最大互信息以及infoNCE的最小化，可知预测信息和真实信息在相同时刻下的相关性应该尽量大，不同时刻下的相关性应该尽量小
        attention = th.mm(encoder_samples[i], th.transpose(predictions[i], 0, 1))
        # Correct classifications are those diagonal elements which has the highest attention in the column they are in.
        # 根据attention中每一列元素的最大值是否处于对角线上来计算分类（也就是CPC所提及的特征的线性可分，
        # 这也是为何CPC将不同的线性层作用于上下文表征以产生若干个未来时刻的编码的原因，线性可分性越强表示
        # 不同时刻上的特征表示解耦越强）的准确性
        accuracy = th.sum(th.eq(th.argmax(softmax(attention), dim=0), th.arange(0, bs)))
        # InfoNCE is computed using log_softmax, and summing diagonal elements.
        # nce is a tensor. torch.diag()表示取出attention在经过log_softmax后的对角线元素
        # 由于是服从每一行数据都服从概率分布，因此对角线上元素最大化，那么对应行上其余元素就会被最小化
        InfoNCE += th.sum(th.diag(log_softmax(attention)))
    # Negate and take average of InfoNCE over batch and time steps.
    # Minimizing -InfoNCE is equivalent to max attention along diagonal elements
    InfoNCE /= -1. * bs * time_steps  # 返回平均的infoNCE，并且为了梯度下降，取了个负数
    # Compute the mean accuracy
    accuracy = 1. * accuracy.item() / bs  # 返回平均准确率
    # Return values
    return InfoNCE, accuracy