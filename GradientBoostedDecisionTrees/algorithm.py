# coding=utf-8
"""
Part 3. The Algorithm
-----------------------------------------
In this section we will implement the gradient boosted regression tree algorithm. In this part
you are required to implement the following:
1. A function named CART that takes the hyper parameters MaxDepth and MinNodeSize
as arguments as well as a dataset object. The function should implement the CART
algorithm that we saw in class for building regression trees. It should return an object of
type RegressionTree
2. A function named GBRT which takes as input the hyperparameters NumTrees,MaxDepth
and MinNodeSize. The function should implement the Gradient Boosted Regression Tree
Algorithm we saw in class and return an object of type RegressionTreeEnsemble
"""

import data_input
import data_structure
import numpy as np


def GetOptimalPartition(dataset):
    # j, s
    return

def CART_helper(node, MaxDepth, MinNodeSize, depth):
    """
    A recursive func to split the tree
    :param MaxDepth:
    :param MinNodeSize:
    :param depth:
    :return: None
    """
    # check for a no split
    if not node.left or not node.right:
        node.MakeTerminal(c?)
        return

    # check for max depth
    if depth >= MaxDepth:
        node.left.MakeTerminal(c?)
        node.right.MakeTerminal(c?)
        return

    # process left child
    # TODO: we need to decide how much samples are inside the child in order to check MinNodeSize !!

    # process right child
    # TODO: we need to decide how much samples are inside the child in order to check MinNodeSize !!

def CART(MaxDepth, MinNodeSize, dataset):
    """
    1. A function named CART that takes the hyper parameters MaxDepth and MinNodeSize
    as arguments as well as a dataset object. The function should implement the CART
    algorithm that we saw in class for building regression trees. It should return an object of
    type RegressionTree
    :param MaxDepth:
    :param MinNodeSize:
    :param dataset:
    :return: RegressionTree
    """
    j, s = GetOptimalPartition(dataset)
    root = data_structure.RegressionTreeNode(j=None, s=None, left=None, right=None, Scalar=None)
    root.Split(j, s)
    CART_helper(root, MaxDepth, MinNodeSize, depth=1)
    tree = data_structure.RegressionTree(root)
    return tree
'''
# TODO: ITERATIVE?
def CART(MaxDepth, MinNodeSize, dataset):
    """
    1. A function named CART that takes the hyper parameters MaxDepth and MinNodeSize
    as arguments as well as a dataset object. The function should implement the CART
    algorithm that we saw in class for building regression trees. It should return an object of
    type RegressionTree
    :param MaxDepth:
    :param MinNodeSize:
    :param dataset:
    :return: RegressionTree
    """
    root = None
    for k in xrange(MaxDepth):
        j, s = GetOptimalPartition(dataset)
        node = data_structure.RegressionTreeNode(j=None, s=None, left=None, right=None, Scalar=None)
        if check_MinNodeSize(): # TODO: complete check_MinNodeSize
            node.Split(j, s)
        else:
            c = get_const() # TODO: complete get_const
            node.MakeTerminal(c)
        # save a pointer to the root, in order to initiate the tree structure later
        if k == 0:
            root = node

    tree = data_structure.RegressionTree(root)
    return tree
'''


def GBRT(NumTrees, MaxDepth, MinNodeSize, dataset):
    """
    2. A function named GBRT which takes as input the hyperparameters NumTrees,MaxDepth
    and MinNodeSize. The function should implement the Gradient Boosted Regression Tree
    Algorithm we saw in class and return an object of type RegressionTreeEnsemble
    :param NumTrees:
    :param MaxDepth:
    :param MinNodeSize:
    :return: RegressionTreeEnsemble
    """
    # initiate the tree_ensemble (called fm(X) in the formula)
    tree_ensemble = data_structure.RegressionTreeEnsemble(trees=trees, weights=weights, M=NumTrees, c=0)

    # g is a matrix of M (NumTrees) rows and each row hold N (number of samples in the dataset) columns
    # so in each iteration we will update the entry at g[m][i]
    g = np.zeros((NumTrees, dataset.shape[0]))

    # for sq. loss we proved in class that the minimum is achieved at:
    # f0(x) <- 1/N Sig[yi]
    f0 = dataset['SalePrice'].mean()
    tree_ensemble.SetInitialConstant(f0)

    # TODO: decide if (1, NumTrees+1). because the first will be g[m][i] -= tree_ensemble.Evaluate(x=dataset[i], m=-1)
    for m in xrange(NumTrees):
        # for i = 1, ..., N do
        for i in xrange(dataset.shape[0]): # df.shape[0] gives number of row count
            # the gradient for sq. loss is -(yi − fm−1 (xi))
            # gim <− -(yi − f_m−1 (xi))
            xi = dataset.iloc[i, list(range(dataset.shape[1]-1))] # extract a row excluding the last column which is the label
            yi = dataset.iloc[i, -1] # extract the cell of the i'th row at the last column which is the label
            g[m][i] -= yi - tree_ensemble.Evaluate(x=xi, m=m-1)
        tree_m = CART(MaxDepth, MinNodeSize, dataset)

        #calc the weight beta_m
        beta_m = 0# -g[m].sum()
        for i in xrange(dataset.shape[0]):  # df.shape[0] gives number of row count
            xi = dataset.iloc[i, list(range(dataset.shape[1] - 1))]  # extract a row excluding the last column which is the label
            tree_m_for_xi = tree_m.Evaluate(x=xi)
            # TODO: consider the minux g; i think beta_m should be positive at the end, so maybe need to delete the minus..
            beta_m += (-g[m][i]*tree_m_for_xi) / tree_m_for_xi**2

        # update fm(X) by adding the tree_m
        # TODO: consider the minux g; i think beta_m should be positive at the end, so maybe need to delete the minus..
        tree_ensemble.Addtree(tree_m, beta_m)
    return tree_ensemble

