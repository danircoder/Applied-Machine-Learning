# coding=utf-8
"""
Part 2. Data structures for boosted trees
-----------------------------------------
In this part we will implement the data-structures we need in order to implement the gradient boosted regression trees.
You are required to implement the following data structures:
1. RegressionTreeNode - This represents a single node in a regression tree. It should have the following member variables:
    (a) j - the index of the feature on which we split at this node
    (b) s - the threshold on which we split at this node
    (c) leftDescendent - the RegressionTreeNode which is the immediate left descendent of the current node
    (d) rightDescendent - the RegressionTreeNode which is the immediate right descendent of the current node
    (e) const - the constant scalar associated with this node
   Note that some fields may take on null values (e.g. const in non-leaf nodes)
   The class should also implement several member methods:
    (a) MakeTerminal( c ) - make the node into a terminal (leaf node) and set its constant value to c
    (b) Split(j,s) - make the node into an internal node which splits on feature index j at threshold s and instantiate its left and right descendents.
    (c) printSubTree() - print the sub- regression tree rooted at the current node in a readable ”if-then” format, for example:
                         if x[’LotArea’]<=7.000000 then:
                             return -23307.173126
                         if x[’LotArea’]>7.000000 then:
                             return 122362.658913
2. RegressionTree - this represents an entire regression tree It should have the following member variables:
    (a) root - this is RegressionTreeNode that represents the root of the tree
   The class should also implement several member methods:
    (a) GetRoot() - Get the root of the tree
    (b) Evaluate(x) - for a vector valued x compute the value of the function represented by the tree
3. RegressionTreeEnsemble - this represents an entire regression tree It should have the following member variables:
    (a) trees - An ordered collection of type RegressionTree
    (b) weights - the weight associated with each regression tree
    (c) M - the number of regression trees
    (d) c - the initial constant value returned before any trees are added
   The class should also implement several member methods:
    (a) Addtree(tree, weight) - add a RegressionTree to the ensemble with corresponding weight
    (b) SetInitialConstant(c) - Set the intital constant c
    (c) Evaluate(x,m)= for a vector valued x compute the value of the function represented by the ensemble of tree, if we consider only the first m trees
   This class encapsulates the model:
    f(x) = sigma_m=1^M (c_m * I[x ∈ Rm])
"""


class RegressionTreeNode:
    """
    Represent a regression tree node (left, right)
    """
    def __init__(self, j=None, s=None, left=None, right=None, scalar=None):
        """
        Initialize a regression tree node with it's variables
        :param j: The index of the feature on which we split at this node
        :param s: The threshold on which we split at this node
        :param left: The RegressionTreeNode which is the immediate left child of the current node
        :param right:  the RegressionTreeNode which is the immediate right child of the current node
        :param scalar: The constant scalar associated with this node
        """
        self.j = j
        self.s = s
        self.left = left
        self.right = right
        self.const = scalar

    def MakeTerminal(self, c):
        """
        Make the node into a terminal (leaf node) and set its constant value to c
        :param c: The constant scalar associated with this node
        """
        self.left = None
        self.right = None
        self.const = c

    def Split(self, j, s):
        """
        Make the node into an internal node which splits on feature index j at threshold s and
        instantiate its left and right children
        :param j: The index of the feature on which we split at this node
        :param s: The threshold on which we split at this node
        """
        self.j = j
        self.s = s
        self.left = RegressionTreeNode()
        self.right = RegressionTreeNode()

    def printSubTree(self):
        """
        Print the sub-regression tree rooted at the current node in a readable ”if-then” format
        """
        print "Shit !"


class RegressionTree:
    """
    Represent a regression tree
    """
    def __init__(self, root):
        """
        Initialize the tree with the root node
        :param root: The tree root node
        """
        self.root = root

    def GetRoot(self):
        """
        Return the tree root node
        :return: Tree root node
        """
        return self.root

    def Evaluate(self, x):
        """
        For a vector valued x compute the value of the function represented by the tree
        :param x: Vector value from the dataset
        :return: The leaf node constant value
        """
        current = self.root
        while not current.const:
            if x[current.j] >= current.s:
                current = current.left
            elif x[current.j] < current.s:
                current = current.right
        return current.const


class RegressionTreeEnsemble:
    """
    Represent a regression tree ensemble (forest)
    """
    def __init__(self, trees, weights, M, c):
        """
        Initialize the regression tree ensemble
        :param trees: An ordered collection of type RegressionTree
        :param weights: The weight associated with each regression tree
        :param M: The number of regression trees
        :param c: The initial constant value returned before any trees are added
        """
        self.trees = trees
        self.weights = weights
        self.M = M
        self.c = c

    def Addtree(self, tree, weight):
        """
        Add a RegressionTree to the ensemble with corresponding weight
        :param tree: A RegressionTree object
        :param weight: The weight associated with the regression tree
        """
        self.trees.append(tree)
        self.weights.append(weight)

    def SetInitialConstant(self, c):
        """
        Set the intital constant c
        :param c: The initial constant value
        """
        self.c = c

    def Evaluate(self, x, m):
        """
        For a vector valued x compute the value of the function represented by the ensemble of tree, if we consider only the first m trees
        :param x: Vector value from the dataset
        :param m: The number of trees to consider (the first)
        :return: The ensemble tree represented function value
        """
        const_sum = 0
        for index in xrange(m):
            tree_evaluation = self.trees[index].Evaluate(x) * self.weights[index]
            const_sum += tree_evaluation
