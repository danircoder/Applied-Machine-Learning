# coding=utf-8
"""
Part 3. The Algorithm
---------------------
In class we discussed two possible approaches to minimizing the cost function in Equation 1: Alternating Least
Squares and Stochastic Gradient Descent.

Part 4. Machine Learning Code Diagnostics
-----------------------------------------
According to the theory our optimization procedure should be decreasing our the value of our error function with
each iteration of our algorithm. One faithful way to debug our code is to make sure this is indeed the case.
One way to determine if we are overfitting the noise in our training data is to measure the error on a validation
set as well.
"""

import time
import numpy as np
from scipy.linalg import solve_triangular
from model import HyperParameters


"""
Section 1.
----------
Derive the gradient of Equation 1 with respect to each of the variables u_m, v_n,b_m, and b_n.
In notebook:

Computing the gradient with respect to each of the parameters:
    ∇u_mE(Θ) =− ∑n ∈ I_m*(r_m,n − (u_mT*v_n + b_n + b_m))*v_n + λu_m 
    ∇v_nE(Θ) =− ∑m ∈ I_n*(r_m,n − (u_mT*v_n + b_n + b_m))*u_m + λv_n 
    ∇b_mE(Θ) =− ∑n ∈ I_m*(r_m,n − (u_mT*v_n + b_n + b_m)) + λb_m 
    ∇b_nE(Θ) =− ∑m ∈ I_n*(r_m,n − (u_mT*v_n + b_n + b_m)) + λb_n


now set the partial gradient to zero and extact required variable:
    ∇u_mE(Θ) = 0 ⟹ u_m = (∑n∈I_m v_n*v_nT + λ_u_m * I)^−1 * ∑n∈I_m (r_m,n * v_n − b_n*v_n − b_m*v_n) 
    ∇v_nE(Θ) = 0 ⟹ v_n = (∑m∈I_n v_n*v_nT + λ_v_n * I)^−1 * ∑m∈I_n (r_m,n * u_m − b_n*u_m − b_m*u_m) 
    ∇b_mE(Θ) = 0 ⟹ b_m = (λ_b_m)^−1 * ∑n∈I_m (r_m,n − u_m*v_n − b_n)
    ∇b_nE(Θ) = 0 ⟹ b_n = (λ_b_n)^−1 * ∑m∈I_n (r_m,n − u_m*v_n − b_m)
"""

"""
Section 2(a).
-------------
Use the derivation in step 1 to derive the update equations for each of the variables u_m, v_n,b_m, and b_n 
needed for Stochastic Gradient Descent.

(the derivation from step 1):
Computing the gradient with respect to each of the parameters:
    ∇u_mE(Θ) =− ∑n ∈ I_m*(r_m,n − (u_mT*v_n + b_n + b_m))*v_n + λu_m 
    ∇v_nE(Θ) =− ∑m ∈ I_n*(r_m,n − (u_mT*v_n + b_n + b_m))*u_m + λv_n 
    ∇b_mE(Θ) =− ∑n ∈ I_m*(r_m,n − (u_mT*v_n + b_n + b_m)) + λb_m 
    ∇b_nE(Θ) =− ∑m ∈ I_n*(r_m,n − (u_mT*v_n + b_n + b_m)) + λb_n

(update equations needed for Stochastic Gradient Descent):
Randomly initialize Θ
for η epochs do:
    foreach (m,n)∈I do:
        um  ←   um−α(−(rm,n−(uTmvn+bn+bm))vn+λumum) 
        vn  ←   vn−α(−(rm,n−(uTmvn+bn+bm))um+λvnvn) 
        bm  ←   bm−α(−(rm,n−(uTmvn+bn+bm))+λbmbm) 
        bn  ←   bn−α(−(rm,n−(uTmvn+bn+bm))+λbnbn)
    END
END
"""


__sgd_train_learning_curve__ = r"output\sgd_train_learning_curve.dat"
__sgd_test_learning_curve__  = r"output\sgd_test_learning_curve.dat"
__als_train_learning_curve__ = r"output\als_train_learning_curve.dat"
__als_test_learning_curve__  = r"output\als_test_learning_curve.dat"


class SGDHyperParameters(HyperParameters):
    """
    Section 2(b).
    -------------
    Implement a class encapsulating all the hyperparameters needed for Stochastic Gradient Descent
    """
    def __init__(self,
                 lambda_user,
                 lambda_item,
                 lambda_bias_user,
                 lambda_bias_item,
                 eta,
                 alpha,
                 latent_factor):
        """
        Initialization method for the SGD hyper parameters class
        :param lambda_user: Regularization value for the users
        :param lambda_item: Regularization value for the items
        :param lambda_bias_user: Regularization value for the user bias
        :param lambda_bias_item: Regularization value for the item bias
        :param eta: Epoch count, iteration count
        :param alpha: Learning rate
        :param latent_factor: The number of latent dimensions of the vectors (d)
        """
        super(SGDHyperParameters, self).__init__(lambda_user, lambda_item, lambda_bias_user, lambda_bias_item, latent_factor)
        self.eta = eta
        self.alpha = alpha


class StochasticGradientDescent(object):
    """
    Implementation of SGD algorithm for training a model
    """
    def __init__(self, model):
        """
        Initialization method for the SGD algorithm
        :param model: MFModel initialized object
        """
        self.model = model
        self.ratings_indexes = list(self.model.ratings_indexes)

    def train(self):
        """
        Train the MFModel according to the SGD algorithm
        """
        np.random.shuffle(self.ratings_indexes)
        self.__update_step()

    def __update_step(self):
        """
        Update step for the SGD algorithm which update the model vectors
        """
        for user, item in self.ratings_indexes:
            # Get prediction and calculate error
            prediction = self.model.predict(user, item)
            error = self.model.ratings[user, item] - prediction

            # Update vectors and bias # Nir: i added minus before the "error" - look in the formula above..
            self.model.user_vectors[user, :] -= self.model.hyper.alpha * ((-error * self.model.item_vectors[item, :]) + (self.model.hyper.lambda_user * self.model.user_vectors[user, :]))
            self.model.item_vectors[item, :] -= self.model.hyper.alpha * ((-error * self.model.user_vectors[user, :]) + (self.model.hyper.lambda_item * self.model.item_vectors[item, :]))
            self.model.user_bias[user] -= self.model.hyper.alpha * (-error + (self.model.hyper.lambda_bias_user * self.model.user_bias[user]))
            self.model.item_bias[item] -= self.model.hyper.alpha * (-error + (self.model.hyper.lambda_bias_item * self.model.item_bias[item]))


def LearnModelFromDataUsingSGD(ratings, model, hyperparameters, additional_ratings=None):
    """
    Section 2(c). (Part 3)
    -------------
    Implement a function LearnModelFromDataUsingSGD which takes in a dataset, a model and a set of algorithm
    hyperparameters and learns the model parameters using Stochastic Gradient Descent.

    Section 1. (Part 4)
    ----------
    Modify the functions LearnModelFromDataUsingSGD and LearnModelFromDataUsingALS to output the sum of squared
    errors on the training data before each iteration to a file.

    Section 2. (Part 4)
    ----------
    Modify the two functions above to optionally take a second dataset and if given output the sum of squared errors
    on this dataset to a second file.
    :param ratings: Ratings matrix (train dataset)
    :param model: Model parameters object
    :param hyperparameters: Set of algorithm hyperparameters
    :param additional_ratings: Additional ratings matrix (test dataset)
    :return: Learnt model using SGD
    """
    model.ratings = ratings
    model.hyper = hyperparameters
    sgd = StochasticGradientDescent(model)
    with open(__sgd_train_learning_curve__, "w") as sgd_train, open(__sgd_test_learning_curve__, "w") as sgd_test:
        print "Stochastic Gradient Descent Training:"
        print "-------------------------------------"
        print "λ - User: {0}, λ - Item: {1}, λ - User Bias: {2}, λ - Item Bias: {3}, α: {4}, Epochs: {5}, Dimensions: {6}.".format(model.hyper.lambda_user,
                                                                                                                                   model.hyper.lambda_item,
                                                                                                                                   model.hyper.lambda_bias_user,
                                                                                                                                   model.hyper.lambda_bias_item,
                                                                                                                                   model.hyper.alpha,
                                                                                                                                   model.hyper.eta,
                                                                                                                                   model.hyper.latent_factor)
        iteration = 0
        test_mse = 0
        iteration_len = 12
        training_mse_len = 12
        test_mse_len = 12
        time_interval_len = 15
        print "+{0}+{1}+{2}+{3}+".format("-" * iteration_len, "-" * training_mse_len, "-" * test_mse_len, "-" * time_interval_len)
        print "| {0} | {1} | {2} | {3} |".format("Iteration".ljust(iteration_len - 2),
                                                 "Train MSE".ljust(training_mse_len - 2),
                                                 "Test MSE".ljust(test_mse_len - 2),
                                                 "Time Interval".ljust(time_interval_len - 2))
        print "+{0}+{1}+{2}+{3}+".format("-" * iteration_len, "-" * training_mse_len, "-" * test_mse_len, "-" * time_interval_len)

        timer = time.clock()

        train_mse = model.mean_squared_error(ratings)
        sgd_train.write("0,{0}\n".format(train_mse))

        if additional_ratings and additional_ratings.any():
            test_mse = model.mean_squared_error(additional_ratings)
            sgd_test.write("0,{0}\n".format(test_mse))

        print "| {0} | {1} | {2} | {3} |".format(str(iteration).ljust(iteration_len - 2),
                                                 str("%.6f" % train_mse).ljust(training_mse_len - 2),
                                                 str("%.6f" % test_mse).ljust(test_mse_len - 2),
                                                 str("%.6f" % (time.clock() - timer)).ljust(time_interval_len - 2))
        for iteration in xrange(1, model.hyper.eta):
            timer = time.clock()

            sgd.train()

            train_mse = model.mean_squared_error(ratings)
            sgd_train.write("{0},{1}\n".format(iteration, train_mse))

            if additional_ratings and additional_ratings.any():
                test_mse = model.mean_squared_error(additional_ratings)
                sgd_test.write("{0},{1}\n".format(iteration, test_mse))

            print "| {0} | {1} | {2} | {3} |".format(str(iteration).ljust(iteration_len - 2),
                                                     str("%.6f" % train_mse).ljust(training_mse_len - 2),
                                                     str("%.6f" % test_mse).ljust(test_mse_len - 2),
                                                     str("%.6f" % (time.clock() - timer)).ljust(time_interval_len - 2))
        print "+{0}+{1}+{2}+{3}+".format("-" * iteration_len, "-" * training_mse_len, "-" * test_mse_len, "-" * time_interval_len)
    return model


"""
Section 3(a).
-------------
Use the derivation in step 1 to derive the update equations for each of the variables u_m, v_n,b_m, and b_n 
needed for Alternating Least Squares

(the derivation from step 1):
∇u_mE(Θ) = 0 ⟹ u_m = (∑n∈I_m v_n*v_nT + λ_u_m * I)^−1 * ∑n∈I_m (r_m,n * v_n − b_n*v_n − b_m*v_n) 
∇v_nE(Θ) = 0 ⟹ v_n = (∑m∈I_n v_n*v_nT + λ_v_n * I)^−1 * ∑m∈I_n (r_m,n * u_m − b_n*u_m − b_m*u_m) 
∇b_mE(Θ) = 0 ⟹ b_m = (λ_b_m)^−1 * ∑n∈I_m (r_m,n − u_m*v_n − b_n)
∇b_nE(Θ) = 0 ⟹ b_n = (λ_b_n)^−1 * ∑m∈I_n (r_m,n − u_m*v_n − b_m)

(the update equations needed for Alternating Least Squares):
Randomly initialize Θ
While E (Θ) has not converged do
    
    foreach n∈(1...N) (in parallel) do:
        vn  ←   (∑m∈I_n u_m*u_mT + λ_v_n * I)^−1 * ∑m∈I_n (r_m,n * u_m − b_n*u_m − b_m*u_m) 
        
    foreach m∈(1...M) (in parallel) do:
        um  ←   (∑n∈I_m v_n*v_nT + λ_u_m * I)^−1 * ∑n∈I_m (r_m,n * v_n − b_n*v_n − b_m*v_n) 
    
    foreach n∈(1...N) (in parallel) do:
        bn  ←   (λ_b_n)^−1 * ∑m∈I_n (r_m,n − u_m*v_n − b_m) 
    
    foreach m∈(1...M) (in parallel) do:
        bm  ←   (λ_b_m)^−1 * ∑n∈I_m (r_m,n − u_m*v_n − b_n) 
END
"""


class ALSHyperParameters(HyperParameters):
    """
    Section 3(b).
    -------------
    Implement a class encapsulating all the hyperparameters needed for Alternating Least Squares
    """
    def __init__(self,
                 lambda_user,
                 lambda_item,
                 lambda_bias_user,
                 lambda_bias_item,
                 eta,
                 epsilon,
                 latent_factor):
        """
        Initialization method for the SGD hyper parameters class
        :param lambda_user: Regularization value for the users
        :param lambda_item: Regularization value for the items
        :param lambda_bias_user: Regularization value for the user bias
        :param lambda_bias_item: Regularization value for the item bias
        :param eta: Epoch count, iteration count
        :param epsilon: Convergence parameter for the algorithm
        :param latent_factor: The number of latent dimensions of the vectors (d)
        """
        super(ALSHyperParameters, self).__init__(lambda_user, lambda_item, lambda_bias_user, lambda_bias_item, latent_factor)
        self.eta = eta
        self.epsilon = epsilon


class AlternatingLeastSquares(object):
    """
    Implementation of SGD algorithm for training a model
    """
    def __init__(self, model):
        """
        Initialization method for the ALS algorithm
        :param model: MFModel initialized object
        """
        self.model = model

    def train(self):
        """
        Train the MFModel according to the ALS algorithm
        """
        self.__update_vectors()
        self.__update_bias()

    # Matrix implementation
    def __update_vectors(self):
        # update user vectors
        VTV = self.model.item_vectors.T.dot(self.model.item_vectors)
        lambda_I = np.eye(VTV.shape[0]) * self.model.hyper.lambda_user
        for user in xrange(self.model.user_vectors.shape[0]):
            a = VTV + lambda_I
            c_m = np.zeros(self.model.ratings[user, :].shape[0])
            for i in np.nonzero(self.model.ratings[user, :]):
                c_m[i] = 1
            b = (self.model.ratings[user, :] - self.model.item_bias - self.model.user_bias[user]*c_m - self.model.mui*c_m).dot(self.model.item_vectors)
            self.model.user_vectors[user, :] = solve_triangular(a, b, check_finite=False)

        # update item vectors
        UTU = self.model.user_vectors.T.dot(self.model.user_vectors)
        lambda_I = np.eye(UTU.shape[0]) * self.model.hyper.lambda_item
        for item in xrange(self.model.item_vectors.shape[0]):
            a = UTU + lambda_I
            c_n = np.zeros(self.model.ratings[:, item].shape[0])
            for i in np.nonzero(self.model.ratings[:, item]):
                c_n[i] = 1
            b = (self.model.ratings[:, item].T - self.model.user_bias - self.model.item_bias[item]*c_n - self.model.mui*c_n).dot(self.model.user_vectors)
            self.model.item_vectors[item, :] = solve_triangular(a, b, check_finite=False)

    '''
    # Iterative implementation
    def __update_vectors(self):
        # calc user vector: using the formula marks z = a^-1 * b
        VTV = np.zeros((self.model.n_items, self.model.hyper.latent_factor, self.model.hyper.latent_factor))
        for item in xrange(self.model.item_vectors.shape[0]):
            VTV[item] = np.asarray([self.model.item_vectors[item, :]]).T.dot(
                np.asarray([self.model.item_vectors[item, :]]))
        for user in xrange(self.model.user_vectors.shape[0]):
            a = np.zeros((self.model.hyper.latent_factor, self.model.hyper.latent_factor))
            b = np.zeros(self.model.hyper.latent_factor)
            I_m = np.nonzero(self.model.ratings[user, :])
            if 0 == len(I_m[0]):
                continue
            for item in np.nditer(I_m):
                a += VTV[item]
                b += (self.model.ratings[user, item] - self.model.item_bias[item] - self.model.user_bias[user] - self.model.mui) * self.model.item_vectors[item, :]
            a += np.eye(self.model.hyper.latent_factor) * self.model.hyper.lambda_user
            a_inv = np.linalg.inv(a)
            self.model.user_vectors[user, :] = a_inv.dot(b)

        # calc item vector: using the formula marks z = a^-1 * b
        UTU = np.zeros((self.model.n_users, self.model.hyper.latent_factor, self.model.hyper.latent_factor))
        for user in xrange(self.model.user_vectors.shape[0]):
            UTU[user] = np.asarray([self.model.user_vectors[user, :]]).T.dot(
                np.asarray([self.model.user_vectors[user, :]]))
        for item in xrange(self.model.item_vectors.shape[0]):
            a = np.zeros((self.model.hyper.latent_factor, self.model.hyper.latent_factor))
            b = np.zeros(self.model.hyper.latent_factor)
            I_n = np.nonzero(self.model.ratings[:, item])
            if 0 == len(I_n[0]):
                continue
            for user in np.nditer(I_n):
                a += UTU[item]
                b += (self.model.ratings[user, item].T - self.model.item_bias[item] - self.model.user_bias[user] - self.model.mui) * self.model.user_vectors[user, :]
            a += np.eye(self.model.hyper.latent_factor) * self.model.hyper.lambda_item
            a_inv = np.linalg.inv(a)
            self.model.item_vectors[item, :] = a_inv.dot(b)
    '''

    # Iterative implementation
    def __update_bias(self):
        for user in xrange(self.model.user_vectors.shape[0]):
            b = np.zeros(1)
            I_m = np.nonzero(self.model.ratings[user, :])
            if 0 == len(I_m[0]):
                continue
            for item in np.nditer(I_m):
                u_m_v_n = self.model.user_vectors[user, :].T.dot(self.model.item_vectors[item, :].T)
                b_n = self.model.item_bias[item]
                b += self.model.ratings[user, item] - (self.model.mui + u_m_v_n + b_n)
            self.model.user_bias[user] = b / (I_m[0].shape[0] + self.model.hyper.lambda_bias_user)

        for item in xrange(self.model.item_vectors.shape[0]):
            b = np.zeros(1)
            I_n = np.nonzero(self.model.ratings[:, item])
            if 0 == len(I_n[0]):
                continue
            for user in np.nditer(I_n):
                u_m_v_n = self.model.user_vectors[user, :].T.dot(self.model.item_vectors[item, :].T)
                b_m = self.model.user_bias[user]
                b += self.model.ratings[user, item] - (self.model.mui + u_m_v_n + b_m)
            self.model.item_bias[item] = b / (I_n[0].shape[0] + self.model.hyper.lambda_bias_item)

        '''
    #  Matrix implementation
    def __update_bias(self):
        b_n = self.model.item_bias
        for user in xrange(self.model.user_vectors.shape[0]):
            I_m = np.nonzero(self.model.ratings[user, :])
            if 0 == len(I_m[0]):
                continue
            r_m_n = self.model.ratings[user, I_m[0]]
            u_m_v_n = self.model.user_vectors[user, :].T.dot(self.model.item_vectors[I_m[0], :].T)
            b = (r_m_n - u_m_v_n - b_n[I_m[0]] - self.model.mui).sum()
            a = I_m[0].shape[0] + self.model.hyper.lambda_bias_user
            self.model.user_bias[user] = b / a

        b_m = self.model.user_bias
        for item in xrange(self.model.item_vectors.shape[0]):
            I_n = np.nonzero(self.model.ratings[:, item])
            if 0 == len(I_n[0]):
                continue
            r_m_n = self.model.ratings[I_n[0], item]
            u_m_v_n = self.model.user_vectors[I_n[0], :].dot(self.model.item_vectors[item, :].T)
            b = (r_m_n - u_m_v_n - b_m[I_n[0]] - self.model.mui).sum()
            a = I_n[0].shape[0] + self.model.hyper.lambda_bias_item
            self.model.item_bias[item] = b / a
        '''


def LearnModelFromDataUsingALS(ratings, model, hyperparameters, additional_ratings=None):
    """
    Section 3(c). (Part 3)
    -------------
    Implement a function LearnModelFromDataUsingALS which takes in a dataset, a model and a set of algorithm
    hyperparameters and learns the model parameters using Alternating Least Squares.

    Section 1. (Part 4)
    ----------
    Modify the functions LearnModelFromDataUsingSGD and LearnModelFromDataUsingALS to output the sum of squared
    errors on the training data before each iteration to a file.

    Section 2. (Part 4)
    ----------
    Modify the two functions above to optionally take a second dataset and if given output the sum of squared errors
    on this dataset to a second file.
    :param ratings: Ratings matrix (train dataset)
    :param model: Model parameters object
    :param hyperparameters: Set of algorithm hyperparameters
    :param additional_ratings: Additional ratings matrix (test dataset)
    :return: Learnt model using ALS
    """
    model.ratings = ratings
    model.hyper = hyperparameters
    als = AlternatingLeastSquares(model)
    with open(__als_train_learning_curve__, "w") as als_train, open(__als_test_learning_curve__, "w") as als_test:
        print "Alternating Least Squares Training:"
        print "-----------------------------------"
        print "λ - User: {0}, λ - Item: {1}, λ - User Bias: {2}, λ - Item Bias: {3}, Epsilon: {4}, Epochs: {5}, Dimensions: {6}.".format(model.hyper.lambda_user,
                                                                                                                                         model.hyper.lambda_item,
                                                                                                                                         model.hyper.lambda_bias_user,
                                                                                                                                         model.hyper.lambda_bias_item,
                                                                                                                                         model.hyper.epsilon,
                                                                                                                                         model.hyper.eta,
                                                                                                                                         model.hyper.latent_factor)
        iteration = 0
        test_mse = 0

        loss_error = model.loss_error(ratings)
        previous_loss_error = loss_error + (2 * model.hyper.epsilon)

        iteration_len = 12
        training_loss_len = 15
        training_mse_len = 12
        test_mse_len = 12
        time_interval_len = 15
        print "+{0}+{1}+{2}+{3}+{4}+".format("-" * iteration_len, "-" * training_loss_len, "-" * training_mse_len, "-" * test_mse_len, "-" * time_interval_len)
        print "| {0} | {1} | {2} | {3} | {4} |".format("Iteration".ljust(iteration_len - 2),
                                                       "Error Loss".ljust(training_loss_len - 2),
                                                       "Train MSE".ljust(training_mse_len - 2),
                                                       "Test MSE".ljust(test_mse_len - 2),
                                                       "Time Interval".ljust(time_interval_len - 2))
        print "+{0}+{1}+{2}+{3}+{4}+".format("-" * iteration_len, "-" * training_loss_len, "-" * training_mse_len, "-" * test_mse_len, "-" * time_interval_len)

        timer = time.clock()

        train_mse = model.mean_squared_error(ratings)
        als_train.write("0,{0}\n".format(train_mse))

        if additional_ratings and additional_ratings.any():
            test_mse = model.mean_squared_error(additional_ratings)
            als_test.write("0,{0}\n".format(test_mse))

        print "| {0} | {1} | {2} | {3} | {4} |".format(str(iteration).ljust(iteration_len - 2),
                                                       str("%.6f" % loss_error).ljust(training_loss_len - 2),
                                                       str("%.6f" % train_mse).ljust(training_mse_len - 2),
                                                       str("%.6f" % test_mse).ljust(test_mse_len - 2),
                                                       str("%.6f" % (time.clock() - timer)).ljust(time_interval_len - 2))

        while (iteration < model.hyper.eta) and (abs(previous_loss_error - loss_error) > model.hyper.epsilon):
            iteration += 1
            timer = time.clock()

            als.train()

            train_mse = model.mean_squared_error(ratings)
            als_train.write("{0},{1}\n".format(iteration, train_mse))

            if additional_ratings and additional_ratings.any():
                test_mse = model.mean_squared_error(additional_ratings)
                als_test.write("{0},{1}\n".format(iteration, test_mse))

            previous_loss_error = loss_error
            loss_error = model.loss_error(ratings)

            print "| {0} | {1} | {2} | {3} | {4} |".format(str(iteration).ljust(iteration_len - 2),
                                                           str("%.6f" % loss_error).ljust(training_loss_len - 2),
                                                           str("%.6f" % train_mse).ljust(training_mse_len - 2),
                                                           str("%.6f" % test_mse).ljust(test_mse_len - 2),
                                                           str("%.6f" % (time.clock() - timer)).ljust(time_interval_len - 2))
        print "+{0}+{1}+{2}+{3}+{4}+".format("-" * iteration_len, "-" * training_loss_len, "-" * training_mse_len, "-" * test_mse_len, "-" * time_interval_len)
    return model
