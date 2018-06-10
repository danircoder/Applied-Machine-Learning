"""
Part 6. The Main Flow
---------------------
In this part we will put all the pieces together. Implement the main flow of your program to do the following:
1. Read in data and split into train and test
2. Set model and algorithm hyperparameters based on input arguments/configuration file
3. Learn model using algorithm specified by input arguments/configuration file (should output train and test
   error to files)
4. Compute the following evaluation metrics on the test set: RMSE, MPR, P@2,P@10,R@2,R@10,MAP
5. Output the hyperparameters,algorithm and metric evaluations to a file, the file should also include the
   amount of time needed for training
"""

import ConfigParser
import time
import algorithm
import model
import evaluation
import data
import diagnostics
import sys

__ALS__ = "ALS"
__SGD__ = "SGD"

__main_flow_output__ = r"output\main_flow_output.dat"
__newline__ = "\n"


def main(sections):
    ratings_train, ratings_test = None, None
    if "Main" in sections:
        # Section 1
        ratings = data.Ratings(r"dataset\ratings.dat")
        ratings_train, ratings_test = ratings.split_dataset()

        # Debug
        """
        # ratings.head(5)
        # movies.head(5)

        # ratings.show_sparsity(ratings.get_matrix(), "Matrix ")
        # ratings.show_sparsity(ratings_train, "Train Matrix ")
        # ratings.show_sparsity(ratings_test, "Test Matrix ")
        """

        # Debug
        """
        debug_ratings_train = np.array([[1, 5, 0, 0, 0],
                                        [3, 0, 1, 0, 0],
                                        [2, 3, 0, 0, 0],
                                        [3, 0, 0, 3, 0],
                                        [0, 4, 1, 0, 0],
                                        [2, 0, 2, 0, 3]])
        debug_ratings_test = np.array([[0, 0, 0, 0, 4],
                                      [0, 4, 0, 0, 0],
                                      [0, 0, 0, 4, 0],
                                      [0, 0, 2, 0, 3],
                                      [1, 0, 0, 0, 0],
                                      [0, 0, 0, 4, 0]])
        """

        # Load configuration file
        config = ConfigParser.RawConfigParser()
        config.read('config.ini')

        # Section 2
        hyper = None
        learning_algorithm = config.get("Configuration", "Algorithm")
        if learning_algorithm == __SGD__:
            hyper = algorithm.SGDHyperParameters(alpha=config.getfloat(__SGD__, "LearningRate"),
                                                 eta=config.getint(__SGD__, "Epochs"),
                                                 lambda_user=config.getfloat(__SGD__, "LambdaUser"),
                                                 lambda_item=config.getfloat(__SGD__, "LambdaItem"),
                                                 lambda_bias_user=config.getfloat(__SGD__, "LambdaUserBias"),
                                                 lambda_bias_item=config.getfloat(__SGD__, "LambdaItemBias"),
                                                 latent_factor=config.getint(__SGD__, "LatentDimensions"))
        elif learning_algorithm == __ALS__:
            hyper = algorithm.ALSHyperParameters(epsilon=config.getfloat(__ALS__, "Epsilon"),
                                                 eta=config.getint(__ALS__, "Epochs"),
                                                 lambda_user=config.getfloat(__ALS__, "LambdaUser"),
                                                 lambda_item=config.getfloat(__ALS__, "LambdaItem"),
                                                 lambda_bias_user=config.getfloat(__ALS__, "LambdaUserBias"),
                                                 lambda_bias_item=config.getfloat(__ALS__, "LambdaItemBias"),
                                                 latent_factor=config.getint(__ALS__, "LatentDimensions"))
        model_object = model.MFModel(ratings_train, hyper)

        # Section 3
        timer = time.clock()
        if learning_algorithm == __SGD__:
            model_object = algorithm.LearnModelFromDataUsingSGD(ratings_train, model_object, hyper, ratings_test)
        elif learning_algorithm == __ALS__:
            model_object = algorithm.LearnModelFromDataUsingALS(ratings_train, model_object, hyper, ratings_test)
        training_time = time.clock() - timer

        # Section 4
        ranked_items_list = evaluation.get_ranked_items_list(model_object, ratings_test)
        rmse = evaluation.root_mean_squared_error(ranked_items_list, ratings_test)  # RMSE
        mpr = evaluation.mean_percentile_rank(ranked_items_list, ratings_test)  # MPR
        p_at_2 = evaluation.precision_at_K(ranked_items_list, ratings_test, 2)  # P@2
        p_at_10 = evaluation.precision_at_K(ranked_items_list, ratings_test, 10)  # P@10
        r_at_2 = evaluation.recall_at_K(ranked_items_list, ratings_test, 2)  # R@2
        r_at_10 = evaluation.recall_at_K(ranked_items_list, ratings_test, 10)  # R10
        map_ = evaluation.mean_average_precision(ranked_items_list, ratings_test)  # MAP

        # Section 5
        with open(__main_flow_output__, "w") as output_handle:
            output_handle.write("Algorithm: {0}{1}".format(learning_algorithm, __newline__))
            output_handle.write("LambdaUser: {0}{1}".format(hyper.lambda_user, __newline__))
            output_handle.write("LambdaItem: {0}{1}".format(hyper.lambda_item, __newline__))
            output_handle.write("LambdaUserBias: {0}{1}".format(hyper.lambda_bias_user, __newline__))
            output_handle.write("LambdaItemBias: {0}{1}".format(hyper.lambda_bias_item, __newline__))
            output_handle.write("LatentDimensions: {0}{1}".format(hyper.latent_factor, __newline__))
            output_handle.write("Epochs: {0}{1}".format(hyper.eta, __newline__))
            if learning_algorithm == __SGD__:
                output_handle.write("LearningRate: {0}{1}".format(hyper.alpha, __newline__))
            elif learning_algorithm == __ALS__:
                output_handle.write("Epsilon: {0}{1}".format(hyper.epsilon, __newline__))
            output_handle.write("RMSE: {0}{1}".format(rmse, __newline__))
            output_handle.write("MPR: {0}{1}".format(mpr, __newline__))
            output_handle.write("P@2: {0}{1}".format(p_at_2, __newline__))
            output_handle.write("P@10: {0}{1}".format(p_at_10, __newline__))
            output_handle.write("R@2: {0}{1}".format(r_at_2, __newline__))
            output_handle.write("R@10: {0}{1}".format(r_at_10, __newline__))
            output_handle.write("MAP: {0}{1}".format(map_, __newline__))
            output_handle.write("TrainingTime: {0}{1}".format(training_time, __newline__))

    print "Finished Main Flow\n"

    """
    Part 7. Diagnostics and Analysis
    """
    print "Starting Diagnostics and Analysis"

    if not (ratings_train or ratings_test):
        ratings = data.Ratings(r"dataset\ratings.dat")
        ratings_train, ratings_test = ratings.split_dataset()

    if "D1" in sections:
        print "\nDeliverable 1"
        diagnostics.deliverable1()

    if "D2" in sections:
        print "\nDeliverable 2"
        diagnostics.deliverable2(ratings_train, ratings_test)

    if "D3" in sections:
        print "\nDeliverable 3"
        diagnostics.deliverable3(ratings_train, ratings_test)

    if "D4" in sections:
        print "\nDeliverable 4"
        diagnostics.deliverable4(ratings_train)

    if "D5" in sections:
        print "\nDeliverable 5"
        movies = data.Movies(r"dataset\movies.dat")
        movies_dict = movies.get_dictionary()
        diagnostics.deliverable5(ratings_train, movies_dict)

    print "Finished Diagnostics and Analysis"


if __name__ == '__main__':
    main(sys.argv[1:])
