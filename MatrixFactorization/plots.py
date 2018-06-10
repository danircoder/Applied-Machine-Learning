# coding=utf-8
"""
Part 7. Diagnostics and Analysis
** This is the Deliverables Plots implementation **
In this part you will use the code you implemented above to create several plots (marked by
green Deliverable tag ) that you will turn in along with your code
"""

import pandas as pd
import matplotlib.pyplot as plt


def deliverable1_plot(sgd_train_learning_curve_path, sgd_test_learning_curve_path,
                      als_train_learning_curve_path, als_test_learning_curve_path):
    sgd_train_learning_curve_path = "sgd_train_learning_curve.dat"
    sgd_test_learning_curve_path = "sgd_test_learning_curve.dat"
    sgd_train_learning_curve_df = pd.read_csv(sgd_train_learning_curve_path, header=None, names={"Iteration", "Error"})
    sgd_test_learning_curve_df = pd.read_csv(sgd_test_learning_curve_path, header=None, names={"Iteration", "Error"})
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(sgd_train_learning_curve_df['Iteration'], sgd_train_learning_curve_df['Error'], 'r', label="train error")
    ax.plot(sgd_test_learning_curve_df['Iteration'], sgd_test_learning_curve_df['Error'], 'b', label="test error")
    ax.legend()
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("MSE for iteration")
    first_line = "LambdaUser=0.1, LambdaItem=0.1, LambdaUserBias=0.1,\n LambdaItemBias=0.1, Epochs=50, LearningRate=0.01, LatentDimensions=20"
    _title = unicode('SGD Train-Test Error. \nParameters: ') + first_line
    ax.set_title(_title)

    als_train_learning_curve_path = "als_train_learning_curve.dat"
    als_test_learning_curve_path = "als_test_learning_curve.dat"
    als_train_learning_curve_df = pd.read_csv(als_train_learning_curve_path, header=None, names={"Iteration", "Error"})
    als_test_learning_curve_df = pd.read_csv(als_test_learning_curve_path, header=None, names={"Iteration", "Error"})
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(als_train_learning_curve_df['Iteration'], als_train_learning_curve_df['Error'], 'r', label="train error")
    ax.plot(als_test_learning_curve_df['Iteration'], als_test_learning_curve_df['Error'], 'b', label="test error")
    ax.legend()
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("MSE for iteration")

    first_line = "LambdaUser=0.1, LambdaItem=0.1, LambdaUserBias=0.1,\n LambdaItemBias=0.1, Epochs=50, Epsilon=1, LatentDimensions=20"
    _title = unicode('ALS Train-Test Error. \nParameters: ') + first_line
    ax.set_title(_title)


def deliverable2_plot(lambda_analysis_output_path):
    with open(lambda_analysis_output_path) as f:
        first_line = f.readline()
    lambda_analysis_output_df = pd.read_csv(lambda_analysis_output_path, header=None, skiprows=4,
                                            names={"_lambda", "log_lambda", "Ea", "Eb"})

    _title = unicode('RMSE error. \nParameters: ') + first_line.decode('utf-8')
    ax = lambda_analysis_output_df.plot(x='log_lambda', y='Ea', title=_title)
    ax.set_xlabel("log(lambda)")
    ax.set_ylabel("value of error")

    _title = unicode('p@k error. \nParameters: ') + first_line.decode('utf-8')
    ax = lambda_analysis_output_df.plot(x='log_lambda', y='Eb', title=_title)
    ax.set_xlabel("log(lambda)")
    ax.set_ylabel("value of error")


def deliverable3_plot(latent_dimensions_analysis_output_path):
    with open(latent_dimensions_analysis_output_path) as f:
        first_line = f.readline()
    deliverable3_df = pd.read_csv(latent_dimensions_analysis_output_path, header=None, skiprows=4, names={"_dimensions", "Ea", "Eb"})

    _title = unicode('RMSE as func of dimensions. \nParameters: ') + first_line.decode('utf-8')
    ax = deliverable3_df.plot(x='_dimensions', y='Ea', title=_title)
    ax.set_xlabel("dimensions")
    ax.set_ylabel("value of error")

    _title = unicode('p@k as func of dimensions. \nParameters: ') + first_line.decode('utf-8')
    ax = deliverable3_df.plot(x='_dimensions', y='Eb', title=_title)
    ax.set_xlabel("dimensions")
    ax.set_ylabel("value of error")


def deliverable4_plot(latent_dimensions_analysis_output_path):
    with open(latent_dimensions_analysis_output_path) as f:
        first_line = f.readline()
    deliverable4_df = pd.read_csv(latent_dimensions_analysis_output_path, header=None, skiprows=4, names={"_dimensions", "RunTime"})

    _title = unicode('RunTime as func of dimensions. \nParameters: ') + first_line.decode('utf-8')
    ax = deliverable4_df.plot(x='_dimensions', y='RunTime', title=_title)
    ax.set_xlabel("dimensions")
    ax.set_ylabel("RunTime")




