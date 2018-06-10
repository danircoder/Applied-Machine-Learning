# coding=utf-8

"""
In order to make sure our algorithm is implemented correctly we want to see that the training log-likelihood is being increased over time in our algorithm
"""

import pandas as pd
import matplotlib.pyplot as plt
import time
import model
import algorithm
import evaluation

__sgd_train_learning_curve_path__ = "output\sgd_mini_batch_log_L.dat"
__sgd_test_learning_curve_path__ = "output\sgd_test_log_L.dat"
__dimensions_analysis_time__ = r"output\dimensions_analysis_time.dat"
__dimensions_analysis_train__ = r"output\dimensions_analysis_train.dat"
__dimensions_analysis_test__ = r"output\dimensions_analysis_test.dat"
__learning_rate_analysis_time__ = r"output\learning_rate_analysis_time.dat"
__learning_rate_analysis_train__ = r"output\learning_rate_analysis_train.dat"
__learning_rate_analysis_test__ = r"output\learning_rate_analysis_test.dat"


def plot_train_test(__sgd_train_learning_curve_path__, __sgd_test_learning_curve_path__, parameters, output_filename):
    sgd_train_learning_curve_df = pd.read_csv(__sgd_train_learning_curve_path__, header=None, names=["time", "Iteration", "log_Likelihood", "avg_log_Likelihood"])
    sgd_test_learning_curve_df = pd.read_csv(__sgd_test_learning_curve_path__, header=None, names=["time", "Iteration", "log_Likelihood", "avg_log_Likelihood"])
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(sgd_train_learning_curve_df['Iteration'], sgd_train_learning_curve_df['avg_log_Likelihood'], 'r', label="train log(L)")
    ax.plot(sgd_test_learning_curve_df['Iteration'], sgd_test_learning_curve_df['avg_log_Likelihood'], 'b', label="test log(L)")
    ax.legend()
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("avg log(L) for iteration")
    _title = unicode('SGD Train-Test log-Likelihood. \nParameters: ') + parameters
    ax.set_title(_title)
    fig.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)


def plot_train_test_separate(__sgd_train_learning_curve_path__, __sgd_test_learning_curve_path__, parameters, output_filename):
    sgd_train_learning_curve_df = pd.read_csv(__sgd_train_learning_curve_path__, header=None, skiprows=0, names=["time", "Iteration", "log_Likelihood", "avg_log_Likelihood"])
    sgd_test_learning_curve_df = pd.read_csv(__sgd_test_learning_curve_path__, header=None, skiprows=0, names=["time", "Iteration", "log_Likelihood", "avg_log_Likelihood"])

    _title = unicode('SGD Train avg-log-Likelihood. \nParameters: ') + parameters.decode('utf-8')
    ax = sgd_train_learning_curve_df.plot(x='Iteration', y='avg_log_Likelihood', title=_title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("avg-log-Likelihood")
    plt.savefig(output_filename.replace(".png", "_train.png"), bbox_inches='tight')

    _title = unicode('SGD Test avg-log-Likelihood. \nParameters: ') + parameters.decode('utf-8')
    ax = sgd_test_learning_curve_df.plot(x='Iteration', y='avg_log_Likelihood', title=_title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("avg-log-Likelihood")
    plt.savefig(output_filename.replace(".png", "_test.png"), bbox_inches='tight')


def deliverable1(train, test):
    """
    Plot the log-likelihood train and test as a function of training iteration.
    Train and test plots should appear in the same figure, clearly marked.
    You may use whatever configuration of hyperparameters you wish for this plot, but make sure you specify the choice.
    Hyperparameters can sometimes have a great effect on generalization performance.
    We will consider the effect of the embedding size and other hyperparams.
    """
    parameters = "size of the word embedding d=50\nLearning Rate = 0.1\n Num iterations = 20000\n Maximum context window size = 5\n mini-batch size = 50\n noise distribution alpha = 0.01\n number of negative samples per context/input pair K=10\n X=100"

    model_hyper = model.ModelHyperParameters(C=5,
                                             d=50,
                                             K=10,
                                             alpha=0.01, # TODO: noise distribution Uniform (α = 0) (or some other small value of α e.g. 0.01)
                                             seed=1.0)
    model_object = model.SkipGramModel(model_hyper)
    model_object.init(train)
    sgd_hyper = algorithm.SGDHyperParameters(alpha=0.1, # TODO: Learning Rate  0.01 − 1.0
                                             mini_batch_size=50,
                                             X=100)
    timer = time.clock()
    sgd = algorithm.LearnParamsUsingSGD(training_set=train,
                                        hyper_parameters=sgd_hyper,
                                        model=model_object,
                                        iterations_number=500, # TODO: set 20000 before submit
                                        test_set=test)

    plot_train_test(__sgd_train_learning_curve_path__,
                    __sgd_test_learning_curve_path__,
                    parameters=parameters,
                    output_filename="output/deliverable1_united.png")
    plot_train_test_separate(__sgd_train_learning_curve_path__,
                             __sgd_test_learning_curve_path__,
                             parameters=parameters,
                             output_filename="output/deliverable1_separated.png")


def deliverable2(train, test):
    """
    Set the hyper params as follows:
        1. Learning Rate = 0.3
        2. Num iterations = 20000
        3. Maximum context window size = 5
        4. mini-batch size = 50
        5. noise distribution = Unigram (alpha = 1)
        6. number of negative samples per context/input pair (denoted K above)=10
    Vary the size of the word embedding ,d from 10 to 300 in 5 evenly spaced intervals.
    In two separate plots, plot both training time and train and test (mean) log-likelihood as a function of d.
    All hyper-parameter configurations of the algorithm should be clearly specified.
    """
    parameters = "Learning Rate = 0.01\n Num iterations = 20000\n Maximum context window size = 5\n mini-batch size = 50\n noise distribution = Unigram (alpha = 1)\n number of negative samples per context/input pair K=10\n X=1000"
    min_d = 10  # min_d = 10
    max_d = 300  # max_d = 300
    d_adder = 70  # d_adder = 70
    d_value = min_d
    print "dimensions Analysis:"
    with open(__dimensions_analysis_time__, 'w') as dimensions_analysis_time, open(__dimensions_analysis_train__, 'w') as dimensions_analysis_train, open(__dimensions_analysis_test__, 'w') as dimensions_analysis_test:
        while d_value <= max_d:
            model_hyper = model.ModelHyperParameters(C=5,
                                                     d=d_value,
                                                     K=10,
                                                     alpha=1.0,
                                                     seed=1.0)
            model_object = model.SkipGramModel(model_hyper)
            model_object.init(train)
            sgd_hyper = algorithm.SGDHyperParameters(alpha=0.01,
                                                     mini_batch_size=50,
                                                     X=1000)
            timer = time.clock()
            sgd = algorithm.LearnParamsUsingSGD(training_set=train,
                                                hyper_parameters=sgd_hyper,
                                                model=model_object,
                                                iterations_number=20000,
                                                test_set=test)

            # output training time to file
            training_time = time.clock() - timer
            print "Dimension: {0},Training Time: {1}".format(d_value, training_time)
            dimensions_analysis_time.write("{0},{1}\n".format(d_value, training_time))

            # output train and test MEAN log-likelihood
            sgd_train_learning_curve_df = pd.read_csv(__sgd_train_learning_curve_path__, header=None,
                                                      names=["time", "Iteration", "log_Likelihood",
                                                             "avg_log_Likelihood"])
            sgd_test_learning_curve_df = pd.read_csv(__sgd_test_learning_curve_path__, header=None,
                                                     names=["time", "Iteration", "log_Likelihood",
                                                            "avg_log_Likelihood"])
            train_mean_log_likelihood = sgd_train_learning_curve_df["avg_log_Likelihood"].mean()
            test_mean_log_likelihood = sgd_test_learning_curve_df["avg_log_Likelihood"].mean()
            print "Dimension: {0},train_mean_log_likelihood: {1}".format(d_value, train_mean_log_likelihood)
            print "Dimension: {0},test_mean_log_likelihood: {1}".format(d_value, test_mean_log_likelihood)
            dimensions_analysis_train.write("{0},{1}\n".format(d_value, train_mean_log_likelihood))
            dimensions_analysis_test.write("{0},{1}\n".format(d_value, test_mean_log_likelihood))

            d_value += d_adder

    # 1. plot training time as func of d
    dimensions_analysis_time_df = pd.read_csv(__dimensions_analysis_time__, header=None, skiprows=0, names=["d", "time"])
    _title = unicode('SGD training time as a function of d\nParameters: ') + parameters.decode('utf-8')
    ax = dimensions_analysis_time_df.plot(x='d', y='time', title=_title)
    ax.set_xlabel("d - size of the word embedding")
    ax.set_ylabel("training time")
    plt.savefig("output\deliverable2_training_time.png", bbox_inches='tight')

    # 2. train and test (mean) log-likelihood as a function of d
    sgd_train_learning_curve_df = pd.read_csv(__dimensions_analysis_train__, header=None, names=["d", "mean_log_likelihood"])
    sgd_test_learning_curve_df = pd.read_csv(__dimensions_analysis_test__, header=None, names=["d", "mean_log_likelihood"])
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(sgd_train_learning_curve_df['d'], sgd_train_learning_curve_df['mean_log_likelihood'], 'r', label="train mean log(L)")
    ax.plot(sgd_test_learning_curve_df['d'], sgd_test_learning_curve_df['mean_log_likelihood'], 'b', label="test mean log(L)")
    ax.legend()
    ax.set_xlabel("d - size of the word embedding")
    ax.set_ylabel("mean_log_likelihood")
    _title = unicode('SGD Train-Test mean log-Likelihood as a function of d\nParameters: ') + parameters.decode('utf-8')
    ax.set_title(_title)
    fig.savefig("output\deliverable2_mean_log_likelihood.png", bbox_inches='tight')
    plt.close(fig)


def deliverable3(train, test):
    """
    Repeat the setup above, but this time fixing d (to your choice) and varying one of {learning rate , mini-batch size, noise distribution}
    (again your choice).
    As before, generate two plots one for training time and one for train and test log-likelihood.
    Clearly specify all your choices.
    """
    parameters = "size of the word embedding (d) = 50\n Num iterations = 20000\n Maximum context window size = 5\n mini-batch size = 50\n noise distribution = Unigram (alpha = 1)\n number of negative samples per context/input pair (denoted K above)=10\n X=1000"
    min_learning_rate = 0.00001
    max_learning_rate = 0.1
    learning_rate_mul = 10
    learning_rate_value = min_learning_rate
    print "dimensions Analysis:"
    with open(__learning_rate_analysis_time__, 'w') as learning_rate_analysis_time, open(__learning_rate_analysis_train__, 'w') as learning_rate_analysis_train, open(__learning_rate_analysis_test__, 'w') as learning_rate_analysis_test:
        while learning_rate_value <= max_learning_rate:
            model_hyper = model.ModelHyperParameters(C=5,
                                                     d=50,
                                                     K=10,
                                                     alpha=1.0,
                                                     seed=1.0)
            model_object = model.SkipGramModel(model_hyper)
            model_object.init(train)
            sgd_hyper = algorithm.SGDHyperParameters(alpha=learning_rate_value,
                                                     mini_batch_size=50,
                                                     X=1000)
            timer = time.clock()
            sgd = algorithm.LearnParamsUsingSGD(training_set=train,
                                                hyper_parameters=sgd_hyper,
                                                model=model_object,
                                                iterations_number=20000,
                                                test_set=test)

            # output training time to file
            training_time = time.clock() - timer
            print "Dimension: {0},Training Time: {1}".format(learning_rate_value, training_time)
            learning_rate_analysis_time.write("{0},{1}\n".format(learning_rate_value, training_time))

            # output train and test MEAN log-likelihood
            sgd_train_learning_curve_df = pd.read_csv(__sgd_train_learning_curve_path__, header=None,
                                                      names=["time", "Iteration", "log_Likelihood",
                                                             "avg_log_Likelihood"])
            sgd_test_learning_curve_df = pd.read_csv(__sgd_test_learning_curve_path__, header=None,
                                                     names=["time", "Iteration", "log_Likelihood",
                                                            "avg_log_Likelihood"])
            train_mean_log_likelihood = sgd_train_learning_curve_df["avg_log_Likelihood"].mean()
            test_mean_log_likelihood = sgd_test_learning_curve_df["avg_log_Likelihood"].mean()
            print "Dimension: {0},train_mean_log_likelihood: {1}".format(learning_rate_value, train_mean_log_likelihood)
            print "Dimension: {0},test_mean_log_likelihood: {1}".format(learning_rate_value, test_mean_log_likelihood)
            learning_rate_analysis_train.write("{0},{1}\n".format(learning_rate_value, train_mean_log_likelihood))
            learning_rate_analysis_test.write("{0},{1}\n".format(learning_rate_value, test_mean_log_likelihood))

            learning_rate_value *= learning_rate_mul

    # 1. plot training time as func of learning_rate
    learning_rate_analysis_time_df = pd.read_csv(__learning_rate_analysis_time__, header=None, skiprows=0, names=["learning_rate", "time"])
    _title = unicode('SGD training time as a function of learning_rate\nParameters: ') + parameters.decode('utf-8')
    ax = learning_rate_analysis_time_df.plot(x='learning_rate', y='time', title=_title)
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("training time")
    plt.savefig("output\deliverable3_training_time.png", bbox_inches='tight')

    # 2. train and test (mean) log-likelihood as a function of d
    sgd_train_learning_curve_df = pd.read_csv(__learning_rate_analysis_train__, header=None, names=["learning_rate", "mean_log_likelihood"])
    sgd_test_learning_curve_df = pd.read_csv(__learning_rate_analysis_test__, header=None, names=["learning_rate", "mean_log_likelihood"])
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(sgd_train_learning_curve_df['learning_rate'], sgd_train_learning_curve_df['mean_log_likelihood'], 'r', label="train mean log(L)")
    ax.plot(sgd_test_learning_curve_df['learning_rate'], sgd_test_learning_curve_df['mean_log_likelihood'], 'b', label="test mean log(L)")
    ax.legend()
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("mean_log_likelihood")
    _title = unicode('SGD Train-Test mean log-Likelihood as a function of learning_rate\nParameters: ') + parameters.decode('utf-8')
    ax.set_title(_title)
    fig.savefig("output\deliverable3_mean_log_likelihood.png", bbox_inches='tight')
    plt.close(fig)


def deliverable4(model_object, parameters):
    """
    Consider the following words: {good, bad, lame, cool, exciting}
        1. Print the top 10 context words according to the model when each of the above is considered to be an input word
        2. Learn a model with d = 2, show a scatter plot visualizing the embedding of the above words.
           Add additional words of your choice to the plot.
           Try using both the input and the context embeddings.
           Clearly specify all hyperparameters used.
    """
    input_words = {"good", "bad", "lame", "cool", "exciting", "actors", "director", "movie", "storyline"}

    # 1. Print the top 10 context words according to the model when each of the above is considered to be an input word
    for input_word in input_words:
        evaluation.predict_contexts(model=model_object, target_word=input_word)

    # 2. Learn a model with d = 2, show a scatter plot visualizing the embedding of the above words.
    evaluation.scatter_plot_words_vector(model=model_object, words=input_words, parameters=parameters, output_path="output\deliverable4_scatter.png")


def deliverable5(model_object):
    """
    Consider the following incomplete sentences:
        1. The "movie" was surprisingly ___ .
        2. ___ was really "disappointing".
        3. Knowing that she ___ was the "best" part.
    Learn a model with your choice of hyperparameters (you may use one from previous sections).
    Use the model to complete the sentences (print 10 best completion) by considering the non-blank words as context.
    Clearly specify all hyperparameters used.
    """
    sentences = ["The movie was surprisingly", "was really disappointing", "Knowing that she was the best part"]

    for sentence in sentences:
        print "for sentence:\t\t {0},\t\t the 10 best completion are:".format(sentence)
        evaluation.predict_target(model=model_object, context_words=sentence.lower().split())


def deliverable6(model_object):
    """
    Consider the following analogy questions:
        1. man is to woman as men is to
        2. good is to great as bad is to
        3. warm is to cold as summer is to
    Learn a model with your choice of hyperparameters (you may use one from previous sections).
    Use the model to answer the analogy questions, using the linear approach specified above (print 10 best completions).
    What happens when you use context embeddings instead of input embeddings?
    Clearly specify all hyperparameters used.

    top_10_likely_contexts = evaluation.predict_contexts(model=model_object, target_word="good")
    top_10_likely_targets = evaluation.predict_target(model=model_object, context_words={"actors", "great", "good"})
    evaluation.scatter_plot_words_vector(model=model_object,
                                         words=["actors", "actor", "paycheck", "famed", "nicely", "great", "good",
                                                "bille", "banana", "yellow", "sun"])
    top_10_likely_last_word = evaluation.analogy_solver(model_object, "man", "woman", "king")
    """
    sentences = ["man is to woman as men is to", "good is to great as bad is to", "warm is to cold as summer is to"]
    for sentence in sentences:
        print "for analogy:\t\t {0},\t\t the 10 best completion are:".format(sentence)
        print "using input embeddings:"
        evaluation.analogy_solver(model_object, sentence.lower().split()[0], sentence.lower().split()[3], sentence.lower().split()[5])
        print "using context embeddings instead of input embeddings:"
        evaluation.analogy_solver_context_embeddings(model_object, sentence.lower().split()[0], sentence.lower().split()[3],sentence.lower().split()[5])
