"""
In this part we will put all the pieces together.
Implement the main flow of your program to do the following:
1. Read in data and split into train and test
2. Set hyperparameters based on input arguments/configuration file
3. Learn model using algorithm specified by input arguments/configuration file (should output train and test performance to files)
4. Output the hyperparameters and final (mean) log-likelihoods to a file, the file should also include the amount of time needed for training.
"""

import sys
import ConfigParser
import algorithm
import data
import deliverables
import model


def main(options):
    # Load configuration file
    config = ConfigParser.RawConfigParser()
    config.read('config.ini')
    sentences_path = config.get("Configuration", "DatasetSentencesPath")
    split_path = config.get("Configuration", "DatasetSplitPath")

    # Section 1
    # Loading datasets
    corpus_split = data.CorpusSplit(split_path)
    corpus_sentences = data.CorpusSentences(sentences_path, corpus_split)

    model_object = None
    parameters = "size of the word embedding (d) = 50\n Num iterations = 10000\n Maximum context window size = 5\n mini-batch size = 50\n noise distribution alpha = 0.01\n number of negative samples per context/input pair K=10\n alpha=0.01\n X=1000"

    if "Main" in options:
        # Section 2 - using this trained model in Deliverable 4-6
        model_hyper = model.ModelHyperParameters(C=5,
                                                 d=50,
                                                 K=10,
                                                 alpha=0.01,
                                                 seed=123.0)
        model_object = model.SkipGramModel(model_hyper)
        model_object.init(corpus_sentences.train)

        # Section 3 - using this trained model in Deliverable 4
        sgd_hyper = algorithm.SGDHyperParameters(alpha=0.01,
                                                 mini_batch_size=50,
                                                 X=1000)
        sgd = algorithm.LearnParamsUsingSGD(training_set=corpus_sentences.train,
                                            hyper_parameters=sgd_hyper,
                                            model=model_object,
                                            iterations_number=10000,
                                            test_set=corpus_sentences.test)

    if "D1" in options:
        print "\nDeliverable 1"
        deliverables.deliverable1(corpus_sentences.train, corpus_sentences.test)

    if "D2" in options:
        print "\nDeliverable 2"
        deliverables.deliverable2(corpus_sentences.train, corpus_sentences.test)

    if "D3" in options:
        print "\nDeliverable 3"
        deliverables.deliverable3(corpus_sentences.train, corpus_sentences.test)

    if "D4" in options:
        print "\nDeliverable 4"
        deliverables.deliverable4(model_object, parameters)

    if "D5" in options:
        print "\nDeliverable 5"
        deliverables.deliverable5(model_object)

    if "D6" in options:
        print "\nDeliverable 6"
        deliverables.deliverable6(model_object)


if __name__ == '__main__':
    main(sys.argv[1:])
