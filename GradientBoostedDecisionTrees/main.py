"""
In this part we will put all the pieces together. Implement the main flow of your program to do
the following:
1. Read in data and split into train and test
2. Code categorical variables and impute missing values in the training data using the methods implemented in part 1
3. Use the coding and imputation maps to code and impute variables in the test data
4. Set hyperparameters based on input arguments/configuration file
5. Learn tree ensemble model from the training data using GBRT algorithm with hyperparameters specified by input arguments/configuration
   file (should output train and test performance to files)
6. Output the hyperparameters and final train and test error to a file , the file should also include the amount of time needed for training.
"""

import data_input
import data_structure
import algorithm
import deliverables
import sys
import ConfigParser
import time


def main(options):
    # Load configuration file
    config = ConfigParser.RawConfigParser()
    config.read('config.ini')
    dataset_path = config.get("Configuration", "DatasetPath")
    project_dataset = config.get("Configuration", "ProjectDatasetPath")

    # Main flow executions
    if "Main" in options:
        print "Starting Main Flow ..."

        # 1. Read in data and split into train and test
        data_frame = data_input.read_dataset(dataset_path)
        train_data_frame, test_data_frame = data_input.split_dataset_train_test(data_frame)

        # 2. Code categorical variables and impute missing values in the training data using the methods implemented in part 1
        train = data_input.Train(train_data_frame)
        train.handle_categorical_features()
        train.handle_numerical_imputation()

        # 3. Use the coding and imputation maps to code and impute variables in the test data
        test = data_input.Test(test_data_frame)
        test.handle_features(train.categorical_map_dict, train.numerical_imputation_value_dict)

        # 4. Set hyperparameters based on input arguments/configuration file

        # 5. Learn tree ensemble model from the training data using GBRT algorithm with hyperparameters
        timer = time.clock()

        training_time = time.clock() - timer

        # 6. Output the hyperparameters and final train and test error to a file , the file should also include the amount of time needed for training

    # Deliverable executions
    if "D1" in options:
        print "\nDeliverable 1"

    if "D2" in options:
        print "\nDeliverable 2"

    if "D3" in options:
        print "\nDeliverable 3"

    if "D4" in options:
        print "\nDeliverable 4"

    if "D5" in options:
        print "\nDeliverable 5"


if __name__ == '__main__':
    main(sys.argv[1:])
