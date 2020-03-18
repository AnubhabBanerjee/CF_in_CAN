from function_instances import train_functions_and_return_function_instances, predict_output
import pickle

"""
In this file with an example we show how to use the CF in your simulation. At first we need a dataset on which these CF instances 
can be trained. After the training is complete, we'll use these instances to make some predictions.
We use the training_dataset.pickle file to train the function. This dataset has been generated in such a way so that the underlying distribution
is a gaussian distribution y = np.exp(-np.power(p1, 2.) / (2 * np.power(p2, 2.)))
"""

if __name__ == "__main__":

    # at first, load the dataset (which is originally in a dict format) from the pickle file and prepare them
    # in the X and y format
    with open('training_dataset.pckl', 'rb') as dataset:
        d = pickle.load(dataset)
    X = []
    y = []
    for key, value in d.items():
        X.append(list(key))
        y.append(value)

    # after dataset is loaded, use it to train the CF instance
    poly, clf = train_functions_and_return_function_instances(X, y)

    # After training is done, test it with some random value
    print(predict_output(poly, clf, [[0, 60], [5, 65]]))