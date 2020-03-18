from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def train_functions_and_return_function_instances(X, y):
    """
    We use a polynomial regression block to model the CFs. X is the input to the polynomial block and
    y is its output. However, for training this block, we need both X and y at the same time. The block
    looks like -

            p1 ------>  |...........|
                        |           |-------> y
            p2 ------>  |...........|

     No matter which dataset you use for training, both X and y should have the following structures:
    structure of X: [[p11, p21], [p12, p22], [p13, p23], ....]
    structure of y: [y1, y2, y3]
    The output are two function instances which is used in the predict_output function
    """
    poly = PolynomialFeatures(degree=5)
    X_ = poly.fit_transform(X)
    clf = LinearRegression()
    clf.fit(X_, y)
    return [poly, clf]

def predict_output(poly, clf, test):
    """
    This function is called after a CF instance is generated using the "train_functions_and_return_function_instances"
    method. We get both poly and clf from the previous method. "Test" is the testing parameter on which wa want to
    test the performance of the predictor. So, the idea of having this function is, we'll give a set of parameter
    values [p1, p2] and it'll be able to predict the output y.
    test should have the following structure: [[p11, p21], [p12, p22], [p13, p23], ....]
    output will have the following structure: [y1, y2, y3]
    """
    predict_ = poly.fit_transform(test)
    return clf.predict(predict_).tolist()