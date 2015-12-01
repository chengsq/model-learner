import cPickle

import theano
import theano.tensor as T

from load_data import *

def predict():
    # load the saved model
    classifier = cPickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred
        )

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    # test_set_y = test_set_y.get_value()

    predicted_values = predict_model(test_set_x[:1000])
    print ("Predicted values for the examples in test set:")
    print predicted_values

    error_num = 0
    for x in range(1000):
        if test_set_y.eval()[x] != predicted_values[x]:
            error_num += 1
            # print '%d: %d & %d' %(x, test_set_y.eval()[x], predicted_values[x])

    print 'error num: %d, test precision: %f %%' %(error_num, (1.0*error_num/1000)*100)