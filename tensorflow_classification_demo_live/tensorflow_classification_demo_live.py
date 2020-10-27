"""
DOCSTRING
"""
import pandas
import tensorflow

dataframe = pandas.read_csv('data.csv')
dataframe = dataframe.drop(['index', 'price', 'sq_price'], axis=1)
dataframe = dataframe[0:10]
print(dataframe)
dataframe.loc[:, ('y1')] = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
dataframe.loc[:, ('y2')] = dataframe['y1'] == 0
dataframe.loc[:, ('y2')] = dataframe['y2'].astype(int)
print(dataframe)
inputX = dataframe.loc[:, ['area', 'bathrooms']].as_matrix()
inputY = dataframe.loc[:, ['y1', 'y2']].as_matrix()
print(inputX)
print(inputY)
# parameters
learning_rate = 0.000001
training_epochs = 2000
display_step = 50
n_samples = inputY.size
x = tensorflow.placeholder(tensorflow.float32, [None, 2])
W = tensorflow.Variable(tensorflow.zeros([2, 2]))
b = tensorflow.Variable(tensorflow.zeros([2]))
y_values = tensorflow.add(tensorflow.matmul(x, W), b)
y = tensorflow.nn.softmax(y_values)
y_ = tensorflow.placeholder(tensorflow.float32, [None, 2])
# cmean squared error
cost = tensorflow.reduce_sum(tensorflow.pow(y_ - y, 2)) / (2 * n_samples)
# gradient descent
optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# initialize variabls and tensorflow session
init = tensorflow.initialize_all_variables()
sess = tensorflow.Session()
sess.run(init)
for i in range(training_epochs):
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY})
    if (i) % display_step == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_:inputY})
        print('Training step:', '%04d' % (i), 'cost=', '{:.9f}').format(cc)
print('Optimization complete.')
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print('Training cost=', training_cost, 'W=', sess.run(W), 'b=', sess.run(b), '\n')
sess.run(y, feed_dict={x: inputX})
sess.run(tensorflow.nn.softmax([1.0, 2.0]))
