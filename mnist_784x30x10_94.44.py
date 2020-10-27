import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data('./mnist.npz')

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train_transfrom = []
y_test_transfrom = []

for i in y_train:
    # print(i)
    item = np.zeros(10)
    item[i] = 1
    y_train_transfrom.append(item)
y_train_transfrom = np.array(y_train_transfrom)


for i in y_test:
    # print(i)
    item = np.zeros(10)
    item[i] = 1
    y_test_transfrom.append(item)
y_test_transfrom = np.array(y_test_transfrom)

# print(y_train_transfrom)
# print(y_test_transfrom)
# 设置每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = x_train.shape[0] // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
theloss = tf.placeholder(tf.float32)
# learning_w = theloss * 10    # 二次代价函数对应的学习率
learning_w = theloss * 0.1

# 创建一个简单的神经网络
w = tf.Variable(tf.random.normal((784, 30), 0, 1/(784 * 30.0)))
b = tf.Variable(tf.random.normal((1, 30), 0, 1/30.0))
h1 = tf.nn.sigmoid(tf.matmul(x, w) + b)

# 再创建一个层次
w_1 = tf.Variable(tf.random.normal((30, 10), 0, 1/300.0))
b_1 = tf.Variable(tf.random.normal((1, 10), 0, 1/10.0))
predict = tf.nn.softmax(tf.matmul(h1, w_1) + b_1)

# 定义一个二次代价函数
# loss = tf.reduce_mean(tf.square(y - predict))
# 交叉熵代价函数
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predict))
# 对数似然代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict))

# 使用梯度下降法。
train_step = tf.train.GradientDescentOptimizer(learning_w).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 测试准确率的方法, 结果存放在一个boolearn型列表中
correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(predict, 1))  # 返回张量中最大值所在的位置

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    the_loss = 0.02
    for epoch in range(101):
        for batch in range(1, n_batch):
            batch_xs = x_train[(batch - 1) * batch_size:batch * batch_size]
            batch_ys = y_train_transfrom[(batch - 1) * batch_size:batch * batch_size]
            sess.run(train_step, feed_dict={x: batch_xs.reshape(100, 784), y: batch_ys, theloss: the_loss})
        the_loss = sess.run(loss, feed_dict={x: x_train.reshape(60000, 784), y: y_train_transfrom, theloss: the_loss})
        acc = sess.run(accuracy, feed_dict={x: x_test.reshape(10000, 784), y: y_test_transfrom, theloss: the_loss})
        acc_1 = sess.run(accuracy, feed_dict={x: x_train.reshape(60000, 784), y: y_train_transfrom})
        print("epoch: %s, acc: %s, acc_1: %s, loss: %s" % (epoch, acc, acc_1, the_loss))