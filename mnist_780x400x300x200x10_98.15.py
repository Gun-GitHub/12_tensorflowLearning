import tensorflow.compat.v1 as tf
from tensorflow import one_hot
from tensorflow.keras.datasets import mnist

# 配置好变量环境
tf.disable_v2_behavior()
# 获取数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist.npz')

# 对数据集进行简单处理
x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = x_train.reshape(x_train.shape[0], x_train.shape[-1] * x_train.shape[-2]), x_test.reshape(x_test.shape[0], x_test.shape[-1] * x_test.shape[-2])
y_train_hot, y_test_hot = one_hot(indices=y_train, depth=10), one_hot(indices=y_test, depth=10)

# 每次个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = x_train.shape[0] // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
learning_w = tf.placeholder(tf.float32)

# 第一层
w1 = tf.Variable(tf.truncated_normal(shape=[784, 400], mean=0, stddev=0.1, dtype=tf.float32))
b1 = tf.Variable(tf.truncated_normal([400], mean=0, stddev=0.1, dtype=tf.float32) + 0.1)
l1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
l1_dropout = tf.nn.dropout(l1, keep_prob)

# 第二层
w2 = tf.Variable(tf.truncated_normal(shape=[400, 300], mean=0, stddev=0.1, dtype=tf.float32))
b2 = tf.Variable(tf.truncated_normal([300], mean=0, stddev=0.1, dtype=tf.float32) + 0.1)
l2 = tf.nn.tanh(tf.matmul(l1_dropout, w2) + b2)
l2_dropout = tf.nn.dropout(l2, keep_prob)

# 第三层
w3 = tf.Variable(tf.truncated_normal(shape=[300, 200], mean=0, stddev=0.1, dtype=tf.float32))
b3 = tf.Variable(tf.truncated_normal([200], mean=0, stddev=0.1, dtype=tf.float32) + 0.1)
l3 = tf.nn.tanh(tf.matmul(l2_dropout, w3) + b3)
l3_dropout = tf.nn.dropout(l3, keep_prob)

# 第四层
w4 = tf.Variable(tf.truncated_normal(shape=[200, 10], mean=0, stddev=0.1, dtype=tf.float32))
b4 = tf.Variable(tf.truncated_normal([10], mean=0, stddev=0.1, dtype=tf.float32) + 0.1)
prediction = tf.nn.softmax(tf.matmul(l3_dropout, w4) + b4)

# 对数似然代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度下降算法优化器
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 测试使用自适应矩估计优化器
train_step = tf.train.AdamOptimizer(learning_w).minimize(loss)

# 初始化变化量
init = tf.global_variables_initializer()

# 计算准确率
accuracy = tf.reduce_mean(
    tf.cast(
        tf.equal(
            tf.argmax(y, 1),
            tf.argmax(prediction, 1)
        ),
        tf.float32
    )
)

with tf.Session() as sess:
    the_y_train_hot = sess.run(y_train_hot)
    the_y_test_hot = sess.run(y_test_hot)
    the_learning_w = 1e-3
    sess.run(init)
    for epoch in range(201):
        for batch in range(1, n_batch):
            batch_xs, batch_ys = x_train[(batch-1)*batch_size:batch*batch_size], the_y_train_hot[(batch-1)*batch_size:batch*batch_size]
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5, learning_w: the_learning_w})
        acc_testData = sess.run(accuracy, feed_dict={x: x_test, y: the_y_test_hot, keep_prob: 1.0, learning_w: the_learning_w})
        acc_trainData = sess.run(accuracy, feed_dict={x: x_train, y: the_y_train_hot, keep_prob: 1.0, learning_w: the_learning_w})
        the_loss = sess.run(loss, feed_dict={x: x_train, y: the_y_train_hot, keep_prob: 1.0, learning_w: the_learning_w})
        the_learning_w = the_loss * 1e-4
        print("epoch: %s, acc_testData: %s, acc_trainData: %s,the_loss: %s" % (epoch, acc_testData, acc_trainData, the_loss))