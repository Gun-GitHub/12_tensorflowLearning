import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import one_hot

# 程序环境设置
tf = tf.compat.v1
tf.disable_v2_behavior()

# 数据加载
(x_train, y_train), (x_test, y_test) = mnist.load_data("mnist.npz")
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = one_hot(y_train, 10), one_hot(y_test, 10)

# 定义基础参数
n_input = 28  # 输入一行，一行有28个数据
max_time = 28  # 一共28行
lstm_size = 100  # 隐藏层单元
n_class = 10  # 分类个数
batch_size = 50  # 批次样本个数
n_batch = x_train.shape[0] // batch_size  # 计算一共有多少个样本

# 定义placeholder
x = tf.placeholder(tf.float32, [None, 28, 28])
y = tf.placeholder(tf.float32, [None, 10])
learning_w = tf.Variable(1e-4)

# 初始化权值
weight = tf.Variable(tf.truncated_normal([lstm_size, n_class], stddev=0.1))
# 初始化偏置值
biase = tf.Variable(tf.truncated_normal([n_class], stddev=0.1))


# 定义RNN
def RNN(x, weight, biase):
    inputs = tf.reshape(x, [-1, max_time, n_input])
    # 定义LSTM基本Cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)  # lstm_size 表示 cell 输出数量
    out_puts, final_stats = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    result = tf.nn.softmax(tf.matmul(final_stats[1], weight) + biase)
    return result


# 计算RNN的返回结果
pridict = RNN(x, weight, biase)

# 设计loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pridict))

# 设计训练步骤
train_step = tf.train.AdamOptimizer(learning_w).minimize(loss)

# 计算准确率
accuracy = tf.reduce_mean(
    tf.cast(
        tf.equal(
            tf.argmax(pridict, 1),
            tf.argmax(y, 1)
        ),
        tf.float32
    )
)

# 初始化所有变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    y_train_hot, y_test_hot = sess.run([y_train, y_test])
    for epoch in range(21):
        for batch in range(1, n_batch):
            batch_xs, batch_ys = x_train[(batch-1) * batch_size:batch * batch_size], y_train_hot[(batch-1) * batch_size:batch * batch_size]
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test_hot})
        print("epoch: %s, acc: %s " % (epoch, acc))