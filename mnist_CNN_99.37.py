import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import one_hot

# 设置环境变量
tf = tf.compat.v1
tf.disable_v2_behavior()

# 载入，处理数据
(x_train, y_train), (x_test, y_test) = mnist.load_data("./mnist.npz")
x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train = tf.dtypes.cast(tf.reshape(x_train, shape=[-1, 28, 28, 1]), dtype=tf.float32)
# x_test = tf.dtypes.cast(tf.reshape(x_test, shape=[-1, 28, 28, 1]),dtype=tf.float32)
y_train_hot, y_test_hot = one_hot(indices=y_train, depth=10), one_hot(indices=y_test, depth=10)


# 参数统计
def variable_summary(var):
    with tf.name_scope("summaries"):
        tf.summary.scalar("mean", tf.reduce_mean(var))
        with tf.name_scope("steddv"):
            tf.summary.scalar("stddev", tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)  # 柱状图


# 每次训练的大小
batch_size = 100
# 训练的次数
n_batch = x_train.shape[0] // batch_size

# 输入
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=[None, 28, 28], name="input_x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="input_y")
    learning_w = tf.Variable(1e-4, name="learning_w")
    keep_prob = tf.placeholder(tf.float32)

# 对新进的 x 进行转换
tansfrom_x = tf.dtypes.cast(tf.reshape(x, shape=[-1, 28, 28, 1]), dtype=tf.float32)

# 第一层卷积 5x5x1 32个 卷积核， relu激活函数， 2x2 maxpool
with tf.name_scope("convolution_layer1"):
    with tf.name_scope("convolution_layer1_w"):
        conv_w1 = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.1))  # 5x5 的窗口，图片是黑白的所以是 1 深度， 共创建32个窗口
        variable_summary(conv_w1)
    with tf.name_scope("convolution_layer1_b"):
        conv_b1 = tf.Variable(tf.random_normal(shape=[32], stddev=0.1))  # 一个卷积核对应一个偏向
        variable_summary(conv_b1)
    conv_2d_l1 = tf.nn.conv2d(input=tansfrom_x, filter=conv_w1, strides=[1, 1, 1, 1], padding='SAME', name="conv_2d_l1")
    l1 = tf.nn.relu(conv_2d_l1 + conv_b1)
    l1_pooling = tf.nn.max_pool2d(input=l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="l1_pooling")

# 第二层卷积 5x5x32 64个 卷积核, relu激活函数， 2x2 maxpool
with tf.name_scope("convolution_layer2"):
    with tf.name_scope("convolution_layer2_w"):
        conv_w2 = tf.Variable(tf.random_normal([5, 5, 8, 64], stddev=0.1))  # 5x5 的窗口， 上一层 pooling 出来的结果是32个卷积核的结果，所以是 32深度
        variable_summary(conv_w2)
    with tf.name_scope("convolution_layer2_b"):
        conv_b2 = tf.Variable(tf.random_normal([64], stddev=0.1))  # 一个卷积核对应一个偏向
        variable_summary(conv_b2)
    conv_2d_l2 = tf.nn.conv2d(input=l1_pooling, filter=conv_w2, strides=[1, 1, 1, 1], padding='SAME', name="conv_2d_l2")
    l2 = tf.nn.relu(conv_2d_l2 + conv_b2)
    l2_pooling = tf.nn.max_pool2d(input=l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='l2_pooling')

# 28x28x1 -> l1 -> 14x14x16 -> l2 -> 7x7x32

# 第一层连接层 7x7x64x500 relu激活函数 dropout 0.5
with tf.name_scope("all_connected_layer1"):
    with tf.name_scope("all_connected_layer1_w"):
        c_w3 = tf.Variable(tf.random_normal(shape=[7 * 7 * 64, 500], stddev=0.1))
        variable_summary(c_w3)
    with tf.name_scope("all_connected_layer1_b"):
        c_b3 = tf.Variable(tf.random_normal(shape=[500], stddev=0.1))
        variable_summary(c_b3)
    c_l3 = tf.nn.relu(
        tf.matmul(
            tf.reshape(l2_pooling, [-1, 7 * 7 * 64]),
            c_w3
        ) + c_b3
    )
    l3_dropout = tf.nn.dropout(x=c_l3, keep_prob=keep_prob)

# 第二层连接层 500x100 relu激活函数 dropout 0.5
with tf.name_scope("all_connected_layer2"):
    with tf.name_scope("all_connected_layer2_w"):
        c_w4 = tf.Variable(tf.random_normal(shape=[500, 100], stddev=0.1))
        variable_summary(c_w4)
    with tf.name_scope("all_connected_layer2_b"):
        c_b4 = tf.Variable(tf.random_normal(shape=[100], stddev=0.1))
        variable_summary(c_b4)
    c_l4 = tf.nn.relu(
        tf.matmul(
            l3_dropout,
            c_w4
        ) + c_b4
    )
    l4_dropout = tf.nn.dropout(x=c_l4, keep_prob=keep_prob)

# 输出层
with tf.name_scope("output"):
    with tf.name_scope("output_w"):
        output_w = tf.Variable(tf.random_normal(shape=[100, 10], stddev=0.1))
        variable_summary(output_w)
    with tf.name_scope("output_b"):
        output_b = tf.Variable(tf.random_normal(shape=[10], stddev=0.1))
        variable_summary(output_b)
    prediction = tf.nn.softmax(tf.matmul(l4_dropout, output_w) + output_b)

# 定义代价函数
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar("loss", loss)

# 定义训练步骤
with tf.name_scope("train_step"):
    train_step = tf.train.AdamOptimizer(learning_w).minimize(loss)

# 定义准确率计算
with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(y, 1),
                tf.argmax(prediction, 1)
            ),
            dtype=tf.float32
        )
    )
    tf.summary.scalar("accuracy", accuracy)

# 定义记录函数
merged = tf.summary.merge_all()

# 定义最大的准确率并初始化为0
themax_acc_testData = 0

# 定义模型保存对象
Saver = tf.train.Saver()

# 初始化所有变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # the_y_train_hot, the_y_test_hot = sess.run([y_train_hot, y_test_hot])
    the_y_train_hot = sess.run(y_train_hot)
    the_y_test_hot = sess.run(y_test_hot)
    sess.run(init)
    write = tf.summary.FileWriter(logdir="logs_tensorBoradLearning", graph=sess.graph)
    for epoch in range(101):
        for batch in range(1, n_batch):
            batch_xs, batch_ys = x_train[(batch - 1) * batch_size:batch * batch_size], the_y_train_hot[(batch - 1) * batch_size:batch * batch_size]
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            # print(sess.run(conv_2d_l1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5}).shape)
        write.add_summary(summary, epoch)
        acc_testData = sess.run(accuracy, feed_dict={x: x_test, y: the_y_test_hot, keep_prob: 1.0})
        # acc_trainData = sess.run(accuracy, feed_dict={x: x_train[0:10000], y: the_y_train_hot[0:10000], keep_prob: 1.0})
        # the_loss = sess.run(loss, feed_dict={x: x_train, y: the_y_train_hot, keep_prob: 1.0})
        print(
            "epoch: %s, acc_testData: %s" %
            (epoch, acc_testData)
        )
        if themax_acc_testData < acc_testData and epoch > 80:
            themax_acc_testData = acc_testData
            Saver.save(
                sess,
                save_path="models/" + str(themax_acc_testData) + "/tensorBoradLearningModel.ckpt",
                global_step=21
            )