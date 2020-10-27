import tensorflow as tf
from tensorflow import one_hot
from tensorflow.keras.datasets import mnist

# 设置环境
tf = tf.compat.v1
tf.disable_v2_behavior()
# 读取数据
(x_train, y_train), (x_test, y_test) = mnist.load_data("mnist.npz")

# 数据处理
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], x_train.shape[-2] * x_train.shape[-1])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[-2] * x_test.shape[-1])
y_train_hot, y_test_hot = one_hot(indices=y_train, depth=10), one_hot(indices=y_test, depth=10)

# 每个批次的大小
batch_size = 100
# 一共有多少个批次
n_batch = x_train.shape[0] // batch_size


# 参数统计
def variable_summary(var):
    with tf.name_scope("summaries"):
        tf.summary.scalar("mean", tf.reduce_mean(var))
        with tf.name_scope("steddv"):
            tf.summary.scalar("stddev", tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)  # 柱状图


# 设置一些基础变量
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="x_input")
    y = tf.placeholder(tf.float32, [None, 10], name="y_input")
    learning_w = tf.Variable(0.001, name="learning_w")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

# 第一层 784x400
with tf.name_scope("layer1"):
    w1 = tf.Variable(tf.random_normal(shape=[784, 400], mean=0, stddev=0.1), name="w1")
    variable_summary(w1)
    b1 = tf.Variable(tf.random_normal(shape=[400], mean=0, stddev=0.1), name="b1")
    variable_summary(b1)
    l1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1, name="sigmoid_l1")
    l1_drop = tf.nn.dropout(l1, keep_prob, name="dropout_l1")

# 第二层 400x300
with tf.name_scope("layer2"):
    w2 = tf.Variable(tf.random_normal(shape=[400, 300], mean=0, stddev=0.1), name="w2")
    variable_summary(w2)
    b2 = tf.Variable(tf.random_normal(shape=[300], mean=0, stddev=0.1), name="b2")
    variable_summary(b2)
    l2 = tf.nn.sigmoid(tf.matmul(l1_drop, w2) + b2, name="sigmoid_l2")
    l2_drop = tf.nn.dropout(l2, keep_prob, name="dropout_l2")

# 第三层 300x30
with tf.name_scope("layer_3"):
    w3 = tf.Variable(tf.random_normal(shape=[300, 30], mean=0, stddev=0.1), name="w3")
    variable_summary(w3)
    b3 = tf.Variable(tf.random_normal(shape=[30], mean=0, stddev=0.1), name="b3")
    variable_summary(b3)
    l3 = tf.nn.sigmoid(tf.matmul(l2_drop, w3) + b3, name="sigmoid_l3")
    l3_drop = tf.nn.dropout(l3, keep_prob, name="dropout_l3")

# 第四层 30x10
with tf.name_scope("layer4"):
    w4 = tf.Variable(tf.random_normal(shape=[30, 10], mean=0, stddev=0.1), name="w4")
    variable_summary(w4)
    b4 = tf.Variable(tf.random_normal(shape=[10], mean=0, stddev=0.1), name="b4")
    variable_summary(b4)
    predict = tf.nn.softmax(tf.matmul(l3_drop, w4) + b4, name="softmax_l4")

# 定义代价函数
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict))
    tf.summary.scalar("loss", loss)

# 定义训练步骤
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_w).minimize(loss)

# 初始化全局变量
init = tf.global_variables_initializer()

sever = tf.train.Saver()

# 计算准确率
with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(y, 1),
                tf.argmax(predict, 1)
            ),
            tf.float32
        )
    )
    tf.summary.scalar("accuracy", accuracy)

# 合并所有统计指标
merged = tf.summary.merge_all()

with tf.Session() as sess:
    the_y_train_hot = sess.run(y_train_hot)
    the_y_test_hot = sess.run(y_test_hot)
    sess.run(init)
    write = tf.summary.FileWriter('logs_tensorBoradLearning/', graph=sess.graph)
    for epoch in range(21):
        for batch in range(1, n_batch):
            batch_xs, batch_ys = x_train[(batch - 1) * batch_size:batch * batch_size], the_y_train_hot[(batch - 1) * batch_size:batch * batch_size]
            # sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            print(sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5}))
            break
        break
    #         summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
    #     write.add_summary(summary, epoch)
    #     acc_testData = sess.run(accuracy, feed_dict={x: x_test, y: the_y_test_hot, keep_prob: 1.0})
    #     acc_trainData = sess.run(accuracy, feed_dict={x: x_train, y: the_y_train_hot, keep_prob: 1.0})
    #     the_loss = sess.run(loss, feed_dict={x: x_train, y: the_y_train_hot, keep_prob: 1.0})
    #     the_learning_w = the_loss * 1e-4
    #     print("epoch: %s, acc_testData: %s, acc_trainData: %s,the_loss: %s" % (
    #     epoch, acc_testData, acc_trainData, the_loss))
    # sever.save(sess, save_path="models/tensorBoradLearningModel.ckpt", global_step=21)
