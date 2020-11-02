# encoding: utf-8
import tensorflow as tf
import tarfile
import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class NodeLookup(object):
    def __init__(self):
        label_lookup_path = 'inception_model_v3_dir/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path = 'inception_model_v3_dir/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    # @staticmethod
    def load(self, label_lookup_path, uid_lookup_path):
        # 加载分类字符串n********对应分类名称的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        # 一行一行读取数据
        for line in proto_as_ascii_lines:
            # 去掉换行符
            line = line.strip('\n')
            # 按照'\t'分割
            parsed_items = line.split('\t')
            # 获取分类编号
            uid = parsed_items[0]
            # 获取分类名称
            human_string = parsed_items[1]
            # 保存编号字符串n********与分类名称映射关系
            uid_to_human[uid] = human_string

        # 加载分类字符串n********对应分类编号1-1000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                # 获取分类编号1-1000
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                # 获取编号字符串n********
                target_class_string = line.split(': ')[1]
                # 保存分类编号1-1000与编号字符串n********映射关系
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # 建立分类编号1-1000对应分类名称的映射关系
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            # 获取分类名称
            name = uid_to_human[val]
            # 建立分类编号1-1000到分类名称的映射关系
            node_id_to_name[key] = name
        return node_id_to_name

    # 传入分类编号1-1000返回分类名称
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


# 处理环境
tf = tf.compat.v1
tf.disable_v2_behavior()

# inception 下载地址
inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# 定义模型存放地址
inception_pretrain_model_dir = "inception_model_v3_dir"
if not os.path.exists(inception_pretrain_model_dir):
    os.mkdir(inception_pretrain_model_dir)

# 文件名和文件路径
filename = "inception-2015-12-05.tgz"
filepath = os.path.join(inception_pretrain_model_dir, filename)

# 检查文件是否存在,不存在就下载
if not os.path.exists(filepath):
    print("download file: %s from %s" % (filename, inception_pretrain_model_url))
    the_file = requests.get(inception_pretrain_model_url, stream=True)
    with open(filepath, "wb") as file:
        for chunk in the_file.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
# for i in range(10):
#      print("\r进度：" + "▉"*i + "%s%%" % i, end="")
#      time.sleep(1)
print("finsh download")

# 解压文件
tarfile.open(filepath, "r:gz").extractall(inception_pretrain_model_dir)

# 模型结构文件
log_dir = "inception_log"
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# 模型文件
inception_graph_def_file = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')

with tf.Session() as sess:
    # 创建一个图来存放训练好的模型
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    # 保存图的结构
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.close()
    # 从 tensorboard 可以知道预测点的图的名字叫 softmax:0
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    # 遍历目标识别的文件夹
    for root, dirs, files in os.walk("images/"):
        for file in files:
            # 载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            # DecodeJpeg/contents:0 这个节点也可在 tensorboard 中看到
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # 图片格式是jpg格式
            predictions = np.squeeze(predictions)  # 把结果转为1维数据

            # 打印图片路径及名称
            image_path = os.path.join(root, file)
            print(image_path)
            # 显示图片
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            # 排序
            top_k = predictions.argsort()[-5:][::-1]
            node_lookup = NodeLookup()
            for node_id in top_k:
                # 获取分类名称
                human_string = node_lookup.id_to_string(node_id)
                # 获取该分类的置信度
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print()