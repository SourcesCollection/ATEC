# # import pandas as pd
# # df = pd.read_csv("./data/train.csv", sep="\t", header=None, encoding="utf-8", names=["id", "s1", "s2", "label"])
# # new_df = df[["s1", "s2"]]
# # new_df.to_csv("./data/train", header=None, index=None, encoding="utf-8", sep="\n")
# import pandas as pd
# from sklearn.model_selection import train_test_split
# train_df = pd.read_csv("./data/train_string_distance.csv", header=None)
# # print(train_df.head(5))
#
# train_features = train_df[list(range(1, 5))]
# train_labels = train_df[[0]]
# print(train_labels.shape)
# print(train_features.shape)
# # print(train_features.head(5))
# x_train, x_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.1)
# # print(x_train.head(5))
# new_df = train_features.join(train_labels)
# print(new_df.head(5))
# import tensorflow as tf
# features = tf.constant(["emerson", "lake", "and", "palmer"])
# table = tf.contrib.lookup.index_table_from_file(
#     vocabulary_file="./deep_model/vocab.txt", default_value=0)
# ids = table.lookup(features)
# with tf.Session() as sess:
#     tf.tables_initializer().run()
#
#     print(ids.eval())
# import codecs
# with codecs.open("./deep_model/model/vocab.txt", "r", "utf-8") as f:
#     with codecs.open("./deep_model/model/vocab.bin", "wb", "utf-8") as fw:
#         for line in f:
#             fw.write(line)
# import tensorflow as tf
# import numpy as np
#
# seed = np.random.seed(1)
#
# x3 = tf.constant(np.random.random((3, 10, 20)), tf.float32)
# x4 = tf.constant(np.random.random((3, 10, 20)), tf.float32)
#
# x3 = tf.expand_dims(x3, -1)
# x4 = tf.expand_dims(x4, -1)
# x3_label = np.array([0, 0, 1])
#
#
# def cnn(seq, seq_len=10, scope=None):
#     with tf.name_scope(scope):
#         # Create a convolution + maxpool layer for each filter size
#         pool_outputs = []
#         for i, filter_size in enumerate([3, 4, 5]):
#             with tf.name_scope("conv-maxpool-%s" % filter_size):
#                 # Convolution Layer
#                 filter_shape = [filter_size, 20, 1, 10]
#                 W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
#                 b = tf.Variable(tf.constant(0.1, shape=[10]), name="b")
#                 conv = tf.nn.conv2d(
#                     seq,
#                     W,
#                     strides=[1, 1, 1, 1],
#                     padding="VALID",
#                     name="conv")
#                 # Apply nonlinearity
#                 h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
#                 # Maxpooling over the outputs
#                 k = [1, h.shape[1].value, 1, 1]
#                 print(h, k)
#                 pooled = tf.nn.max_pool(
#                     h,
#                     ksize=k,
#                     strides=[1, 1, 1, 1],
#                     padding="VALID",
#                     name="pool")
#                 # print(pooled.get_shape())
#                 pool_outputs.append(pooled)
#         # Combine all the pooled features
#         num_filters_total = 10 * 3
#         h_pool = tf.concat(pool_outputs, 3)
#         h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
#
#         # Add dropout
#         with tf.name_scope("dropout"):
#             h_drop = tf.nn.dropout(h_pool_flat, 0.5)
#
#         return h_drop
#
#
# x3 = cnn(x3)
# x4 = cnn(x4)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     # x3_norm = tf.sqrt(tf.reduce_sum(tf.square(x3), axis=1))
#     # x4_norm = tf.sqrt(tf.reduce_sum(tf.square(x4), axis=1))
#     x3_l2 = tf.nn.l2_normalize(x3, axis=1)
#     x4_l2 = tf.nn.l2_normalize(x4, axis=1)
#
#     # x3_x4 = tf.reduce_sum(tf.multiply(x3, x4), axis=1)
#     # xx = tf.reduce_sum(tf.multiply(x3_l2, x4_l2), axis=1)
#     # cosin = x3_x4 / (x3_norm * x4_norm)
#     # cosin1 = tf.divide(x3_x4, tf.multiply(x3_norm, x4_norm))
#     dis = tf.losses.cosine_distance(x3_l2, x4_l2, axis=1, reduction=tf.losses.Reduction.NONE)
#     sim = 1 - dis
#     logits = tf.concat([dis, sim], axis=1)
#     pred = tf.argmax(logits, axis=1)
#     label_onehot = tf.one_hot(x3_label, depth=2)
#     loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=x3_label))
#     # loss1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_onehot))
#     # a, b, c, d = sess.run([xx, cosin1, loss, loss1])
#     print(sess.run([logits, pred]))
#     # print(x3.shape)
#     # print("a", a)
#     # print("b", b)
#     # print("c", c)
#     # print("d", d)
#     # print(c.shape)
#     # print(a.shape)
#     # print(x3.eval())
#
# import numpy as np
# labels = np.array([[1,1,1,0],
#                    [1,1,1,0],
#                    [1,1,1,0],
#                    [1,1,1,0]], dtype=np.uint8)
#
# predictions = np.array([[1,0,0,0],
#                         [1,1,0,0],
#                         [1,1,1,0],
#                         [0,1,1,1]], dtype=np.uint8)
#
# n_batches = len(labels)
#
# import tensorflow as tf
#
# graph = tf.Graph()
# with graph.as_default():
#     # Placeholders to take in batches onf data
#     tf_label = tf.placeholder(dtype=tf.int32, shape=[None])
#     tf_prediction = tf.placeholder(dtype=tf.int32, shape=[None])
#
#     # Define the metric and update operations
#     tf_metric, tf_metric_update = tf.metrics.accuracy(tf_label,
#                                                       tf_prediction,
#                                                       name="my_metric")
#
#     # Isolate the variables stored behind the scenes by the metric operation
#     running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
#
#     # Define initializer to initialize/reset running variables
#     running_vars_initializer = tf.variables_initializer(var_list=running_vars)
#
#
# with tf.Session(graph=graph) as session:
#     session.run(tf.global_variables_initializer())
#
#     # initialize/reset the running variables
#     session.run(running_vars_initializer)
#
#     for i in range(n_batches):
#         # Update the running variables on new batch of samples
#         feed_dict={tf_label: labels[i], tf_prediction: predictions[i]}
#         print(session.run(tf_metric_update, feed_dict=feed_dict))
#
#     # Calculate the score
#     score = session.run(tf_metric)
#     print("[TF] SCORE: ", score)
#
# with tf.Session(graph=graph) as session:
#     session.run(tf.global_variables_initializer())
#
#     for i in range(n_batches):
#         # Reset the running variables
#         session.run(running_vars_initializer)
#
#         # Update the running variables on new batch of samples
#         feed_dict={tf_label: labels[i], tf_prediction: predictions[i]}
#         session.run(tf_metric_update, feed_dict=feed_dict)
#
#         # Calculate the score on this batch
#         score = session.run(tf_metric)
#         print("[TF] batch {} score: {}".format(i, score))
# import numpy as np
# import torch
# a = np.random.random((2,3,3))
# b = np.random.random((2,3,4))
# c = torch.cat([a,b],2)
# print(c.shape)