import tensorflow as tf

x = tf.placeholder("int32")
y = tf.placeholder("int32")

z = tf.multiply(x, y)
# 행렬 곱셈은 matmul
# 같은 위치 곱셈은 element_wise
# 같은 행의 곱셈은 multiply

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    result = sess.run(z, feed_dict = {x : 2, y : 5})
    print(result)
