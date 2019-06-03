import tensorflow as tf

x = tf.constant(10, name = 'x')
y = tf.Variable(x + 5, name = 'y')
# tensorflow에서는 항상 변수 초기화를 해줘야함

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(y))