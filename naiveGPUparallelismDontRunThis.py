
# I have no GPU access so I did not test this code
with tf.device("/gpu:0"):
    a = tf.Variable(tf.ones(()))
    a = tf.square(a)
with tf.device("/gpu:1"):
    b = tf.Variable(tf.ones(()))
    
with tf.device("/gpu:2"):
    c = tf.Variable(tf.ones(()))
with tf.device("/gpu:3"):
    d = c+a
with tf.device("/cpu:0"):
    loss = (a*b)/c+d
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(10):
    loss0, _ = sess.run([loss, train_op])
    print("loss", loss0)