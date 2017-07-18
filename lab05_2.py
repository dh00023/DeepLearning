import tensorflow as tf
filename_queue = tf.train.string_input_producer(
    ['data03.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

# cast는 true면 1 false면 0으로 값을 반환
predict = tf.cast(hypothesis>0.5,dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y),dtype=tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	for step in range(10001):
		x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
		cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
		if step % 200 ==0:
			print(step,cost_val)
	coord.request_stop()
	coord.join(threads)

		
			

	h,c,a = sess.run([hypothesis,predict,accuracy],feed_dict={X: x_data,Y: y_data})
	print("\nhypothesis:",h,"\nCorrect(Y):",c,"\nAccuracy:",a)