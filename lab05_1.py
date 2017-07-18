import tensorflow as tf
import numpy as np

xy = np.loadtxt('data03.csv',delimiter=',',dtype=np.float32)
x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

X = tf.placeholder(tf.float32,shape=[None,8])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([8,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")

# Logistic 함수
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

# cast는 true면 1 false면 0으로 값을 반환
predict = tf.cast(hypothesis>0.5,dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y),dtype=tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(10001):
		cost_val,_ = sess.run([cost,train],feed_dict={X: x_data,Y:y_data})
		if step % 200 ==0:
			print(step,cost_val)

	h,c,a = sess.run([hypothesis,predict,accuracy],feed_dict={X: x_data,Y: y_data})
	print("\nhypothesis:",h,"\nCorrect(Y):",c,"\nAccuracy:",a)