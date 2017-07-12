# 그래프구현
x_train = [1,2,3]
y_train = [1,2,3]

## 랜덤 값 주기
W = tf.Variable(tf.random_normal([1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')
## Variable 텐서플로우가 자체적으로 변경시키는 값이다.(즉, 학습하는 과정에서 영향을 미치는 값이다.)

hypothesis = x_train*W +b

## 값이 들어 왔을때 평균을 내주는 것 (reduce_mean)
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

## Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train =  optimizer.minimize(cost)

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
	sess.run(train)
   if step % 20 == 0:
       print(step, sess.run(cost), sess.run(W), sess.run(b))

# Placeholders(값을 위에처럼 지정하지 않고, feed_dict로 넘겨줄 수 있다.)

X = tf.placeholder(tf.float32,shape[None])
Y = tf.placeholder(tf.float32,shape[None])

...

for step in range(2001):
   cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
   if step % 20 == 0:
       print(step, cost_val, W_val, b_val)

# 제대로 작동하는지 확인
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))