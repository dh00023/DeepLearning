# 머신러닝 / 딥러닝 Day02

## 이론(Linear Regression)

|x(hours)|y(score)|
|------|------|
|10|90|
|9|80|
|3|50|
|2|30|

supervisor learning : 위의 형태의 데이터로 학습(training)시키는 것
위의 형태의 데이터는 Regression형태이다.

Regression형태의 데이터는 가설을 세울 필요가 있다. Linear Hypothesis 이러한 데이터에 잘 맞는 선을 찾는 것이 학습을 하는 것이다.

1. `H(x) = Wx+b`의 형태로 가설을 세운다.
2. 어떤 모델(선)이 더 좋은 모델일까?
3. 점들과의 거리가 멀면 나쁜것, 거리가 가까우면 좋은 것(Cost function, Loss function) `(H(x)-y)^2` 제곱을 하면 양수, 음수를 구별안해도 되고, 거리가 먼건 더 크게 값이 나타난다.

Linear Regression의 학습은 cost(W,b)를 최소화하는 것이다.

## Tensorflow 실습

```python
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

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

...

for step in range(2001):
   cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
   if step % 20 == 0:
       print(step, cost_val, W_val, b_val)
```
