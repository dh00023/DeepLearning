# 03. How to minimize cost

## 이론

`Gradient descent algorithm`을 이용해 Cost function의 최소값을 찾는다. 이때 어느 점에서 시작하든지 최소 값을 찾을 수 있다.

Cost function을 W로 미분을 해주면된다. 미분은 한 점에서의 기울기를 의미한다.
cost function의 모양이 convex function이 되는지 확인해야한다.

## 실습

Cost function이 어떻게 생겼는지 확인하기 위해서 `tesorflow`와 `matplotlib.pyplot` 라이브러리를 사용할 것이다.

#### matplotlib 설치
```
$ curl -O https://bootstrap.pypa.io/get-pip.py
```
```
$ python3 get-pip.py
$ pip install matplotlib
```

### Cost function
```python
import tensorflow as tf
import matplotlib.pyplot as plt

X = [1,2,3]
Y = [1,2,3]

W = tf.placeholder(tf.float32)

hypothesis = X*W

cost = tf.reduce_mean(tf.square(hypothesis-Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
W_val=[]
cost_val=[]
for i in range(-30,50):
	feed_W = i*0.1
	curr_cost, curr_W = sess.run([cost,W],feed_dict={W: feed_W})
	W_val.append(curr_W)
	cost_val.append(curr_cost)

plt.plot(W_val,cost_val)
plt.show
plt.savefig("temp.png") # 이미지로 저장
```

### Gradient descent
```
import tensorflow as tf
x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_normal([1]),name = "weight")
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X*W

cost = tf.reduce_sum(tf.square(hypothesis-Y))

#Gradient descent
learning_rate = 0.1
gradient = tf.reduce_mean((W*X-Y)*X)
descent = W -learning_rate*gradient
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(21):
	sess.run(update,feed_dict={X: x_data, Y: y_data})
	print(step,sess.run(cost,feed_dict={X: x_data, Y: y_data}),sess.run(W))
```

```
0 5.8498 [ 0.35359246]
1 1.66394 [ 0.6552493]
2 0.473299 [ 0.81613296]
3 0.134627 [ 0.9019376]
4 0.038294 [ 0.94770002]
5 0.0108925 [ 0.9721067]
6 0.00309832 [ 0.98512357]
7 0.000881298 [ 0.99206591]
8 0.000250681 [ 0.99576849]
9 7.13039e-05 [ 0.99774319]
10 2.02839e-05 [ 0.99879634]
11 5.76971e-06 [ 0.99935806]
12 1.64103e-06 [ 0.99965763]
13 4.66575e-07 [ 0.99981743]
14 1.32868e-07 [ 0.99990261]
15 3.77147e-08 [ 0.99994808]
16 1.07645e-08 [ 0.99997228]
17 3.05909e-09 [ 0.99998522]
18 8.66635e-10 [ 0.99999213]
19 2.40746e-10 [ 0.99999583]
20 7.02158e-11 [ 0.99999774]
```
결과 값을 보면 cost값은 점점 작아지고 W의 값은 1과 가까워지는 것을 확인 할 수 있다.

```python
#Gradient descent
learning_rate = 0.1
gradient = tf.reduce_mean((W*X-Y)*X)
descent = W -learning_rate*gradient
update = W.assign(descent)
```
이 미분하는 부분은 tensorflow의 `.GradientDescentOptimizer`를 사용하면 구할 수 있다.
```python
# Minimize: Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)
```

W의 값이 어디서 시작하는지 상관없이 결국 1.0에 수렴하는 것을 볼 수 있다.

