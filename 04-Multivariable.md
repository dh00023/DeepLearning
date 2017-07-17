# 04 Multivariable linear regression

## 이론
input(x1,x2,x3....)의 값이 여러개인 경우를 알아보자.

- **Hypothesis** : H(x)=**w1x1+w2x2+w3x3**+b
- Cost function은 전과 같다.

#### Matrix
여기서 핵심은 행렬로 표현 하는 것이다. instance와 variable이 많이지면 계산과 표현에 힘듦이 있으므로 matrix로 표현해준다.
이때 Matrix는 보통 대문자로 표현한다. 또한 Matrix로 표현할 때는 X를 먼저 표시해준다.

## 실습

```python
import tensorflow as tf

x_data = [[73., 80., 75.], [93., 88., 93.],
         [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

X = tf.placeholder(tf.float32,shape=[None,3])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")

hypothesis = tf.matmul(X,W)+b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
	cost_val,hy_val,_ = sess.run([cost,hypothesis,train],feed_dict={X: x_data,Y: y_data})
	if step % 10 == 0:
		print(step, "Cost: ",cost_val,"\nPrediction:\n",hy_val)
```

matrix로 더 간단하게 표현할 수 있다.

#### 파일에서 데이터 읽어오기

파이썬의 가장 큰 장점은 Slicing에 있다.
`numpy`로 가면서 강력한 Slicing이 가능하다.
```python
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('rootkey.csv',delimiter=',',dtype=np.float32)
x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

print(x_data.shape,x_data,len(x_data))
print(y_data.shape,y_data)

X=tf.placeholder(tf.float32,shape=[None,3])
Y=tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")

hypothesis=tf.matmul(X,W)+b

cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer= tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
	cost_val,hy_val,_ = sess.run([cost,hypothesis,train],feed_dict={X: x_data,Y: y_data})
	if step % 10 == 0:
		print(step, "Cost: ",cost_val,"\nPrediction:\n",hy_val)

print("Your score will be ", sess.run(hypothesis,feed_dict={X: [[100,70,101]]}))
print("Other score will be ", sess.run(hypothesis,feed_dict={X: [[60,70,110],[90,100,80]]}))
```

#### Queue runner

1. 여러개의 파일을 큐에 쌓는다.
2. 큐를 Reader에 연결한다.
3. 데이터를 읽어들여 데이터 양식에 맞게 Decoding을 한다.
4. 읽어온 값을 다시 큐에 쌓는다.

굉장히 큰 데이터를 쓸때 유용하다.

```python
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)

# Ask my score
print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
```


