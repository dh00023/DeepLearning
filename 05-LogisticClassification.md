# 05 Logistic (regression) classification

## 이론

### Binary Classification
두개 중 한개를 정해진 것으로 고르는 것
```
ex) 이메일 : 스팸 or 햄
```

즉, 0과 1의 encoding이다.
```
ex) 이메일 : 스팸(1) or 햄(0)
```

Linear Regression으로도 0과 1으로 이루어진 것을 구할 수 있지않을까? 여기에는 2가지 문제점이 있다.
1. 기울기에 따라 Pass인데도 Fail로 나타날 수 있다.
2. 1보다 크거나 0보다 작은 값이 나올 수 있다.

그렇다면, 어떠한 값이 들어가도 0과 1사이에 오는 가설이 있을까??

### Logistic Function(Sigmod Function)
![](http://www.saedsayad.com/images/ANN_Sigmoid.png)

#### Cost function

이전의 cost fucntion을 적용하게되면 Gradient가 0인값이 여러개가 생기게된다. 즉, Local minimum과 Global minimum이 생긴다.(non-convex) 하지만 training을 시킬때 local minimum에서 멈추게되면 기능이 안좋은 상태에서 끝나게된다.
![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQj5EkWtb5cTKN7y4ft63yE0CKcYmR6m1h6bnb2PTLiKSIBeHiucQ)
이를 방지하기위해서 cost function이 convex형태여야한다.
![](http://www.holehouse.org/mlclass/06_Logistic_Regression_files/Image%20[12].png)
![](https://cognitree.com/wp/wp-content/uploads/2016/07/logistic-regression-5-300x292.png)

## 실습

tensorflow에서 `sigmoid`를 이용해서 sigmoid fucntion(logistic)을 구할 수 있다.

```python
import tensorflow as tf

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

X = tf.placeholder(tf.float32,shape=[None,2])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([2,1]),name="weight")
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
```