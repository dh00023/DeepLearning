# 머신러닝 / 딥러닝 기본 Day01

## 이론
머신러닝은 일종의 소프트웨어 - 프로그램 자체가 학습을해서 배우는 프로그램

학습을 하기위해서 데이터가 미리 주어져야한다.
- Supervised learning : learning with labeled data(Training data set)
	- Regression
    - Binary Classification
    - multi-label classification
- Unsupervised learning : unlabeled data



## Tensorflow 실습

```python
$ python3
>>> import tensorflow as tf
>>> tf.__version__
```

python3 실행 후 텐서플로우를 import 해준다. `tf.__version__`은 tensorflow의 버전을 확인할 수 있다.

```python
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)

>>>print("node1:",node1,"node2",node2)
node1: Tensor("Const:0", shape=(), dtype=float32) node2 Tensor("Const_1:0", shape=(), dtype=float32)
>>> print("node3:",node3)
node3: Tensor("Add:0", shape=(), dtype=float32)

sess = tf.Session()
>>> print("sess.run(node1,node2)",sess.run([node1,node2]))
sess.run(node1,node2) [3.0, 4.0]
>>> print("sess.run(node3)",sess.run(node3))
sess.run(node3) 7.0```
session을 만들어야지 원하는 값을 출력할 수 있다.
```

#### Placeholder

그래프를 정의할 수 있따. 하나의 노드를 placehold로 만들 수 있다.

> 플레이스홀더는 레이지 기법으로 값을 미리 정하지 않고, 프로그래밍이 돌아가는 시점에 정해지는 값입니다. 변수의 형태(type)만 정해 놓고, 변수의 값은 정해지지 않았지만 텐서플로우가 실행되는 동안에 정해집니다. 값이 미리 정해지지 않고, 나중에 정해지는 점은 굉장히 유연한 프로그래밍을 할 수 있게 도와줍니다.

#### Tensor
- Ranks(scalar,vector,matrix, 3-Tensor, ....)
- Shapes
- Types


1. 그래프를 먼저 설계
2. 설계한 후 우리가 원하는 값 넘겨준다.
3. 그 결과로 값을 리턴한다.

