# 06. Multinomial Classification - Softmax Classification

## 이론

### Logistic regression
![](http://cfile26.uf.tistory.com/image/236A814058BE07222FC2F5)
위의 그래프에서 A와 B를 구분하는 선을 학습하는 것이다.

### Multinomial Classification

여러개의 값을 예측할 때!

![](http://cfile6.uf.tistory.com/image/215DE63B58BE002609B1F5)
여기서 3개의 선으로 구별을 할 수 있다.
1. A or not
2. B or not
3. C or not

![](http://cfile26.uf.tistory.com/image/232FE93358BE093C2DB375)

![](http://cfile28.uf.tistory.com/image/22361D3558BE3360331402)

Vector로 결과 값이 나오게된다.

이때 결과값의 합이 1이 나오면 좋겠다. ==> **Softmax**

#### Softmax
![](http://andersonjo.github.io/assets/posts/TensorFlow-Softmax-Regression/udacity_logistic02.png)
1. 0~1사의 값
2. 전체의 합이 1이된다.

그러면 각각을 확률로 볼 수 있다.

**One-Hot Encoding**은 가장 큰 값을 1, 나며지는 0


#### Cost function
![](http://cfile29.uf.tistory.com/image/2153C64F5797F9C32E9735)

S(y)는 softmax가 예측한 값이고, L(y)는 실제 Y의 값이다.

![](http://cfile23.uf.tistory.com/image/243B804E579806C11E2A8F)
이때, 실제의 값과 예측이 같을 경우에는 cost가 0이나오지만 틀릴경우에는 무한대로 큰 값이 나오는 것을 볼 수 있다.

## 실습
