# **신경망 최적화 방법**

* ### Learning rate 조정

* ### Normalization
  * #### Standardization
  ```
  X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
  ```
  * #### MinMaxScaler
  ```
  xy = MinMaxScaler(xy)
  ```
  [MinMaxScaler Example Code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-07-3-linear_regression_min_max.py "GitHub")

* ### Activate Function 선택
  * #### Sigmoid
  출력층의 Activate Function에만 사용됨

  0 ~ 1 사이 값을 출력하기 때문에 확률로 해석 가능
  ``` 
  hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
  cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
  ```
  [Sigmoid Example Code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-05-1-logistic_regression.py "GitHub")
  * #### SoftMax
  multi classification에서 주로 사용
  0 ~ 1 사이 값을 출력하기 때문에 확률로 해석 가능
  ```
  # 1)과 2) 역할은 동일
  # 1)
  hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
  cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
  # 2)
  logits = tf.matmul(X, W) + b
  hypothesis = tf.nn.softmax(logits)
  cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
  cost.tf.reduce_mean(cost_i)
  ```
  [SoftMax 1) Example Code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-06-1-softmax_classifier.py "GitHub")

  [SoftMax 2) Example Code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-06-2-softmax_zoo_classifier.py "GitHub")
  * #### ReLU
  max(0, x)
  x가 음수일 땐 0을 리턴, 양수일 땐 값 유지
  ```
  hypothesis = tf.nn.relu(tf.matmul(X, W) + b)
  ```
  [ReLU Example Code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-2-mnist_nn.py "GitHub")
  * #### tanh
  sigmoid의 중심점을 (0, 0.5)에서 (0, 0)으로 변경시킨 것
  * #### LeakyReLU
  max(0.1x, x)
  ReLU에서 x가 음수일 때의 값을 0이 아닌 0.1x로 변경
  * #### ELU
  ReLU에서 x가 음수일 때의 값을 0으로 고정시키지 않고 조절
  * #### Maxout

* ### Weight Initializer 선택
  * #### RBM
  * #### Xavier
  ```
  # fan_in: 이전 레이어 출력 개수, fan_out: 다음 레이어 입력 개수
  W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
  W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in/2)
  ```
  [Xavier Example Code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-3-mnist_nn_xavier.py "GitHub")
  * #### LSUV
  * #### OrthoNorm
  * #### OrthoNorm-MSRA scaled
  * #### MSRA

* ### Overfitting 방지
  * #### Drop Out
  ```
  dropout_rate = tf.placeholder("float")
  _L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1)
  L1 = tf.nn.dropout(_L1, dropout_rate)
  # ...
  # 학습 시에는 dropout 적용 O (dropout_rate는 보통 0.5 ~ 0.7 사이 값 입력)
  sess.run(optimizer, feed_dict={X: train_images, Y: train_labels, dropout_rate: 0.7})
  # ...
  # 실제 정확도 측정 시에는 dropout 적용 X (dropout_rate = 1)
  print("Accuracy: ", accuracy.eval({X: test_images, Y: test_labels, dropout_rate: 1})
  ```
  [DropOut Example Code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-5-mnist_nn_dropout.py "GitHub")
  * #### Regularization
  ```
  #lamdba 값으로 주로 0.001 사용
  l2reg = lambda * tf.reduce_sum(tf.square(W))
  ```
  * #### Ensemble

* ### Bath Normalization
  [Batch Normalization Example Code](https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-10-6-mnist_nn_batchnorm.ipynb "GitHub")
