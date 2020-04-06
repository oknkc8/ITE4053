# Training for Binary Classification using Logistic Regression

#### 2017029807 성창호

### Index

1. Introduction

2. Running Environment

3. Model Implementation

4. Experiments

   * Time Comparison

   * Estimated unknown function parameters $W$, $b$, and hyper parameter $\alpha$

   * Accuracy



## 1. Introduction

These programs train a model for binary classification using logistic regression.  There are two versions for training, element-wise and vectorized. The two versions of running time and accuracy will be compared.

In these programs, we define a model $y = \sigma (w_1 x_1 + w_2 x_2 + b)$ that has three function parameters $w_1,w_2, b$. ($\sigma$ : sigmoid function) The goal of this model is to try to learn the parameters so that the output of model becomes a good estimate of the probability of $y=1$ given $x_1, x_2$.

To train a model, we have to measure how well parameters $w_1,w_2$ and $b$ are doing on a single training example. So we use cross-entropy loss $\mathcal{L}(\hat{y}, y) = -(y\log{\hat{y} + (1-y)\log{(1-\hat{y})}})$ as loss(error) function. ($\hat{y}$ : output of model)

We implemented these programs as Python. The vectorized version use numpy library to make vector operations easier and faster.



## 2. Running Environment

* OS : Ubuntu 16.04.6 (WSL)
* CPU : Intel(R) Core(TM) i7-7660U CPU 250Hz
* RAM : 16GB
* Language : Python 3.7.3



## 3. Model Implementation

#### Usage

```shell
usage: prac1.py [-h] --version VERSION [--size_m SIZE_M] [--size_n SIZE_N]
                [--size_k SIZE_K] [--log_step LOG_STEP] [--alpha ALPHA]
                [-initial_zero]

optional arguments:
  -h, --help           show this help message and exit
  --version VERSION    version of logistic regression (element_wise,
                       vectorized, compare, find_alpha)
  --size_m SIZE_M      number of train samples
  --size_n SIZE_N      number of test samples
  --size_k SIZE_K      number of iterations
  --log_step LOG_STEP  step for printing log
  --alpha ALPHA        learning rate
  -initial_zero        set initial parameter as zero
```







#### Generate train & test sample

Before training and testing model, we should make training samples and test samples for the binary classification. We generated $m$ training samples as below.
$$
x^{(i)}=\begin{bmatrix}x_1^{(i)} \\x_2^{(i)}\end{bmatrix},
\ y^{(i)}=\left\{\begin{matrix}0 \ \ \ \ \mathrm{if} \ (x_1^{(i)}+x_2^{(i)})>0 \\
1 \ \ \ \ \mathrm{if} \ (x_1^{(i)}+x_2^{(i)})\leq0\end{matrix}\right.
$$
And we generated $n$ test samples in the same way. 

```python
# Training Samples
x1_train = []
x2_train = []
y_train = []
for i in range(m):
    x1_train.append(random.randint(-10,10))
    x2_train.append(random.randint(-10,10))

    if x1_train[-1] + x2_train[-1] > 0:
        y_train.append(1)
    else:
        y_train.append(0)

# Test Samples : same way
```



#### Element-wise version

Before the gradient descent, we should reset function parameters $w_1, w_2, b$. If `-initial_zero` option is on, we set parameters as *zero*.

We define logistic regression model as $y = \sigma (w_1 x_1 + w_2 x_2 + b)$ and use cross-entropy loss, so we implemented model $y$, sigmoid function $\sigma$, cross-entropy loss as below.

```python
def cross_entropy_loss(y_hat, y):
    a1 = -(y * math.log(y_hat))
    a2 = (1 - y) * math.log(1 - y_hat)
    return a1 + a2

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def model(x1, x2, w1, w2, b):
    return sigmoid(w1*x1 + w2*x2 + b)
```

Then, update function parameters using gradient descent for input $k$ times. In this version, we should compute cost and gradients as element-wise for $m$ training samples and compute accuracy for $m$ training samples and $n$ test samples as below.

```python
for iteration in range(iterations):
    cost = 0
    dw1 = dw2 = db = 0
    acc = 0
    print("######### Training #########")
    for i in range(m):
        y_hat = model(x1_train[i], x2_train[i], w1, w2, b)
        cost += -cross_entropy_loss(y_hat, y_train[i])
        
        if (y_hat > 0.5 and y_train[i] == 1) or (y_hat <= 0.5 and y_train[i] == 0):
            acc = acc +1

        dz = y_hat - y_train[i]
        dw1 += x1_train[i] * dz
        dw2 += x2_train[i] * dz
        db += dz
    
    cost /= m
    dw1 /= m
    dw2 /= m
    db /= m

    acc = 0
    cost = 0
	print("######## Evaluation ########")
    for i in range(n):
        y_hat = model(x1_test[i], x2_test[i], w1, w2, b)
        cost += -cross_entropy_loss(y_hat, y_test[i])
        
        if (y_hat > 0.5 and y_test[i] == 1) or (y_hat <= 0.5 and y_test[i] == 0):
            acc = acc +1

    cost /= n
	
    # Parameters Update
    w1 = w1 - alpha * dw1
    w2 = w2 - alpha * dw2
    b = b - alpha * db
```



#### Vectorized version

Similar to the element-wise version, but the vectorized version uses numpy library to compute quickly. To do this, we should convert lists to numpy array and set function parameters as vector. Parameter $b$ is same as before, but $w_1, w_2$ should be converted to vector $W=\begin{bmatrix}w_1 \\w_2\end{bmatrix}$. 

```python
# Convert to Numpy array
x_train = np.array((x1_train, x2_train))
y_train = np.array(y_train)
x_test = np.array((x1_test, x2_test))
y_test = np.array(y_test)

for iteration in range(iterations):
    print("######### Training #########")
    y_hat = model(x_train, W, b)
    cost = np.sum((-cross_entropy_loss(y_hat, y_train))) / m
    
    dz = y_hat - y_train
    dW = np.dot(x_train, dz) / m
    db = np.sum(dz) / m

    y_hat[y_hat > 0.5] = 1
    y_hat[y_hat <= 0.5] = 0
    acc = np.sum(y_hat == y_train)

	print("######## Evaluation ########")
    y_hat = model(x_test, W, b)
    cost = np.sum((-cross_entropy_loss(y_hat, y_test))) / m
    
    y_hat[y_hat > 0.5] = 1
    y_hat[y_hat <= 0.5] = 0
    acc = np.sum(y_hat == y_test)

    # Parameters Update
    W = W - alpha * dW
    b = b - alpha * db
```

In addition the model, sigmoid function and cross-entropy loss should be converted as below.

```python
def cross_entropy_loss(y_hat, y):
    a1 = -(y * np.log(y_hat))
    a2 = (1 - y) * np.log(1 - y_hat)
    return a1 + a2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model(x, W, b):
    return sigmoid(np.dot(W, x) + b)
```



## 4. Experiments

#### Time Comparison

We measured the running time for $k$ iterations when we train the model using `compare` option. The running time is measured from before the loop to the end of the loop. The result is as follow:

```shell
######## RESULT ########
num of train sample (m) : 1000
num of test sample (n) : 100
num of iterations (k) : 1000
Alpha : 0.001000
initial_zero : True

------ Element-wise ------
Running Time : 1.791510
Training Accuracy : 98.856000
Test Accuracy : 98.462000

------ Vectorized ------
Running Time : 0.193028
Training Accuracy : 98.856000
Test Accuracy : 98.462000
```

As expected, the vectorized version is much faster than the Element-wise version. This is thought to be due to the use of the numpy library.



#### Estimated unknown function parameters $W$, $b$, and hyper parameter $\alpha$

After many experiments, we could see that accuracy increases as it increases. So it is possible to determine that the function parameters after $k$ iterations are the most optimal. The results of the experiment with the same samples as above are as follows.

```shell
1000 iteration Parameters w1: 0.409825, w2: 0.408325, b: -0.015614
######### Training #########
Cost: 0.022471
Training Accuracy: 100.000000%
######## Evaluation ########
Cost: -0.000329
Evaluation Accuracy: 100.000000%
```

$\alpha$, which has a significant effect on accuracy, is a hyper parameter. So we found the best $\alpha$ through binary search using `find_alpha` option. Samples from this experiment are different from those above.

```
Best value of alpha : 0.562500
Num of search : 54
```



#### Accuracy

We measured the accuracy in vectorized version and set `initial_zero` option `True`. And $\alpha$ was fixed at 0.001

|                               | $(m,k) = (10, 100)$ | $(m,k) = (100, 100)$ | $(m,k) = (1000, 100)$ | $(m,k) = (100, 10)$ | $(m,k) = (100,100)$ | $(m,k) = (100, 1000)$ |
| :---------------------------: | :-----------------: | :------------------: | :-------------------: | :-----------------: | :-----------------: | :-------------------: |
| Accuracy (with train samples) |       86.10%        |        95.13%        |        88.923%        |       84.00%        |       94.93%        |        95.08%         |
| Accuracy (with test samples)  |       76.35%        |        95.00%        |        92.23%         |       76.00%        |       84.34%        |        93.44%         |

When $k$ is fixed and $m$ is changed, the smaller $m$, the greater the difference in accuracy between training samples and test samples. Maybe it's because the model is fitted to the training samples. Conversely, if $m$ is fixed and $k$ is changed, the accuracy of test samples is increased as $k$ increases. This can conclude that the model is influenced by the number of times($k$) it learns.