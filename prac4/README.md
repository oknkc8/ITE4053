# Training for Binary Classification with Tensorflow



### Introduction

These programs train a model for binary classification using logistic regression with `Tensorflow`.

We generated the data at `generate` function under the following conditions. Generating data is performed only when there are no files according to the input $n$ and $m$.
$$
x^{(i)}=\begin{bmatrix}x_1^{(i)} \\x_2^{(i)}\end{bmatrix},
\ y^{(i)}=\left\{\begin{matrix}1 \ \ \ \ \mathrm{if} \ (x_1^{(i)})^2> (x_2^{(i)})^2 \\
0 \ \ \ \ \mathrm{if} \ (x_1^{(i)})^2\leq(x_2^{(i)})^2\end{matrix}\right.
$$
We implemented these programs as Python. And we use `Tensorflow` library to compute easier and faster.



### Running Environment

* OS : Ubuntu 16.04.6 (WSL)

* CPU : Intel(R) Core(TM) i7-7660U CPU 250Hz

* GPU : using Google Colab

* RAM : 16GB

* Language : Python 3.7.3, Tensorflow 2.0

  

### How to Run

```shell
usage: prac4.py [-h] [--m M] [--n N] [--epoch EPOCH] [--batch BATCH]
                [--loss LOSS] [--optimizer OPTIMIZER] [--log_step LOG_STEP]
                [--lr LR]

optional arguments:
  -h, --help            show this help message and exit
  --m M                 number of train samples
  --n N                 number of test samples
  --epoch EPOCH         number of epochs
  --batch BATCH         batch size
  --loss LOSS           kind of loss function
  --optimizer OPTIMIZER
                        kind of optimizer
  --log_step LOG_STEP   step for printing log
  --lr LR               learning rate
```



### Result

#### Comparing Loss Functions (in Google Colab using GPU)

```shell
Experiment Condition:
	m : 1000,
	n : 100,
	epoch : 1000,
	batch size : 1000,
	optimizer : SGD
```

|                               | BinaryCrossEntropy (BCE) | MeanSquareError (MSE) |
| :---------------------------: | :----------------------: | :-------------------: |
| **Accuracy (with train set)** |          99.1%           |         99.4%         |
| **Accuracy (with train set)** |          98.0%           |         98.0%         |
|    **Best Learning Rate**     |           1.0            |          1.0          |

#### Comparing Optimizers (in Google Colab using GPU)

```shell
Experiment Condition:
	m : 1000,
	n : 100,
	epoch : 1000,
	batch size : 1000
```

|                                 |   SGD    | RMSProp  |   Adam   |
| :-----------------------------: | :------: | :------: | :------: |
|  **Accuracy (with train set)**  |  99.4%   |  99.6%   |  100.0%  |
|  **Accuracy (with train set)**  |  98.0%   |  99.0%   |  99.0%   |
|      **Train Time[$sec$]**      | 0.017207 | 0.017317 | 0.014112 |
| **Inference(test) Time[$sec$]** | 0.008518 | 0.007638 | 0.007439 |
|        **Loss Function**        |   MSE    |   MSE    |   MSE    |
|     **Best Learning Rate**      |   1.0    |   0.65   |   1.05   |

#### Comparing Library and Hardware

```shell
Experiment Condition:
	m : 1000,
	n : 100,
	epoch : 1000,
	batch size : 1000
```

|                                 | Using Numpy (CPU) | Using TF (CPU) | Using TF (GPU) |
| :-----------------------------: | :---------------: | :------------: | :------------: |
|  **Accuracy (with train set)**  |       99.7%       |     100.0%     |     100.0%     |
|  **Accuracy (with train set)**  |      100.0%       |     100.0%     |     100.0%     |
|      **Train Time[$sec$]**      |     0.000577      |    0.008453    |    0.013601    |
| **Inference(test) Time[$sec$]** |     0.000457      |    0.004693    |    0.006564    |
|        **Loss Function**        |        BCE        |      MSE       |      MSE       |
|          **Optimizer**          |         -         |      Adam      |      Adam      |
|     **Best Learning Rate**      |     13.113908     |      1.05      |      1.05      |

#### Comparing Batch Size (in Google Colab using GPU)

```shell
Experiment Condition:
	m : 1000,
	n : 100,
	epoch : 1000
```

|                                 | Mini batch = 1 | Mini batch = 32 | Mini batch = 128 | Mini batch = 1000 |
| :-----------------------------: | :------------: | :-------------: | :--------------: | :---------------: |
|  **Accuracy (with train set)**  |     98.6%      |      99.4%      |      99.5%       |      100.0%       |
|  **Accuracy (with train set)**  |     99.0%      |     100.0%      |      100.0%      |      100.0%       |
|      **Train Time[$sec$]**      |    7.039868    |    0.209159     |     0.057385     |     0.013601      |
| **Inference(test) Time[$sec$]** |    0.361966    |    0.014770     |     0.005621     |     0.006564      |
|        **Loss Function**        |      BCE       |       BCE       |       BCE        |        MSE        |
|          **Optimizer**          |      Adam      |      Adam       |       Adam       |       Adam        |
|     **Best Learning Rate**      |      0.05      |       0.5       |       1.0        |       1.05        |
