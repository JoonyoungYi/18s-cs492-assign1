# KAIST CS492C Programming Assignment #1 		

- Author: Joonyoung Yi(20183453, School of Computing, Master's degree student)

- Date: 2018-04-27

  ​

### 1. Environment Settings

#### 1-1. How to Install 

* Python 3.5 and TensorFlow 1.4
* `requirements/local.txt` only for development setting. 
  * `requirements/local.txt` packages is not necessary for TAs.
```
$ virtualenv .venv -p python3.5
$ . .venv/bin/activate
$ pip install -r requirements/common.txt
$ pip install -r requirements/local.txt
```

* If you don't want to install gpu version of tensorflow, please remove below line in `requirements/common.txt` or make that line be comment.
```
tensorflow-gpu==1.4
```



#### 1-2. How to Run

* You can run this code by typing below commands:

```
$ . .venv/bin/activate			# activate the virtual environment
$ python run.py
```
* If you already installed required packages I specified in `requirements/` folder, you don't have to enable virtual environment.



#### 1-3. Warning

* I've tested this code in the machine with `TITAN Xp`. Please check the memory size of the testing machine.
  * If you want to run on a computer with a small amount of memory, modify `model_idx` to a different number that is not already used and modify the `hidden_layer_size` or `hidden_layer_number` in the `config.py` file.



### 2. The number of parameters of my networks

​	The number of parameters are as below:

| The number of layers | The number of parameters |
| :------------------: | :----------------------: |
|          3           |          647004          |
|          5           |         1147008          |
|          7           |         1647012          |

​	First, I will describe why the number of parameters of my 7-layer network is []. I built a model as below:

```
	INPUT LAYER 
	HIDDEN LAYER 1 : DENSE LAYER 1 + BN LAYER 1 + ACTIVATION LAYER 1
	HIDDEN LAYER 2 : DENSE LAYER 2 + BN LAYER 2 + ACTIVATION LAYER 2
	HIDDEN LAYER 3 : DENSE LAYER 3 + BN LAYER 3 + ACTIVATION LAYER 3
	HIDDEN LAYER 4 : DENSE LAYER 4 + BN LAYER 4 + ACTIVATION LAYER 4
	HIDDEN LAYER 5 : DENSE LAYER 5 + BN LAYER 5 + ACTIVATION LAYER 5
	HIDDEN LAYER 6 : DENSE LAYER 6 + BN LAYER 6 + ACTIVATION LAYER 6
	OUTPUT LAYER 
```

​	In the "INPUT LAYER", there is no paramters. All variables are given in that layer. For the "HIDDEN LAYER 1", there are $784 \times 500 = 392000$ paramters on "DENSE LAYER 1"(I don't use bias in a dense layer. Because, I used batch normalization). On the first batch normalization layer "BN LAYER 1", there are additional $2$ parameters. Because, it has the mean and variance parameters for each layer. I used ReLU for activation layer. So, the activation layers don't have paramters. Therefore, the number of parameters in "HIDDEN LAYER 1" is $392002$. For "HIDDEN LAYER 2", $500 \times 500 = 250000$ parameters on "DENS LAYER 2", there, also, are 2 additional parameters on "BN LAYER 2". So, the number of paramters on HIDDEN LAYER 2 is $250002$. The parameter counting methods of hidden layer 3 to 6 are same as that of "HIDDEN LAYER 2". For output layer, the number of paramters is $10 \times 500 = 5000$. 

​	To sum it up, the number of paramters on 7-layer networks is calculated as below:
$$
392002 + 250002 \times 5 + 5000 = 1647012
$$
Similarly, the number of paramters on 3-layer and 5-layer networks is calculated as below:
$$
\text{For 5-layer, } 392002 + 250002 \times 3 + 5000 = 1147008 \\
\text{For 3-layer, }392002 + 250002 \times 1 + 5000 = 647004
$$


### 3. The Regularization Techniques on my Networks.

​	Before analyzing the results of my networks, I think I should introduce you to the regularization techiniques I used first. This is because regularization techniques have a significant impact on the results. Before looking at changes in the number of layers, it is necessary to first explain the effect of each network.



#### 3-1. Dropout

​	I used dropout for each dense layer with probability $0.5$. I have heard that dropping a node with a probability of $0.5$ works well to prevent overfitting. It is common to set this value between $0.1$ and $0.5$, which I have chosen to be $0.5$. Because, I've learned that 0.5 is the quite good probability to avoid overffitting when I tested the probabilities of 0.1, 0.2, 0.3, 0.4, 0.5 under certain hyperparameters. Especially, when dropout is performed with a probability of 0.1, it can be confirmed that overfitting occurs insignificantly. 



#### 3-2. Batch Normalization

​	I also did batch normalization with dropout. I was careful in ordering layers to proceed with dropout and batch normalization. Batch normalization certainly increases the time taken per epoch. However, I could feel empirically that the degree of convergence per epoch is larger, resulting in a faster convergence of the network. It was much faster to converge than using dropout without batch normalization. As a result, we could find a good hyper parameter through more trials. 

​	And as the batch normalization progresses, the network seems to be more stable. If batch normalization is not used, the end result may vary from trial to trial. I obseerved that the performance of the evaluation set varies from 55% to 65%. Using batch normalization, we can confirm that 60% of the performance is stable regardless of the trial. The more detail information is served in 4-2.



#### 3-3. Advarsarial Learning

​	We learned about Advarsarial Learning in class. Adding gaussian noise to training data is a way to make the network more stable. I read the paper the lecture slide metioned. They added Gaussian noise with a standard deviation of 0.06 ~ 0.1 for MNIST data. The data we received in this assignment is also handwritten data that is similar to MNIST data, so I expected the performance would be more stable if I added a similar level of noise. Since the data we received is more noise data than MNIST, I add a gaussian noise with a standard deviation of 0.06(the minimum level of noise in the paper).

​	This noise level can also be seen as a hyper parameter, but other hyper parameters other than 0.06 have not been tested much because of the time limitation. However, it seems that it contributes to the performance improvement properly when I choose noise level properly.



#### 3-4. Data Augmentation

​	We learned about data augmentation techniques in class. It is a technique to create more training data based on given training data to solve the problem of a small number of training data. To augment data, I have to investigate what types of input data are. So, I analyzed the .npy file given in the assignment and checked what real images we were trying to classify. The detailed process are described in [this page](https://github.com/JoonyoungYi/18s-cs492-assign1-data).

​	The task given in this assignment was to predict the digit from very noisy digit image data. 

![data-example-0](data-example-0.png) ![data-example-1](data-example-1.png) ![data-example-2](data-example-2.png) ![data-example-3](data-example-3.png) 

The four images shown above are examples. The images are very noisy as you can see. From seeing the examples of the training data, I realized that I can augment data by rotating the images and flipping data. In this idea, I was able to produce 8 times the data. 

​	I can see that the letter in the image were all white, and their thickness was different to each images. I thought I would use the opencv or pillow module to adjust the text thickness and add it to the training data, but I did not think the opencv or pillow module would be installed on the TA's computer and would not work properly. So this method did not proceed. It is expected that adding this method will definitely produce a classifier with better performance. 

​	However, instead of using the opencv or pillow modules, I mimiced it with numpy code. I was able to get a 1px thick text, and I added this data to my training data to train. Thus, the data size can be increased by a total of 16 times. 

​	Without data augmentation, we only had about 50% performance and we could achieve 60% performance with 16 times as much data.



#### 3-5. ReLU(Activation Function) 

​	It is not regularization technique, But I want to introduce my activation function. In the one of the lectures, we learned that **MAXOUT** activation works well with dropout regularization technique. But, when I tested the **MAXOUT** activation, **MAXOUT** exhibited lower performance than **ReLU**. I also tested leaky ReLU, but I've adopted ReLU as an activation function to reduce the number of hyper parameters because there is not much difference in performance between ReLU and leaky ReLU.



#### 3-6. Mini Batch

​	I also used mini batch technique, one of the most used regularization. The batch size I used was 8000. The total size of training data(after augmentation) is 16000. I used the 5% of data on each iteration. I've tested the lesser size of batch size. But, It seems to be not good. Because I used advarsarial learning. So, I have to increase min-batch size to 8000 to improve the effect of advarsarial learning.



### 4. Performance and Analyzation 

#### 4-1. The Testing Machine

​	I used GPU-version of Tensorflow. And, my machine is TITAN Xp with relatively large memory size. The detailed information of testing machine as below:

```
>> from tensorflow.python.client import device_lib
>> print(device_lib.list_local_devices())

[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 8981862457484279446
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 11970576384
locality {
  bus_id: 1
}
incarnation: 16036923018358585866
physical_device_desc: "device: 0, name: TITAN Xp, pci bus id: 0000:03:00.0, compute capability: 6.1"
]
```

I reported this detailed machine information to notify to TAs that this code doesn't works on the machine with lower memory size. 



#### 4-2. Performance

​	Now, I want to report the performance metrics of each networks are as below:

| The number of layers | Training Time (hh:mm:ss) | Training Accuracy | Training Loss | Evaluation Accuracy | Evaluation Loss |
| :------------------: | :----------------------: | :---------------: | :-----------: | :-----------------: | :-------------: |
|          3           |         03:48:05         |       73.0%       |    0.9730     |        58.0%        |     1.2468      |
|          5           |         04:35:46         |       73.2%       |    0.9278     |        57.4%        |     1.2713      |
|          7           |         05:40:25         |       70.2%       |    0.9906     |        58.8%        |     1.2215      |

I trained 100 epochs for each networks. The config file I've used in those experiments as below:

```
IMAGE_SIZE = 28
INPUT_LAYER_SIZE = IMAGE_SIZE * IMAGE_SIZE
OUTPUT_LAYER_SIZE = 10

hidden_layer_size = 500
hidden_layer_number = 6			# 2 or 4 
MODEL_FOLDER_NAME = './models/pa1-{}'
DROPOUT_RATE = 0.5
learning_rate = 1e-4

NOISE_STD = 0.06
TRAINING_DATA_NUMBER = 160000
BATCH_SIZE = 8000
assert TRAINING_DATA_NUMBER % BATCH_SIZE == 0
BATCH_ITER_NUMBER = TRAINING_DATA_NUMBER // BATCH_SIZE
EPOCH = 100
TRAIN_ITER_NUMBER = EPOCH * BATCH_ITER_NUMBER

EVERY_N_ITER = 1000
EARLY_STOP_TRAIN_LOSS = 0.001
EARLY_STOP_VALID_LOSS_MULTIPLIER = 1.5

BN_MOMENTUM = 0.9
```

​	Before showing the anlyzations of each results, I want to introduce another experiment on human. Recall 3.4 Section, I served some of images in training data. As you can see from the examples, there are some cases where there is a lot of noise in a given image so that it is hard to see what number it is in human's eyes. 

​	In fact, I also tried digit recognition experiments with humans. I experimented with two friends. I gave them 10 images and request them to answer which digit is. Of course, they were asked to do testing only with no training data. One friend hit 7 of 10, and one friend hit 5. I wan to emphasize that these training data is very hard to predict. And I thought that the final about 60% accuracy was pretty good.



#### 4-3. Performance Analysis

​	As the number of nodes increases, the time taken for 100 epochs becomes longer. This is because there are a lot of parameters to run. The number of nodes per layer is the same. So, the small number of layers means that the number of parameters is small, and the small number of parameters means that the network capacity is small. The small capacity of the network means that the network is less expressive. However, with the hyper parameter and normalization methods that I used in this code, there is not much difference in the performance according to the number of layers. Several additional experiments show that the number of nodes and other hyper parameters are not significantly affected. If the network capacity is low, the original training error should be higher. But, it indicates a similar level of training error and accuracy. 

​	Why did this result come out? I think that I use regularization techniques properly. That is to say, the capacity of 500 nodes per layer seems to be bigger than the proper capacity to learn this problem. In other words, if more layers are used, overfitting should result in more evalution error. But, in my case, the proper regularization schemes may be used to avoid overfitting. As a result, I concluded that using regularizations enabled me to construct a robust network independent to hyper parameters.



### 5. ETC

​	This codes will be uploaded in `Github`. You can see this code and materials [this page](https://github.com/JoonyoungYi/18s-cs492-assign1), also.