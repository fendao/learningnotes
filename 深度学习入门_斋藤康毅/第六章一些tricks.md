```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```


```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import Image
import sys, os
sys.path.append(os.pardir)
from common2.functions import softmax, cross_entropy_error
from common2.functions import *
from common2.gradient import numerical_gradient
from common2.layers import *
from collections import OrderedDict
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    def gradient(self, x, t):
        self.loss(x, t)
        
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads
```


```python
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
optimizer = SGD()
```


```python
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]
```


```python
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr*grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```


```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_liek(val)
                self.v[key] = np.zeros_like(val)
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        for key in params.keys():
            self.m[key] += (1-self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1-self.beta2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
```


```python
from common2.util import smooth_curve
from common2.multi_layer_net import MultiLayerNet
from common2.optimizer import *
```


```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()
networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100], output_size=10)
    train_loss[key] = []
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    if i % 100 == 0:
        print('========='+ 'iteration:'+str(i)+ '=======')
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ':' + str(loss))
```

    =========iteration:0=======
    SGD:2.351657790855041
    Momentum:2.494503989978383
    AdaGrad:2.0582646388370263
    Adam:2.2119898165683156

    =========iteration:1900=======
    SGD:0.2378771894712399
    Momentum:0.12075655489791401
    AdaGrad:0.0317038818553226
    Adam:0.0485309126618826



```python
markers = {'SGD': 'o', 'Momentum': 'x', 'AdaGrad': 's', 'Adam': 'D'}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel('iterations')
plt.ylabel('loss', rotation=0, labelpad=10)
plt.ylim(0,1)
plt.legend()
plt.show()
```


    
![](https://img-blog.csdnimg.cn/9ce9c5ecc3ad4eaa978bfc584d108a6a.png#pic_center)




```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def Relu(x):
    return np.maximum(0, x)
def tanh(x):
    return np.tanh(x)
```


```python
input_data = np.random.randn(1000, 100)
node_num = 100
w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
hidden_layer_size = 5
activations = {}
x = input_data
for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
#     w = np.random.randn(node_num, node_num) * 1
#     w = np.random.randn(node_num, node_num) * 0.01
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z
```


```python
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1)+ '-layer')
    if i != 0: plt.yticks([], [])
    plt.hist(a.ravel(), 30, range=(0,1))
plt.show()
```


    

![](https://img-blog.csdnimg.cn/028eb6d24cad49e1b59810e4d67f9402.png#pic_center)




```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
optimizer = SGD(lr=0.01)
networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100],
                                 output_size=10, weight_init_std=weight_type)
    train_loss[key] = []
    
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    if i % 100 == 0:
        print('iteration:'+str(i))
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ':' + str(loss))
```

    iteration:0
    std=0.01:2.302526554595131
    Xavier:2.305248117769575
    He:2.3881201553755456
  
    iteration:1900
    std=0.01:2.2937058362392113
    Xavier:0.3092111457688009
    He:0.21794875051740975



```python
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.ylim(0, 2.5)
plt.legend()
plt.show()
```


    
![](https://img-blog.csdnimg.cn/871b8816e9bd4557935ef6e1daa58a2e.png#pic_center)

    



```python
from common2.multi_layer_net_extend import MultiLayerNetExtend
from common2.optimizer import SGD, Adam
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
x_train = x_train[:1000]
t_train = t_train[:1000]
max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01
def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100,100,100,100,100], output_size=10,
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100,100,100,100,100], output_size=10,
                                 weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
            
            print('epoch:'+str(epoch_cnt)+'|'+str(train_acc)+'-'+str(bn_train_acc))
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
    return train_acc_list, bn_train_acc_list
```


```python
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)
for i, w in enumerate(weight_scale_list):
    print('------'+str(i+1)+'/16'+'-----')
    
    train_acc_list, bn_train_acc_list = __train(w)
    plt.subplot(4,4,i+1)
    plt.title('W:'+str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='BatchNorm', markevery=2)
        plt.plot(x, train_acc_list, linestyle='--', label='Norm(NoBatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle='--', markevery=2)
    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel('accuracy')
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel('epochs')
    plt.legend(loc='lower right')
plt.show()
```

    ------1/16-----
    epoch:0|0.097-0.098
    epoch:1|0.097-0.1


    epoch:2|0.097-0.15
    epoch:3|0.097-0.185
    epoch:4|0.097-0.219
    epoch:5|0.097-0.25
    epoch:6|0.097-0.275
    epoch:7|0.097-0.296
    epoch:8|0.097-0.316
    epoch:9|0.097-0.325
    epoch:10|0.097-0.351
    epoch:11|0.097-0.362
    epoch:12|0.097-0.374
    epoch:13|0.097-0.391
    epoch:14|0.097-0.401
    epoch:15|0.097-0.412
    epoch:16|0.097-0.43
    epoch:17|0.097-0.446


    

    epoch:19|0.116-0.601
    ------16/16-----
    epoch:0|0.117-0.263
    epoch:1|0.117-0.234
    epoch:2|0.116-0.297
    epoch:3|0.116-0.401
    epoch:4|0.116-0.335
    epoch:5|0.116-0.453
    epoch:6|0.116-0.452
    epoch:7|0.116-0.496
    epoch:8|0.116-0.492
    epoch:9|0.117-0.51
    epoch:10|0.117-0.499
    epoch:11|0.117-0.518
    epoch:12|0.117-0.51
    epoch:13|0.117-0.512
    epoch:14|0.117-0.524
    epoch:15|0.117-0.524
    epoch:16|0.117-0.509
    epoch:17|0.117-0.526
    epoch:18|0.117-0.526
    epoch:19|0.117-0.526



![](https://img-blog.csdnimg.cn/d49fd9547e5242e0ae1cb590421356d6.png#pic_center)




```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
x_train = x_train[:300]
t_train = t_train[:300]

weight_decay_lambda = 0
# weight_decay_lambda = 0.1

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100,100,100,100], output_size=10,
                       weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01)
max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0
for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        print('epoch:'+str(epoch_cnt)+',train acc:'+ str(train_acc)+',test acc:'+str(test_acc))
        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break
```

    epoch:0,train acc:0.07333333333333333,test acc:0.0905
 
    epoch:200,train acc:1.0,test acc:0.7637



```python
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
```


    
![](https://img-blog.csdnimg.cn/5ba8848bc23f4df2a27fa0c203e6a43a.png#pic_center)

    



```python
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    def backward(self, dout):
        return dout * self.mask
```


```python
from common2.trainer import Trainer
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
x_train = x_train[:300]
t_train = t_train[:300]

use_dropout = True
dropout_ratio=0.2

network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100,100,100,100,100,100],
                             output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                 epochs=301, mini_batch_size=100,
                 optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)
trainer.train()
train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list
```

    train loss:2.312183069291624
    === epoch:1, train acc:0.09666666666666666, test acc:0.1066 ===
    train loss:2.3169822700249454
    train loss:2.3112063670472973
    train loss:2.3288220824148604
    
    =============== Final Test Accuracy ===============
    test acc:0.5586



```python
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.axis([0, 300, 0, 1.0])
plt.legend(loc='lower right')
plt.show()
```


    

![](https://img-blog.csdnimg.cn/e57feb20fe6847b7b5e78afa9b759f4b.png#pic_center)




```python
from common2.util import *
(x_train, t_train), (x_test, t_test) = load_mnist()
x_train = x_train[:500]
t_train = t_train[:500]
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]
def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100,100,100],
                           output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                     epochs=epocs, mini_batch_size=100,
                     optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()
    return trainer.test_acc_list, trainer.train_acc_list
```


```python
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # 此处调整权值衰减系数与学习率
    weight_decay = 10 ** np.random.uniform(-8, -6)
    lr = 10 ** np.random.uniform(-3, -2)
    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print('val acc:'+str(val_acc_list[-1])+'| lr:'+str(lr), ', weight decay:'+str(weight_decay))
    key = 'lr:' + str(lr) + ', weight decay:' + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list
```

    val acc:0.5| lr:0.002037394644433468 , weight decay:1.5461805687350062e-07
   
    val acc:0.84| lr:0.009269786798484373 , weight decay:1.9876268683746202e-07



```python
print('------Hyper-Parameter Optimization Result------')
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0
plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
    wspace=None, hspace=0.45)
for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print('Best-'+str(i+1)+'(val acc:'+str(val_acc_list[-1])+')|'+key)
    
    plt.subplot(row_num, col_num, i+1)
    plt.title('Best-'+str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], '--')
    i += 1
    if i >= graph_draw_num:
        break
plt.show()
```

    ------Hyper-Parameter Optimization Result------
    Best-1(val acc:0.84)|lr:0.009269786798484373, weight decay:1.9876268683746202e-07
 
    Best-20(val acc:0.7)|lr:0.003517136148258115, weight decay:1.7365461511569004e-08



![](https://img-blog.csdnimg.cn/c1ff4e85ddc24b9dba4fc0ea6dee47a8.png#pic_center)



```python

```
