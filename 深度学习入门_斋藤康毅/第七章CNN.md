```python
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

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
from common2.util import im2col
from common2.util import col2im
```


```python
x1 = np.random.rand(1,3,7,7)
coll = im2col(x1,5,5,stride=1,pad=0)
print(coll.shape)
x2 = np.random.rand(10,3,7,7)
col2 = im2col(x2,5,5,stride=1,pad=0)
print(col2.shape)
```

    (9, 75)
    (90, 75)



```python
np.random.rand(1,3,7,7)
```




    array([[[[0.25749528, 0.69136824, 0.8576301 , 0.10787306, 
             ......
              0.78826597, 0.06973861]]]])




```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        # 加入反向
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None
    def forward(self, x):
        # 滤波与输入数据的批量通道高宽
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        # 输出数据的高宽
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)
        
        # 横向展开的数据
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 纵向展开的一列滤波
        col_W = self.W.reshape(FN, -1).T
        # 权重乘积累加与偏置
        out = np.dot(col, col_W) + self.b
        # 输出数据转换为批量通道高宽维度
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)
        
        # 加入反向
        self.x = x
        self.col = col
        self.col_W = col_W
        return out
    def backward(self, dout):
        # 滤波批量通道高宽
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1,0).reshape(FN, C, FH, FW)
        
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx
```


```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        # 反向
        self.x = None
        self.arg_max = None
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)
        
        self.x = x
        self.arg_max = arg_max
        return out
    def backward(self, dout):
        dout = dout.transpose(0,2,3,1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.ravel()] = dout.ravel()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx
```


```python
class SimpleConvNet:
    def __init__(self, input_dim=(1,28,28),
                conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                hidden_size=100, output_size=10, weight_init_std=0.01):
        # 取卷积层超参，并计算输出
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
        # 以下是要学习的参数。权重参数初始化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        # 生成各层
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                          self.params['b1'],
                                          conv_param['stride'],
                                          conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    def gradient(self, x, t):
        # 正向，预测概率类
        self.loss(x, t)
        # 反向，返回参数的梯度
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db
        return grads
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        acc =0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y==tt)
        return acc / x.shape[0]
    def save_params(self, file_name='params.pkl'):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    def load_params(self, file_name='params.pkl'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
#     def save_params(self, file_name="params.pkl"):
#         params = {}
#         for key, val in self.params.items():
#             params[key] = val
#         with open(file_name, 'wb') as f:
#             pickle.dump(params, f)

#     def load_params(self, file_name="params.pkl"):
#         with open(file_name, 'rb') as f:
#             params = pickle.load(f)
#         for key, val in params.items():
#             self.params[key] = val

#         for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
#             self.layers[key].W = self.params['W' + str(i+1)]
#             self.layers[key].b = self.params['b' + str(i+1)]
```


```python
from common2.trainer import Trainer
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
(x_train, t_train) = (x_train[:5000], t_train[:5000])
(x_test, t_test) = (x_test[:1000], t_test[:1000])
max_epochs = 20
network = SimpleConvNet(input_dim=(1,28,28),
                       conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                       hidden_size=100, output_size=10, weight_init_std=0.01)
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                 epochs=max_epochs, mini_batch_size=100,
                 optimizer='Adam', optimizer_param={'lr':0.001},
                 evaluate_sample_num_per_epoch=1000)
trainer.train()
network.save_params('params.pkl')
print('卷积网参数保存成功')
```

    train loss:2.2989758455005456
    ......
   
    =============== Final Test Accuracy ===============
    test acc:0.963
    卷积网参数保存成功



```python
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.legend(loc='lower right')
plt.ylim(0, 1.0)
plt.xlabel('epochs')
plt.ylabel('a\nc\nc\nu\nr\na\nc\ny', rotation=0, labelpad=10)
plt.show()
```


    
![](https://img-blog.csdnimg.cn/59983375a14e49dfbf0162cc19af75ce.png#pic_center)

    



```python
def filter_show(filters, nx=8, margin=3, scale=10):
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))
    
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
```


```python
network = SimpleConvNet()
filter_show(network.params['W1'])
network.load_params("params.pkl")
filter_show(network.params['W1'])
```


    
![](https://img-blog.csdnimg.cn/08b6ce1d5d74444f9abd19397d8e5bd2.png#pic_center)

    



    
![](https://img-blog.csdnimg.cn/06f7138aed56415883633cac65a6a115.png#pic_center)

    



```python

```
