```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
```

# 回调函数

## callbacks参数_Checkpoint、EarlyStopping、
ReduceLROnPlateau、DIY


```python
import keras
callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor='acc',
        patience=1,),
    keras.callbacks.ModelCheckpoint(
        filepath='my_model.h5',
        monitor='val_loss',
        save_best_only=True,)
]
model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['acc'])
model.fit(x, y,
         epochs=10,
         batch_size=32,
         callbacks=callbacks_list,
         validation_data=(x_val, y_val))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [11], in <cell line: 14>()
          2 callbacks_list = [
          3     keras.callbacks.EarlyStopping(
          4         monitor='acc',
       (...)
          9         save_best_only=True,)
         10 ]
         11 model.compile(optimizer='rmsprop',
         12              loss='binary_crossentropy',
         13              metrics=['acc'])
    ---> 14 model.fit(x, y,
         15          epochs=10,
         16          batch_size=32,
         17          callbacks=callbacks_list,
         18          validation_data=(x_val, y_val))


    NameError: name 'x' is not defined



```python
callbacks_list = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,)
]
model.fit(x, y,
         epochs=10,
         batch_size=32,
         callbacks=callbacks_list,
         validation_data=(x_val, y_val))
```


```python
import keras
class ActivationLogger(keras.callbacks.Callback):
    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input,
                                                   layer_outputs)
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(f, activations)
        f.close()
```

# TensorBoard


```python
import keras
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import layers
# 最常见单词数、文本截断量
max_features = 2000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# 硬改成500列、25000条评论、500个特征字符
# 变成特征500列的单词对应索引
# 词嵌入：距离表示语义距离，方向表示相同的坐标变换
# Embedding的参数：第一个为最常见单词数、第二嵌入的维度，第三输入数据的特征量（序列长度
# Embedding返回（样本量、特征量、嵌入维度）
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = keras.models.Sequential()
# 输入长度
model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embd'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embd (Embedding)            (None, 500, 128)          256000    
                                                                     
     conv1d_4 (Conv1D)           (None, 494, 32)           28704     
                                                                     
     max_pooling1d_2 (MaxPooling  (None, 98, 32)           0         
     1D)                                                             
                                                                     
     conv1d_5 (Conv1D)           (None, 92, 32)            7200      
                                                                     
     global_max_pooling1d_2 (Glo  (None, 32)               0         
     balMaxPooling1D)                                                
                                                                     
     dense_2 (Dense)             (None, 1)                 33        
                                                                     
    =================================================================
    Total params: 291,937
    Trainable params: 291,937
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
```


```python
! mkdir my_log_dir
```

    mkdir: my_log_dir: File exists



```python
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='my_log_dir',
        histogram_freq=1,
        embeddings_freq=1)
]
history = model.fit(x_train, y_train,
                   epochs=20,
                   batch_size=128,
                   validation_split=0.2,
                   callbacks=callbacks)
```

    Epoch 1/20
    157/157 [==============================] - 13s 82ms/step - loss: 0.6335 - acc: 0.6573 - val_loss: 0.4171 - val_acc: 0.8410
    Epoch 2/20
    157/157 [==============================] - 13s 82ms/step - loss: 0.4356 - acc: 0.8500 - val_loss: 0.4662 - val_acc: 0.8434
    Epoch 3/20
    157/157 [==============================] - 13s 83ms/step - loss: 0.3948 - acc: 0.8810 - val_loss: 0.4644 - val_acc: 0.8634
    Epoch 4/20
    157/157 [==============================] - 13s 85ms/step - loss: 0.3504 - acc: 0.9021 - val_loss: 0.4939 - val_acc: 0.8710
    Epoch 5/20
    157/157 [==============================] - 14s 87ms/step - loss: 0.3065 - acc: 0.9181 - val_loss: 0.5244 - val_acc: 0.8716
    Epoch 6/20
    157/157 [==============================] - 14s 87ms/step - loss: 0.2796 - acc: 0.9306 - val_loss: 0.6114 - val_acc: 0.8682
    Epoch 7/20
    157/157 [==============================] - 14s 88ms/step - loss: 0.2269 - acc: 0.9503 - val_loss: 0.7420 - val_acc: 0.8648
    Epoch 8/20
    157/157 [==============================] - 14s 89ms/step - loss: 0.2108 - acc: 0.9571 - val_loss: 0.7751 - val_acc: 0.8664
    Epoch 9/20
    157/157 [==============================] - 14s 89ms/step - loss: 0.1801 - acc: 0.9712 - val_loss: 1.4105 - val_acc: 0.8078
    Epoch 10/20
    157/157 [==============================] - 14s 90ms/step - loss: 0.1563 - acc: 0.9793 - val_loss: 0.9810 - val_acc: 0.8640
    Epoch 11/20
    157/157 [==============================] - 14s 90ms/step - loss: 0.1350 - acc: 0.9864 - val_loss: 1.2624 - val_acc: 0.8432
    Epoch 12/20
    157/157 [==============================] - 14s 91ms/step - loss: 0.1243 - acc: 0.9878 - val_loss: 1.0894 - val_acc: 0.8696
    Epoch 13/20
    157/157 [==============================] - 14s 90ms/step - loss: 0.1191 - acc: 0.9894 - val_loss: 1.1511 - val_acc: 0.8628
    Epoch 14/20
    157/157 [==============================] - 14s 91ms/step - loss: 0.1172 - acc: 0.9897 - val_loss: 1.1220 - val_acc: 0.8694
    Epoch 15/20
    157/157 [==============================] - 14s 91ms/step - loss: 0.1157 - acc: 0.9895 - val_loss: 1.1832 - val_acc: 0.8662
    Epoch 16/20
    157/157 [==============================] - 14s 91ms/step - loss: 0.1156 - acc: 0.9902 - val_loss: 1.2614 - val_acc: 0.8644
    Epoch 17/20
    157/157 [==============================] - 14s 91ms/step - loss: 0.1171 - acc: 0.9901 - val_loss: 1.2307 - val_acc: 0.8694
    Epoch 18/20
    157/157 [==============================] - 14s 91ms/step - loss: 0.1179 - acc: 0.9896 - val_loss: 1.2111 - val_acc: 0.8696
    Epoch 19/20
    157/157 [==============================] - 14s 91ms/step - loss: 0.1106 - acc: 0.9912 - val_loss: 1.2025 - val_acc: 0.8714
    Epoch 20/20
    157/157 [==============================] - 14s 92ms/step - loss: 0.1117 - acc: 0.9908 - val_loss: 1.2334 - val_acc: 0.8698



```python
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
```




    
![png](https://github.com/fendao/imgs/blob/main/dl_7/output_11_0.png)
    



# 性能发挥
## 批标准化、深度可分离卷积


```python
from keras.models import Sequential, Model
from keras import layers
height = 64
width = 64
channels = 3
num_classes = 10

model = Sequential()
model.add(layers.SeparableConv2D(32, 3,
                                activation='relu',
                                input_shape=(height, width, channels,)))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentyropy')
```


```python
model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     separable_conv2d (Separable  (None, 62, 62, 32)       155       
     Conv2D)                                                         
                                                                     
     separable_conv2d_1 (Separab  (None, 60, 60, 64)       2400      
     leConv2D)                                                       
                                                                     
     max_pooling2d (MaxPooling2D  (None, 30, 30, 64)       0         
     )                                                               
                                                                     
     separable_conv2d_2 (Separab  (None, 28, 28, 64)       4736      
     leConv2D)                                                       
                                                                     
     separable_conv2d_3 (Separab  (None, 26, 26, 128)      8896      
     leConv2D)                                                       
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 13, 13, 128)      0         
     2D)                                                             
                                                                     
     separable_conv2d_4 (Separab  (None, 11, 11, 64)       9408      
     leConv2D)                                                       
                                                                     
     separable_conv2d_5 (Separab  (None, 9, 9, 128)        8896      
     leConv2D)                                                       
                                                                     
     global_average_pooling2d (G  (None, 128)              0         
     lobalAveragePooling2D)                                          
                                                                     
     dense_3 (Dense)             (None, 32)                4128      
                                                                     
     dense_4 (Dense)             (None, 10)                330       
                                                                     
    =================================================================
    Total params: 38,949
    Trainable params: 38,949
    Non-trainable params: 0
    _________________________________________________________________


## 超参优化、模型集成
