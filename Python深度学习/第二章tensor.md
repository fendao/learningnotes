```python
import pretty_errors
```


```python
# %colors Nocolor
# %colors LightBG
%colors Neutral
# %colors Linux
```


```python
import random
import tensorflow as tf


def Z(a):
    return a*(4**2 + 4)


def Y(a, x):
    return a*(x**2 + 4)


x = tf.Variable(0.0, name='var')
optimizer = tf.keras.optimizers.Adam(0.01)

for _ in range(500):
    input = random.randint(1, 255)
    target = Z(input)
    with tf.GradientTape() as tape:
        pre = Y(input, x)
        loss = abs(pre - target)
        gradient = tape.gradient(loss, x)
        optimizer.apply_gradients([(gradient, x)])
        if _ % 100 == 0:
            print("loss:", loss.numpy())
print("x: ", x.numpy())
```

    loss: 1360.0
    loss: 1792.0
    loss: 1088.0
    loss: 608.0
    loss: 1280.0
    x:  0.0



```python
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```


```python
import pretty_errors
pretty_errors.configure(
    separator_character = '*',
    filename_display    = pretty_errors.FILENAME_EXTENDED,
    line_number_first   = True,
    display_link        = True,
    lines_before        = 5,
    lines_after         = 2,
    code_color          = '  ' + pretty_errors.default_config.line_color,
    display_timestamp   = True,
    display_locals      = False,
    line_color          = pretty_errors.CYAN_BACKGROUND + pretty_errors.BRIGHT_RED,
    reset_stdout = True,
)
```


```python
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
```

    (60000, 28, 28)
    (60000,)
    (10000, 28, 28)
    (10000,)



```python
from keras import models
from keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation='softmax'))
```


```python
network.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])
```


```python
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255
```


```python
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```


```python
network.fit(train_images, train_labels, epochs=5, batch_size=128)
```

    Epoch 1/5
    469/469 [==============================] - 1s 2ms/step - loss: 0.0281 - accuracy: 0.9916
    Epoch 2/5
    469/469 [==============================] - 1s 2ms/step - loss: 0.0218 - accuracy: 0.9936
    Epoch 3/5
    469/469 [==============================] - 1s 3ms/step - loss: 0.0162 - accuracy: 0.9952
    Epoch 4/5
    469/469 [==============================] - 1s 3ms/step - loss: 0.0128 - accuracy: 0.9962
    Epoch 5/5
    469/469 [==============================] - 1s 2ms/step - loss: 0.0102 - accuracy: 0.9971





    <keras.callbacks.History at 0x16bf9df30>




```python
test_loss, test_acc = network.evaluate(test_images, test_labels)
```

    313/313 [==============================] - 0s 604us/step - loss: 0.0666 - accuracy: 0.9823



```python
print('test_acc:', test_acc)
```

    test_acc: 0.9822999835014343



```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = train_images[4]
```


```python
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
```


    
![png](https://github.com/fendao/imgs/blob/main/dl_tensor/output_14_0.png)
    


# 张量运算


```python
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x
def naive_add(x, y):
    assert len(x.shape) == 2
    assert len(x.shape) == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x
```


```python
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x
```


```python
import numpy as np
x = np.random.random((64,3,32,10))
y = np.random.random((32,10))
z = np.maximum(x, y)
```


```python
def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z
```


```python
def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z
```

# 梯度下降


```python
past_velocity = 0.
momentum = 0.1
while loss > 0.01:
    w, loss, gradient = get_current_parameters()
    velocity = past_velocity * momentum - learning_rate * gradient
    w = w + momentum * velocity - learning * gradient
    past_velocity = velocity
    update_parameter(w)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [77], in <cell line: 3>()
          2 momentum = 0.1
          3 while loss > 0.01:
    ----> 4     w, loss, gradient = get_current_parameters()
          5     velocity = past_velocity * momentum - learning_rate * gradient
          6     w = w + momentum * velocity - learning * gradient


    NameError: name 'get_current_parameters' is not defined

