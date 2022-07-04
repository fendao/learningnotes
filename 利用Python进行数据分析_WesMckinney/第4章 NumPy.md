# 4.1ndarray对象（数组
import numpy as np标准导入  
np.random.randn(2, 3)生成2行3列数组对象  
shape属性表示维数，dtype描述数据类型  
## 生成ndarray
二维数组np.array([[1, 2], [3, 4]])，三维数组np.array([[[1, 2],[3, 4]],[[2, 1],[4, 3]]])  
np.zeros((3, 6))三行六列二维全零数组，ones全1，np.empty((2, 3, 2))二维三行两列无初始值数组  
np.arange生成等差数组  
**数组生成函数表**  asarray、eye,identity  
## ndarray数据类型
dtype获取  
**数据类型表**  
arr.astype(np.float64)数据类型转换，并生成新数组
## NumPy数组算数
向量化：不需要for循环进行批量操作  
加减乘除，比较大小都是逐元素操作  
但是如果运算尺寸不同的两数组，则会用到广播特性
## 索引切片
**切片是视图，给切片赋值即是修改原数组**，此特性也是numpy省内存的原因  
二维数组，递归获取单元素arr2d[0]\[2\]、arr2d[0, 2]  第一行第三个  
arr3d[0]获取第一个二维数组，arr3d[1, 0]第二个二维数组的第一行
### 二维数组切片
arr2d[:2, 1:]前2行，后2列  
arr2d[:2, 2]前2行，第3列  
arr2d[:, :2]全选行，前两列
## 布尔索引（用布尔值索引
利用比较操作符产生的布尔值数组作为索引data[names == 'Bob', 2:]  
np.random.randn(7, 4)7行4列随机正态数组  
逻辑符号：取反!= ~  和& |  
布尔值索引出来的数据会生成拷贝
## 神奇索引（用整数数组索引
arr[4, 3, 0, 6]要所有这些行组成的数组，负数是倒着数  
np.arange(32).reshape((8, 4))生成32位等差8行4列数组  
arr[[1, 5, 7, 2], [0, 3, 1, 2]]要2683行的分别1423列
## 数组转置
np.dot(arr.T, arr)自乘转置数组  
二维数组转置axis=0纵，1横  
三维数组转置：？？？  
    arr.transpose((1,0,2))  
    arr.swapaxed(1, 2)  axis为0两个二维数组的相同坐标，axis为1每个二维数组每行的相同列
# 4.2通用函数（逐元素数组函数
简单函数的向量化封装  
np.sqrt()平方根  
np.exp()指数函数  
np.modf()返回浮点数组的小数整数部分  
**一元通用函数表**  
**二元通用函数表**
# 4.3面向数组编程
向量化：使用数组表达式，不使用循环  
np.meshgrid(arr1d,arr1d)生成二维矩阵
## 数组操作条件逻辑
numpy.where(cond, xarr, yarr)  
即x if condition else y
## 数学统计
注意axis为0或1  
arr.mean()、arr.sum()、arr.cumsum()  
**基础数组统计方法表**
## 布尔数组方法
0为False，1为True  
bools.any()检查至少一True，bool.all()检查是否全True
## 排序
arr.sort()从小到大，**返回一份拷贝**  
large_arr[int(0.05 * len(large_arr))]返回百分之五分位数
## 唯一值与集合逻辑
np.unique(arr)  
sorted(set(names))  
np.in1d(arr1, arr2)检查数组1是否在2中  
**数组集合操作表**
## 数组的文件操作
np.savez('array_archive', a=arr, b=brr)，arch = np.load('array_archive.npz')

# 4.5线性代数
点乘的三种表示：x.dot(y) np.dot(x, y) x@y  
X.T.dot(X)自乘转置阵  
**numpy.linalg矩阵操作函数集**  
inv求逆阵，qr计算QR分解

# 4.6生成伪随机数
np.random.normal(size=(4, 4))生成4乘4正态分布样本  
np.random.seed()更改随机数种子  
**numpy.random的函数表**  
randn均值0方差1的正态分布样本，normal高斯分布样本，uniform均匀分布样本

# 4.7随机漫步
py实现：定义初始位置、步数列表、总步数、对总步数循环遍历、单步随机1或-1、位置追加、步数列表追加  
np.random模块实现：定义总次数nsteps、randint函数生成nsteps行的0或1随机数、np.where实现方向1或-1、cumsum实现步数累加  
np.abs(walk)>=10 表示是否同方向连续走10步  
(np.abs(walk)>=10).argmax()获取第一次的位置
## 多次漫步
总次数（行数）nwalks、实验步数nsteps  
0或1数组（实验结果数组） draws = np.random.randint(0, 2, size=(nwalks, nsteps))  
1或-1方向数组          steps = np.where(draws>0, 1, -1)  
步数累加               walks = steps.cumsum(1)  
检查各行是否包含同方向30 a = (np.abs(walks)>=30).any(1)  1是横向，生成布尔值数组  
True行30步的位置       (np.abs(walks[a])>=30).argmax(1)  1是横向


```python
import numpy as np
```


```python
my_arr = np.arange(1000000)
```


```python
my_list = list(range(1000000))
```


```python
%time for _ in range(10): my_arr2 = my_arr * 2
```

    CPU times: user 15.7 ms, sys: 20.5 ms, total: 36.1 ms
    Wall time: 34.2 ms



```python
%time for _ in range(10): my_list2 = [x * 2 for x in my_list]
```

    CPU times: user 432 ms, sys: 89.2 ms, total: 522 ms
    Wall time: 520 ms



```python
data = np.random.randn(2, 3)
```


```python
data
```




    array([[ 0.78964053,  0.02030056, -1.17439476],
           [-0.31484583, -0.81007074, -0.10189272]])




```python
data * 10
```




    array([[  7.89640534,   0.20300562, -11.74394757],
           [ -3.14845831,  -8.10070743,  -1.01892724]])




```python
data + data
```




    array([[ 1.57928107,  0.04060112, -2.34878951],
           [-0.62969166, -1.62014149, -0.20378545]])




```python
data.shape
```




    (2, 3)




```python
data.dtype
```




    dtype('float64')




```python
data1 = [6, 7.5, 8, 0, 1]
```


```python
arr1 = np.array(data1)
```


```python
arr1
```




    array([6. , 7.5, 8. , 0. , 1. ])




```python
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
```


```python
arr2 = np.array(data2)
```


```python
arr2
```




    array([[1, 2, 3, 4],
           [5, 6, 7, 8]])




```python
arr2.ndim
```




    2




```python
arr2.shape
```




    (2, 4)




```python
np.zeros(10)
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
np.empty((2, 3, 2))
```




    array([[[0.00000000e+000, 2.47032823e-322],
            [0.00000000e+000, 0.00000000e+000],
            [0.00000000e+000, 3.76231868e+174]],
    
           [[5.33359226e-091, 3.92902028e-061],
            [5.50924231e+169, 3.43881765e+175],
            [3.99910963e+252, 1.46030983e-319]]])




```python
np.arange(15)
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])




```python
 arr1 = np.array([1, 2, 3], dtype = np.float64)
```


```python
arr2 = np.array([1, 2, 3], dtype=np.int32)
```


```python
arr1.dtype
```




    dtype('float64')




```python
arr2.dtype
```




    dtype('int32')




```python
arr = np.array([1, 2, 3, 4, 5])
```


```python
arr.dtype
```




    dtype('int64')




```python
float_arr = arr.astype(np.float64)
```


```python
float_arr.dtype
```




    dtype('float64')




```python
arr = np.array([3.7, 1.2, 2.2, -4.4])
```


```python
arr
```




    array([ 3.7,  1.2,  2.2, -4.4])




```python
arr.astype(np.int32)
```




    array([ 3,  1,  2, -4], dtype=int32)




```python
numeric_string = np.array(['1.4', '3.2', '-4.7'], dtype=np.string_)
```


```python
numeric_string.astype(float)
```




    array([ 1.4,  3.2, -4.7])




```python
int_array = np.arange(10)
```


```python
calibers = np.array([.22, .270, .357, .380], dtype=np.float64)
```


```python
int_array.astype(calibers.dtype)
```




    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])




```python
empty_unit32 = np.empty(8, dtype='u4')
```


```python
empty_unit32
```




    array([2576980378, 1074633113,  858993459, 1072902963, 2576980378,
           1073846681, 2576980378, 3222378905], dtype=uint32)




```python
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
```


```python
arr
```




    array([[1., 2., 3.],
           [4., 5., 6.]])




```python
arr * arr
```




    array([[ 1.,  4.,  9.],
           [16., 25., 36.]])




```python
arr - arr
```




    array([[0., 0., 0.],
           [0., 0., 0.]])




```python
1/arr
```




    array([[1.        , 0.5       , 0.33333333],
           [0.25      , 0.2       , 0.16666667]])




```python
arr ** 0.5
```




    array([[1.        , 1.41421356, 1.73205081],
           [2.        , 2.23606798, 2.44948974]])




```python
arr2 = np.array([[0., 4., 1], [7., 2., 12.]])
```


```python
arr2
```




    array([[ 0.,  4.,  1.],
           [ 7.,  2., 12.]])




```python
arr2 > arr
```




    array([[False,  True, False],
           [ True, False,  True]])




```python
arr = np.arange(10)
```


```python
arr
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
arr[5]
```




    5




```python
arr[5:8]
```




    array([5, 6, 7])




```python
arr[5:8] = 12
```


```python
arr
```




    array([ 0,  1,  2,  3,  4, 12, 12, 12,  8,  9])




```python
arr_slice = arr[5:8]
```


```python
arr_slice
```




    array([12, 12, 12])




```python
arr_slice[1] = 12345
```


```python
arr
```




    array([    0,     1,     2,     3,     4,    12, 12345,    12,     8,
               9])




```python
arr_slice[:] = 64
```


```python
arr
```




    array([ 0,  1,  2,  3,  4, 64, 64, 64,  8,  9])




```python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```


```python
arr2d[2]
```




    array([7, 8, 9])




```python
arr2d[0][2]
```




    3




```python
arr2d[0, 2]
```




    3




```python
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
```


```python
arr3d
```




    array([[[ 1,  2,  3],
            [ 4,  5,  6]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
old_values = arr3d[0].copy()
```


```python
arr3d[0] = 42
```


```python
arr3d
```




    array([[[42, 42, 42],
            [42, 42, 42]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
arr3d[0]= old_values
```


```python
arr3d
```




    array([[[ 1,  2,  3],
            [ 4,  5,  6]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
arr3d[1, 0]
```




    array([7, 8, 9])




```python
arr
```




    array([ 0,  1,  2,  3,  4, 64, 64, 64,  8,  9])




```python
arr[1:6]
```




    array([ 1,  2,  3,  4, 64])




```python
arr2d
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
arr2d[:2]
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
arr2d[:2, 1:]
```




    array([[2, 3],
           [5, 6]])




```python
arr2d[1, :2]
```




    array([4, 5])




```python
arr2d[:2, 2]
```




    array([3, 6])




```python
arr2d[:, :1]
```




    array([[1],
           [4],
           [7]])




```python
arr2d[:1]
```




    array([[1, 2, 3]])




```python
arr2d[:2, 1:] = 0
```


```python
arr2d
```




    array([[1, 0, 0],
           [4, 0, 0],
           [7, 8, 9]])




```python
arr2d[:2, 1:]
```




    array([[0, 0],
           [0, 0]])




```python
arr2d[2]
```




    array([7, 8, 9])




```python
arr2d[2, :]
```




    array([7, 8, 9])




```python
arr2d[2:, :]
```




    array([[7, 8, 9]])




```python
arr2d[:, :2]
```




    array([[1, 0],
           [4, 0],
           [7, 8]])




```python
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
```


```python
data = np.random.randn(7, 4)
```


```python
names
```




    array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'], dtype='<U4')




```python
data
```




    array([[-0.34265503,  0.74639569,  1.88236036, -0.0289728 ],
           [ 0.12827425, -0.8322475 , -0.45584797, -0.0556073 ],
           [ 1.55722525,  0.43260405,  2.46171687, -0.47351526],
           [-0.98686969,  0.35934022,  0.39846314, -1.08751914],
           [ 0.38991196,  0.81299871,  2.14056279,  0.92477939],
           [-0.04927175, -0.18901597, -0.04317426, -0.1633588 ],
           [-2.85028648,  0.74275962,  0.37161692, -2.2697054 ]])




```python
names == 'Bob'
```




    array([ True, False, False,  True, False, False, False])




```python
data[names == 'Bob']
```




    array([[-0.34265503,  0.74639569,  1.88236036, -0.0289728 ],
           [-0.98686969,  0.35934022,  0.39846314, -1.08751914]])




```python
data[names =='Bob', 2]
```




    array([1.88236036, 0.39846314])




```python
data[names =='Bob', 3]
```




    array([-0.0289728 , -1.08751914])




```python
names != 'Bob'
```




    array([False,  True,  True, False,  True,  True,  True])




```python
data[~(names == 'Bob')]
```




    array([[ 0.12827425, -0.8322475 , -0.45584797, -0.0556073 ],
           [ 1.55722525,  0.43260405,  2.46171687, -0.47351526],
           [ 0.38991196,  0.81299871,  2.14056279,  0.92477939],
           [-0.04927175, -0.18901597, -0.04317426, -0.1633588 ],
           [-2.85028648,  0.74275962,  0.37161692, -2.2697054 ]])




```python
cond = names == 'Bob'
```


```python
data[~cond]
```




    array([[ 0.12827425, -0.8322475 , -0.45584797, -0.0556073 ],
           [ 1.55722525,  0.43260405,  2.46171687, -0.47351526],
           [ 0.38991196,  0.81299871,  2.14056279,  0.92477939],
           [-0.04927175, -0.18901597, -0.04317426, -0.1633588 ],
           [-2.85028648,  0.74275962,  0.37161692, -2.2697054 ]])




```python
mask = (names == 'Bob') | (names == 'Will')
```


```python
data[mask]
```




    array([[-0.34265503,  0.74639569,  1.88236036, -0.0289728 ],
           [ 1.55722525,  0.43260405,  2.46171687, -0.47351526],
           [-0.98686969,  0.35934022,  0.39846314, -1.08751914],
           [ 0.38991196,  0.81299871,  2.14056279,  0.92477939]])




```python
data[data < 0] = 0
```


```python
data
```




    array([[0.        , 0.74639569, 1.88236036, 0.        ],
           [0.12827425, 0.        , 0.        , 0.        ],
           [1.55722525, 0.43260405, 2.46171687, 0.        ],
           [0.        , 0.35934022, 0.39846314, 0.        ],
           [0.38991196, 0.81299871, 2.14056279, 0.92477939],
           [0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.74275962, 0.37161692, 0.        ]])




```python
data[names != 'Joe'] = 7
```


```python
data
```




    array([[7.        , 7.        , 7.        , 7.        ],
           [0.12827425, 0.        , 0.        , 0.        ],
           [7.        , 7.        , 7.        , 7.        ],
           [7.        , 7.        , 7.        , 7.        ],
           [7.        , 7.        , 7.        , 7.        ],
           [0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.74275962, 0.37161692, 0.        ]])




```python
arr = np.empty((8, 4))
```


```python
for i in range(8):
    arr[i] = i
```


```python
arr
```




    array([[0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [2., 2., 2., 2.],
           [3., 3., 3., 3.],
           [4., 4., 4., 4.],
           [5., 5., 5., 5.],
           [6., 6., 6., 6.],
           [7., 7., 7., 7.]])




```python
arr[[4, 3, 0, 6]]
```




    array([[4., 4., 4., 4.],
           [3., 3., 3., 3.],
           [0., 0., 0., 0.],
           [6., 6., 6., 6.]])




```python
arr[[-3, -5, -7]]
```




    array([[5., 5., 5., 5.],
           [3., 3., 3., 3.],
           [1., 1., 1., 1.]])




```python
arr = np.arange(32).reshape((8, 4))
```


```python
arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [28, 29, 30, 31]])




```python
arr[[1, 5, 7, 2], [0, 3, 1, 2]]
```




    array([ 4, 23, 29, 10])




```python
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]
```




    array([[ 4,  7,  5,  6],
           [20, 23, 21, 22],
           [28, 31, 29, 30],
           [ 8, 11,  9, 10]])




```python
arr = np.arange(15).reshape((3, 5))
```


```python
arr
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])




```python
arr.T
```




    array([[ 0,  5, 10],
           [ 1,  6, 11],
           [ 2,  7, 12],
           [ 3,  8, 13],
           [ 4,  9, 14]])




```python
arr = np.random.randn(6, 3)
```


```python
arr
```




    array([[ 0.37807541,  1.02700807,  0.11754885],
           [ 0.28372899, -0.64198134,  0.42085648],
           [ 0.87462074,  0.52887815,  0.4048748 ],
           [ 2.18106336, -2.35173998, -1.10350659],
           [ 0.80003679, -1.13883929,  0.64064729],
           [-0.01648363,  0.8012621 ,  1.29196073]])




```python
np.dot(arr.T, arr)
```




    array([[ 6.38577257, -5.38490937, -1.39760917],
           [-5.38490937,  9.21625455,  2.96543648],
           [-1.39760917,  2.96543648,  3.65217979]])




```python
arr = np.arange(16).reshape((2, 2, 4))
```


```python
arr
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7]],
    
           [[ 8,  9, 10, 11],
            [12, 13, 14, 15]]])




```python
arr.transpose((1, 0, 2))
```




    array([[[ 0,  1,  2,  3],
            [ 8,  9, 10, 11]],
    
           [[ 4,  5,  6,  7],
            [12, 13, 14, 15]]])




```python
np.transpose??
```


```python
arr
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7]],
    
           [[ 8,  9, 10, 11],
            [12, 13, 14, 15]]])




```python
arr.swapaxes(1, 2)
```




    array([[[ 0,  4],
            [ 1,  5],
            [ 2,  6],
            [ 3,  7]],
    
           [[ 8, 12],
            [ 9, 13],
            [10, 14],
            [11, 15]]])




```python
np.swapaxes??
```


```python
arr = np.arange(10)
```


```python
arr
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.sqrt(arr)
```




    array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ,
           2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ])




```python
np.exp(arr)
```




    array([1.00000000e+00, 2.71828183e+00, 7.38905610e+00, 2.00855369e+01,
           5.45981500e+01, 1.48413159e+02, 4.03428793e+02, 1.09663316e+03,
           2.98095799e+03, 8.10308393e+03])




```python
x = np.random.randn(8)
```


```python
y = np.random.randn(8)
```


```python
x
```




    array([-0.8181999 ,  0.2232388 ,  0.81795466, -0.47110494,  0.05137669,
            1.37628491,  0.98922121, -0.27149077])




```python
y
```




    array([-1.15570058, -0.46513045, -1.04093349,  0.20068236, -1.01011185,
           -1.0160647 , -0.19757365, -1.4213736 ])




```python
np.maximum(x, y)
```




    array([-0.8181999 ,  0.2232388 ,  0.81795466,  0.20068236,  0.05137669,
            1.37628491,  0.98922121, -0.27149077])




```python
arr = np.random.randn(7) * 5
```


```python
arr
```




    array([ 9.02072432,  2.71629302,  1.50993182, -0.6299129 ,  7.00501799,
            2.10903902, -2.82435162])




```python
remainder, whole_part = np.modf(arr)
```


```python
remainder
```




    array([ 0.02072432,  0.71629302,  0.50993182, -0.6299129 ,  0.00501799,
            0.10903902, -0.82435162])




```python
whole_part
```




    array([ 9.,  2.,  1., -0.,  7.,  2., -2.])




```python
arr
```




    array([ 9.02072432,  2.71629302,  1.50993182, -0.6299129 ,  7.00501799,
            2.10903902, -2.82435162])




```python
np.sqrt(arr)
```

 


    array([3.00345207, 1.64811802, 1.22879283,        nan, 2.64669945,
           1.45225308,        nan])




```python
np.sqrt(arr, arr)
```

   



    array([3.00345207, 1.64811802, 1.22879283,        nan, 2.64669945,
           1.45225308,        nan])




```python
arr
```




    array([3.00345207, 1.64811802, 1.22879283,        nan, 2.64669945,
           1.45225308,        nan])




```python
points = np.arange(-5, 5, 0.01)
```


```python
points
```




    array([-5.0000000e+00, -4.9900000e+00, -4.9800000e+00, -4.9700000e+00,
          ......
            4.9600000e+00,  4.9700000e+00,  4.9800000e+00,  4.9900000e+00])




```python
xs, ys = np.meshgrid(points, points)
```


```python
ys
```




    array([[-5.  , -5.  , -5.  , ..., -5.  , -5.  , -5.  ],
           [-4.99, -4.99, -4.99, ..., -4.99, -4.99, -4.99],
           [-4.98, -4.98, -4.98, ..., -4.98, -4.98, -4.98],
           ...,
           [ 4.97,  4.97,  4.97, ...,  4.97,  4.97,  4.97],
           [ 4.98,  4.98,  4.98, ...,  4.98,  4.98,  4.98],
           [ 4.99,  4.99,  4.99, ...,  4.99,  4.99,  4.99]])




```python
z = np.sqrt(xs ** 2 + ys ** 2)
```


```python
z
```




    array([[7.07106781, 7.06400028, 7.05693985, ..., 7.04988652, 7.05693985,
            7.06400028],
           [7.06400028, 7.05692568, 7.04985815, ..., 7.04279774, 7.04985815,
            7.05692568],
           [7.05693985, 7.04985815, 7.04278354, ..., 7.03571603, 7.04278354,
            7.04985815],
           ...,
           [7.04988652, 7.04279774, 7.03571603, ..., 7.0286414 , 7.03571603,
            7.04279774],
           [7.05693985, 7.04985815, 7.04278354, ..., 7.03571603, 7.04278354,
            7.04985815],
           [7.06400028, 7.05692568, 7.04985815, ..., 7.04279774, 7.04985815,
            7.05692568]])




```python
import matplotlib.pyplot as plt
```


```python
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x7f780e7b7210>




```python
plt.title('Image plot of $\sqrt{x^2 + y^2}$ for a grid of values')
```




    Text(0.5, 1.0, 'Image plot of $\\sqrt{x^2 + y^2}$ for a grid of values')




```python
plt.show()
```


    
![png](https://github.com/fendao/imgs/blob/main/usepy_4/output_157_0.png)
    



```python
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
```


```python
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
```


```python
cond = np.array([True, False, True, True, False])
```


```python
result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
```


```python
result
```




    [1.1, 2.2, 1.3, 1.4, 2.5]




```python
result = np.where(cond, xarr, yarr)
```


```python
result
```




    array([1.1, 2.2, 1.3, 1.4, 2.5])




```python
np.where??
```


```python
arr = np.random.randn(4, 4)
```


```python
arr
```




    array([[ 1.03161975, -0.07373538,  0.89284631,  0.49699086],
           [ 0.07184594, -0.11277721, -1.12213215, -0.31011687],
           [ 1.02449438, -0.0209985 , -0.01611278, -0.00971766],
           [ 0.6882975 ,  1.07682616,  0.96890584, -0.80508982]])




```python
arr >0 
```




    array([[ True, False,  True,  True],
           [ True, False, False, False],
           [ True, False, False, False],
           [ True,  True,  True, False]])




```python
np.where(arr>0, 2, -2)
```




    array([[ 2, -2,  2,  2],
           [ 2, -2, -2, -2],
           [ 2, -2, -2, -2],
           [ 2,  2,  2, -2]])




```python
np.where(arr>0, 2, arr)
```




    array([[ 2.        , -0.07373538,  2.        ,  2.        ],
           [ 2.        , -0.11277721, -1.12213215, -0.31011687],
           [ 2.        , -0.0209985 , -0.01611278, -0.00971766],
           [ 2.        ,  2.        ,  2.        , -0.80508982]])




```python
arr = np.random.randn(5, 4)
```


```python
arr
```




    array([[ 0.72345285, -0.06208981,  0.56013037, -0.15391921],
           [-0.45321083, -2.7499984 ,  0.48296228,  1.06334137],
           [-0.7253634 ,  0.30226873,  0.80927719, -0.95316146],
           [-0.48854369, -1.1781029 ,  0.42924121, -1.44288921],
           [ 0.68800887,  0.6266169 , -0.53456849,  1.56713044]])




```python
arr.mean()
```




    -0.07447085929163098




```python
np.mean(arr)
```




    -0.07447085929163098




```python
arr.sum()
```




    -1.4894171858326197




```python
arr.mean(axis=1)
```




    array([ 0.26689355, -0.4142264 , -0.14174473, -0.67007365,  0.58679693])




```python
arr.sum(axis=0)
```




    array([-0.25565619, -3.06130548,  1.74704257,  0.08050192])




```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
```


```python
arr.cumsum()
```




    array([ 0,  1,  3,  6, 10, 15, 21, 28])




```python
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
```


```python
arr
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
arr.cumsum(axis=0)
```




    array([[ 0,  1,  2],
           [ 3,  5,  7],
           [ 9, 12, 15]])




```python
arr.cumprod(axis=1)
```




    array([[  0,   0,   0],
           [  3,  12,  60],
           [  6,  42, 336]])




```python
arr.sum(axis=0)
```




    array([ 9, 12, 15])




```python
arr = np.random.randn(100)
```


```python
(arr > 0).sum()
```




    50




```python
bools = np.array([False, False, True, False])
```


```python
bools.any()
```




    True




```python
bools.all()
```




    False




```python
arr = np.random.randn(6)
```


```python
arr
```




    array([-0.62640628, -0.13433304,  0.39549543, -0.63006546,  1.01894547,
           -0.08093893])




```python
arr.sort()
```


```python
arr
```




    array([-0.63006546, -0.62640628, -0.13433304, -0.08093893,  0.39549543,
            1.01894547])




```python
arr = np.random.randn(5, 3)
```


```python
arr
```




    array([[ 2.43673387, -1.22252588, -0.1331568 ],
           [-1.71264497, -0.06367967,  0.44597924],
           [-0.71070856, -1.08637326, -1.0784576 ],
           [-1.01944857,  0.70263375, -1.87898785],
           [ 0.27618568,  0.51428372, -0.17907052]])




```python
arr.sort()
```


```python
arr
```




    array([[-1.22252588, -0.1331568 ,  2.43673387],
           [-1.71264497, -0.06367967,  0.44597924],
           [-1.08637326, -1.0784576 , -0.71070856],
           [-1.87898785, -1.01944857,  0.70263375],
           [-0.17907052,  0.27618568,  0.51428372]])




```python
large_arr = np.random.randn(1000)
```


```python
large_arr.sort()
```


```python
large_arr[int(0.05 * len(large_arr))]
```




    -1.5186026251198514




```python
names = np.array(['Joe', 'Bob', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
```


```python
np.unique(names)
```




    array(['Bob', 'Joe', 'Will'], dtype='<U4')




```python
ints = np.array([3, 3, 2, 2, 1, 1, 4, 4])
```


```python
np.unique(ints)
```




    array([1, 2, 3, 4])




```python
sorted(set(names))
```




    ['Bob', 'Joe', 'Will']




```python
values = np.array([6, 0, 0, 3, 2, 5, 6])
```


```python
np.in1d(values, [2, 3, 6])
```




    array([ True, False, False,  True,  True, False,  True])




```python
arr = np.arange(10)
```


```python
np.save('some_array', arr)
```


```python
np.load('some_array.npy')
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.savez('array_archive.npz', a=arr, b=arr)
```


```python
arch = np.load('array_archive.npz')
```


```python
arch['b']
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.savez_compressed('arrays_compressed.npz', a=arr, b=arr)
```


```python
x = np.array([[1., 2., 3.], [4., 5., 6.]])
```


```python
y = np.array([[6., 23.], [-1, 7], [8, 9]])
```


```python
x
```




    array([[1., 2., 3.],
           [4., 5., 6.]])




```python
y
```




    array([[ 6., 23.],
           [-1.,  7.],
           [ 8.,  9.]])




```python
x.dot(y)
```




    array([[ 28.,  64.],
           [ 67., 181.]])




```python
np.dot(x, np.ones(3))
```




    array([ 6., 15.])




```python
x @ np.ones(3)
```




    array([ 6., 15.])




```python
from numpy.linalg import inv, qr
```


```python
X = np.random.randn(5, 5)
```


```python
mat = X.T.dot(X)
```


```python
inv(mat)
```




    array([[ 3.11557603,  1.11986666,  0.03576491, -0.30802192,  0.2188662 ],
           [ 1.11986666,  1.00789215,  1.36847578, -1.46572924, -0.42792998],
           [ 0.03576491,  1.36847578,  4.18656045, -4.01278659, -1.5226549 ],
           [-0.30802192, -1.46572924, -4.01278659,  4.00271489,  1.48116394],
           [ 0.2188662 , -0.42792998, -1.5226549 ,  1.48116394,  0.77875077]])




```python
mat.dot(inv(mat))
```




    array([[ 1.00000000e+00, -1.38777878e-16,  1.33226763e-15,
            -1.77635684e-15, -1.77635684e-15],
           [-3.19189120e-16,  1.00000000e+00,  1.11022302e-16,
            -3.88578059e-16,  0.00000000e+00],
           [ 1.80411242e-16, -8.88178420e-16,  1.00000000e+00,
             2.77555756e-16, -1.77635684e-15],
           [ 2.22044605e-16,  1.11022302e-16,  1.28785871e-14,
             1.00000000e+00,  0.00000000e+00],
           [-6.66133815e-16, -4.44089210e-16, -2.66453526e-15,
             1.77635684e-15,  1.00000000e+00]])




```python
inv??
```


```python
qr??
```


```python
q, r = qr(mat)
```


```python
r
```




    array([[ -2.5740407 ,   7.0918124 ,  -0.3793516 ,   1.72425163,
              0.8316886 ],
           [  0.        ,  -3.75602521,  -8.77143628, -11.56247251,
              3.281949  ],
           [  0.        ,   0.        ,  -5.52420086,  -4.269043  ,
             -3.5670171 ],
           [  0.        ,   0.        ,   0.        ,  -0.90882307,
              2.33619203],
           [  0.        ,   0.        ,   0.        ,   0.        ,
              0.43234563]])




```python
samples = np.random.normal(size=(4, 4))
```


```python
samples
```




    array([[ 1.13897241,  2.05324113, -1.57185217,  1.01339013],
           [-1.18389273, -0.69698219, -0.17810784, -0.04006216],
           [ 0.7002058 , -2.30858076, -1.79351966,  0.19727974],
           [-0.23460936, -0.92062793,  1.06297183, -1.62325007]])




```python
from random import normalvariate
```


```python
N = 1000000
```


```python
%timeit samples = [normalvariate(0, 1) for _ in range(N)]
```

    562 ms ± 1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
%timeit np.random.normal(size=N)
```

    23.3 ms ± 47.5 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
np.random.seed(1234)
```


```python
rng = np.random.RandomState(1234)
```


```python
rng.randn(10)
```




    array([ 0.47143516, -1.19097569,  1.43270697, -0.3126519 , -0.72058873,
            0.88716294,  0.85958841, -0.6365235 ,  0.01569637, -2.24268495])




```python
import random
```


```python
position = 0
```


```python
walk = [position]
```


```python
steps = 1000
```


```python
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
```


```python
plt.plot(walk[:100])
```




    [<matplotlib.lines.Line2D at 0x7f781a9b6cd0>]




```python
plt.show()
```


    
![png](https://github.com/fendao/imgs/blob/main/usepy_4/output_246_0.png)
    



```python
random.randint??
```


```python
nsteps = 1000
```


```python
draws = np.random.randint(0, 2, size=nsteps)
```


```python
steps = np.where(draws > 0, 1, -1)
```


```python
walk = steps.cumsum()
```


```python
np.random.randint??
```


```python
walk.min()
```




    -9




```python
walk.max()
```




    60




```python
(np.abs(walk) >= 10).argmax()
```




    297




```python
nwalks = 5000
```


```python
nsteps = 1000
```


```python
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
```


```python
steps = np.where(draws > 0, -1, 1)
```


```python
walks = steps.cumsum(1)
```


```python
walks
```




    array([[ -1,  -2,  -3, ..., -46, -47, -46],
           [ -1,   0,  -1, ..., -40, -41, -42],
           [ -1,  -2,  -3, ...,  26,  27,  28],
           ...,
           [ -1,   0,  -1, ..., -64, -65, -66],
           [ -1,  -2,  -1, ...,  -2,  -1,   0],
           [  1,   2,   3, ..., -32, -33, -34]])




```python
walks.max()
```




    128




```python
walks.min()
```




    -122




```python
hits30 = (np.abs(walks) >= 30).any(1)
```


```python
hits30
```




    array([ True,  True,  True, ...,  True, False,  True])




```python
hits30.sum()
```




    3368




```python
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
```


```python
crossing_times.mean()
```




    509.99762470308787




```python
# 正态分布随机漫步
```


```python
steps = np.random.normal(loc=0, scale=0.25, size=(nwalks, nsteps))
```


```python
walks = steps.cumsum(1)
```


```python
walks
```




    array([[-2.04721967e-01, -2.38798674e-01, -4.86614938e-01, ...,
            -7.77977051e-01, -5.65471210e-01, -1.25528443e-01],
           [-9.06659494e-02, -4.32547941e-02,  5.54687233e-02, ...,
             1.29059312e+01,  1.31599860e+01,  1.29374340e+01],
           [ 9.42323969e-03,  4.36060080e-02, -3.79448524e-01, ...,
             3.46412368e+00,  3.34591495e+00,  2.63473663e+00],
           ...,
           [ 1.23011802e-01, -5.85454165e-02,  6.92772514e-02, ...,
             7.70007785e+00,  7.76554564e+00,  8.24474598e+00],
           [-2.05502686e-01, -4.09069667e-02, -7.22936453e-02, ...,
            -1.77989085e+00, -1.70157451e+00, -1.42603395e+00],
           [-2.30561721e-01,  5.86823124e-02,  1.43130384e-01, ...,
            -4.71520448e+00, -4.73661181e+00, -4.67317877e+00]])


