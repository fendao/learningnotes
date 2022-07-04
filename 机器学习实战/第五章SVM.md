# 线性SVMclg
SVM对缩放超敏感，记得StandardScaler()  
## soft-margin
模型lenearSVC(C= , loss='hinge')  
或者SVC(kernel='linear', C= )  
或者SGDClassifier(loss='hinge', alpha=1/(m\*C))
# 非线性SVMclg
低维不可分，高维可分  
    Pipeline([  
        ('poly_features', PolynomialFeatures(degree=3)),  
        ('scaler', StandardScaler()),  
        ('svm', LinearSVC(C=10, loss='hinge'))  
    ])  
核技巧(转换的过程作内积)——似添加而未添加  
    Pipeline([  
        ('scaler', StandardScaler()),  
        ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))  
    ])  
RBF(高斯核函数)——添加相似特征  
     每个实例创为地标，并测每个实例与地标在z空间的相似度  
     更有可分离的机会，就是如果m太大，则特征太多，有点慢  
     Pipeline([  
         ('scaler', StandardScaler()),  
         ('svm_clf', SVC(kernel='rbf', gamma=5, C=0.001))  
     ])
# SVMreg？？？
LinearSVR(epsilon=1.5)  
SVR(kernel='poly', degree=2, C=100, epsilon=0.1)
# 工作原理？？？


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
```


```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
```


```python
from sklearn.svm import SVC
```


```python
iris = datasets.load_iris()
X = iris['data'][:, (2,3)]
y = (iris['target'] == 2).astype(np.float64)

```


```python
# 150个True
setosa_or_versicolor = (y==0) | (y==1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

```


```python
svm_clf = SVC(kernel='linear', C=float('inf'))
svm_clf.fit(X,y)
```


```python
x0 = np.linspace(0, 5.5, 200)
pred_1 = 5*x0 - 20
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5
# 绘制决策边界（二维）
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    # 得到权数W与截距b，以求向量长度WX+b，除以单位长度，得垂直投影
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    
    # 决策曲线（三维切割面、二维是分割线）方程满足w0x0+w1x1+b=0
    # 也即x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin,xmax,200)
    # 分隔实线方程/分隔平面方程，x1为分隔平面的点？
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]
    # 最大化边距，w[1]为向量模？
    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    # 支撑向量
    SVs = svm_clf.support_vectors_
    # 两边的支撑向量
    plt.scatter(SVs[:,0], SVs[:,1], s=180, facecolors='#FFAAAA')
    # 分隔实线与边距虚线
    plt.plot(x0, decision_boundary, 'k-', linewidth=2)
    plt.plot(x0, gutter_up, 'k--', linewidth=2)
    plt.plot(x0, gutter_down, 'k--', linewidth=2)
```


```python
svm_clf2.support_vectors_
```




    array([[4.9, 1.5],
           [4.8, 1.8],
           [4.9, 1.5],
           [5. , 1.7],
           [5.1, 1.6],
           [4.5, 1.7],
           [5. , 1.5],
           [4.9, 2. ],
           [6.1, 1.9],
           [5.1, 1.5],
           [5. , 1.9]])




```python
fig,axes = plt.subplots(ncols=2, figsize=(10,2.7),sharey=True)
plt.sca(axes[0])
plt.plot(x0, pred_1, 'g--', linewidth=2)
plt.plot(x0, pred_2, 'm-'. linewidth=2)
plt.plot(x0, pred_3, 'r-', linewidth=2)
plt.plot(X[:,0][y==1], X[:,1][y==1], 'bs', label='弗')
plt.plot(X[:,0][y==0], X[:,1][y==0], 'yo', label='山')
plt.xlabel('长'，fontsize=14)
plt.ylabel('宽'，fontsize=14)
plt.legend(loc='upper left', fontsize=14)
plt.axis([0, 5.5, 0, 2])

plt.sca(axes[1])
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'bs')
plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'yo')
plt.xlabel('长', fontsize=14)
plt.ylabel('宽', fontsize=14)
plt.axis([0, 5.5, 0, 2])
plt.show()
```


```python
Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
ys = np.array([0, 0, 1, 1])
svm_clf = SVC(kernel="linear", C=100)
svm_clf.fit(Xs, ys)

plt.figure(figsize=(9,2.7))
plt.subplot(121)
plt.plot(Xs[:, 0][ys==1], Xs[:, 1][ys==1], "bo")
plt.plot(Xs[:, 0][ys==0], Xs[:, 1][ys==0], "ms")
plot_svc_decision_boundary(svm_clf, 0, 6)
plt.xlabel("$x_0$", fontsize=20)
plt.ylabel("$x_1$    ", fontsize=20, rotation=0)
plt.title("Unscaled", fontsize=16)
plt.axis([0, 6, 0, 90])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(Xs)
svm_clf.fit(X_scaled, ys)

plt.subplot(122)
plt.plot(X_scaled[:, 0][ys==1], X_scaled[:, 1][ys==1], "bo")
plt.plot(X_scaled[:, 0][ys==0], X_scaled[:, 1][ys==0], "ms")
plot_svc_decision_boundary(svm_clf, -2, 2)
plt.xlabel("$x'_0$", fontsize=20)
plt.ylabel("$x'_1$  ", fontsize=20, rotation=0)
plt.title("Scaled", fontsize=16)
plt.axis([-2, 2, -2, 2])
plt.show()
```


```python
len(svm_clf2.coef_)
```




    1




```python
# 软边分类决策边界图
# 自定义两个个异常数据。。。
X_outliers = np.array([[3.4, 1.3], [3.2, 0.8]])
# 自定义非弗的异常值。。。
y_outliers = np.array([0,0])
# 151行2列，长度宽度数据，添加异常数据的长宽
Xo1 = np.concatenate([X, X_outliers[:1]], axis=0)
yo1 = np.concatenate([y, y_outliers[:1]], axis=0)
Xo2 = np.concatenate([X, X_outliers[1:]], axis=0)
yo2 = np.concatenate([y, y_outliers[1:]], axis=0)
svm_clf2 = SVC(kernel='linear', C=1)
svm_clf2.fit(Xo2, yo2)
fig,axes = plt.subplots(ncols=2, figsize=(10,2.7),sharey=True)
# 第一个散点子图
plt.sca(axes[0])
# 50个弗的长宽对应点-蓝方块
plt.plot(Xo1[:,0][yo1==1], Xo1[:,1][yo1==1], 'bs')
# 100个非弗数据与一个异常对应的长宽-黄实圈
plt.plot(Xo1[:,0][yo1==0], Xo1[:,1][yo1==0], 'yo')
plt.text(0.3, 1.0, '不可能', fontsize=20, color='r')
plt.xlabel('长', fontsize=14)
plt.ylabel('宽', fontsize=14)
# 第一个异常对应的箭头
plt.annotate('Outlier',
            xy=(X_outliers[0][0], X_outliers[0][1]),
             xytext=(2.5, 1.7),
            ha='center',
            arrowprops=dict(facecolor='black', shrink=0.1),
            fontsize=16)
plt.axis([0, 5.5, 0, 2])    # xy轴范围

plt.sca(axes[1])
# 包含第二个异常的长宽数据
plt.plot(Xo2[:,0][yo2==1], Xo2[:, 1][yo2==1], 'bs')
plt.plot(Xo2[:,0][yo2==0], Xo2[:, 1][yo2==0], 'yo')
plot_svc_decision_boundary(svm_clf2, 4, 5.5)
plt.xlabel('长', fontsize=14)
plt.annotate('异常',
            xy=(X_outliers[1][0], X_outliers[1][1]),
            xytext=(3.2, 0.88),
            ha='center',
            arrowprops=dict(facecolor='black', shrink=0.1),
            fontsize=16)
plt.axis([0, 5.5, 0, 2])
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/439adc2720d542d399583efcc37b8a60.png#pic_center)

    



```python
iris = datasets.load_iris()
svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(C=1, loss='hinge'))
])
svm_clf.fit(X, y)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('linear_svc', LinearSVC(C=1, loss='hinge'))])




```python
svm_clf.predict([[5.5, 1.7]])
```




    array([1.])




```python
# -4到4九个整数排一列
X1D = np.linspace(-4, 4, 9).reshape(-1,1)
# 上述与相应平方排两列
X2D = np.c_[X1D, X1D**2]
# 设置5个+1，4个-1
y = np.array([0,0,1,1,1,1,1,0,0])
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.grid(True, which='both')
# 设置水平线线坐标，c颜色， ls风格，lw宽度
plt.axhline(y=0, color='k')
#  索引前2后2x坐标，对应y4个0
plt.plot(X1D[:, 0][y==0], np.zeros(4), 'bs')
#  索引中间5个x坐标，对应y5个0
plt.plot(X1D[:, 0][y==1], np.zeros(5), 'g^')
# y轴不可见，并关闭y轴坐标刻度
plt.gca().get_yaxis().set_ticks([])
plt.xlabel(r'$x_1$', fontsize=20)
plt.axis([-4.5, 4.5, -0.2, 0.2]) # xy轴范围

plt.subplot(122)
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
# 设置垂直线线坐标，c颜色， ls风格，lw宽度
plt.axvline(x=0, color='k')
plt.plot(X2D[:, 0][y==0], X2D[:, 1][y==0], 'bs')
plt.plot(X2D[:, 0][y==1], X2D[:, 1][y==1], 'g^')
plt.xlabel(r'$x_1$', fontsize=20)
plt.ylabel(r'$x_2$', fontsize=20, rotation=0)
plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])
plt.plot([-4.5, 4.5], [6.5, 6.5], 'r--', linewidth=3)
plt.axis([-4.5, 4.5, -1, 17])
plt.subplots_adjust(right=1)
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/1ef7f31cb0bc414298d646c87de2c234.png#pic_center)

    



```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
```


```python
X,y = make_moons(n_samples=100, noise = 0.15)
polynomial_svm_clf = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(C=10, loss='hinge'))
])
polynomial_svm_clf.fit(X,y)
```




    Pipeline(steps=[('poly_features', PolynomialFeatures(degree=3)),
                    ('scaler', StandardScaler()),
                    ('svm_clf', LinearSVC(C=10, loss='hinge'))])




```python
def plot_dataset(X, y, axes):
    ###绘制数据集函数，X轴两列坐标数据，y轴标签索引数据，xy轴范围数据###
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'bs')
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'g^')
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r'$x_1$', fontsize=20)
    plt.ylabel(r'$x_2$', fontsize=20, rotation=0)

plot_dataset(X,y,[-1.5, 2.5, -1, 1.5])
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/dda5666143ab4aa3b9f33c180634683a.png#pic_center)

    



```python
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/e8f2254534bd4fca9e1d635c99801ad7.png#pic_center)

    



```python
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(X, y)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('svm_clf', SVC(C=5, coef0=1, kernel='poly'))])




```python
plot_dataset(X,y,[-1.5,2.5,-1,1.5])
plot_predictions(poly_kernel_svm_clf, [-1.5,2.5,-1,1.5])
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/445d1079413c409ab636fa7025fe8895.png#pic_center)

    



```python
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=10, coef0=100, C=5))
])
poly_kernel_svm_clf.fit(X, y)
plot_dataset(X,y,[-1.5,2.5,-1,1.5])
plot_predictions(poly_kernel_svm_clf, [-1.5,2.5,-1,1.5])
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/a4234f5777774a46bdb5eff796e6dcb2.png#pic_center)

    



```python
def gaussian_rbf(x, landmark, gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1) ** 2)
gamma = 0.3
# 注意x0s、X1D
x0s = np.linspace(-4.5, 4.5, 200).reshape(-1,1)
# 不需要reshape
x1s = gaussian_rbf(x0s, -2, gamma).reshape(-1,1)
x2s = gaussian_rbf(x0s, 1, gamma).reshape(-1,1)
# 九个实例点分别对两个地标的相似特征
xss = np.c_[gaussian_rbf(X1D, -2, gamma), gaussian_rbf(X1D, 1, gamma)]
yk = np.array([0,0,1,1,1,1,1,0,0])
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.scatter(x=[-2, 1], y=[0, 0], s=150, alpha=0.5 , c='red') # 注意s顺序
plt.plot(X1D[:, 0][yk==0], np.zeros(4), 'cs')
plt.plot(X1D[:, 0][yk==1], np.zeros(5), 'mo')
plt.plot(x0s, x1s, 'k--')
plt.plot(x0s, x2s, 'y--')
plt.xlabel(r'$x_1$', fontsize=20)
plt.ylabel(r'相似度', fontsize=14)
plt.gca().get_yaxis().set_ticks([0.00, 0.25, 0.50, 0.75, 1.00])
plt.annotate('X',
            xy=(X1D[3,0], 0), # 直接索引数据点，而非-1，0
             xytext=(-0.5, 0.23),
            ha='center',
            arrowprops=dict(facecolor='black', shrink=0.1),
            fontsize=16)
plt.text(-2, 0.9, r'$x_2$',ha='center', fontsize=20, color='m') # 突出大X小2
plt.text(1, 0.9, r'$x_3$', ha='center',fontsize=20, color='c')
plt.axis([-4.5, 4.5, -0.1, 1.10])

plt.subplot(122)
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
# 经典两类数据索引
plt.plot(xss[:,0][yk==0], xss[:,1][yk==0], 'cs')
plt.plot(xss[:,0][yk==1], xss[:,1][yk==1], 'mo')
plt.annotate(r'$\phi\left(\mathbf{x}\right)$', # 数学函数正则表达
            xy=(xss[3,0], xss[3,1]), # 索引定位数据点
             xytext=(0.65, 0.5),
            ha='center',
            arrowprops=dict(facecolor='black', shrink=0.1),
            fontsize=16)
plt.plot([-0.1, 1.1], [0.57, -0.1], 'k--', linewidth=0.3)
plt.xlabel(r'$x_2$', fontsize=20)
plt.ylabel(r'$x_3$', fontsize=14)
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.subplots_adjust(right=1)
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/0603c47f7fad464b82dcbbb5086ccd52.png#pic_center)

    



```python
y.shape
```




    (100,)




```python
rbf_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='rbf', gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit(X, y)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('svm_clf', SVC(C=0.001, gamma=5))])




```python
gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)
svm_clfs = []
for gamma, C in hyperparams:
    # 遍历不同参数
    rbf_kernel_svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='rbf', gamma=gamma, C=C))
    ])
    rbf_kernel_svm_clf.fit(X,y)
    # 列表追加已训练模型实例
    svm_clfs.append(rbf_kernel_svm_clf)
# 定义子图规格、下左为xy
fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(10.5, 7), sharex=True, sharey=True)
# enumerate返回索引与列表数据
for i, svm_clf in enumerate(svm_clfs):
    # //向负取整、%取余  
    plt.sca(axes[i // 2, i%2]) # 0,0, 0,1 1,0 1,1
    # 分割线与等高线
    plot_predictions(svm_clf, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.45, -1, 1.5]) # 卫星数据
    gamma, C = hyperparams[i]
    plt.title(r'$gamma = {}, C={}$'.format(gamma, C), fontsize=16) # 字：绘制等号自赋值形式
    if i in (0, 1):
        plt.xlabel('')
    if i in (1, 3):
        plt.ylabel('')
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/e40232928d3d47b29774c9779f917d06.png#pic_center)

    



```python
np.random.seed(42)
m=50
X = 2 * np.random.rand(m, 1) # 50行1列0,1分布
y = (4 + 3 * X + np.random.randn(m, 1)).ravel() # 变成1行50列
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5, random_state=42)
svm_reg.fit(X,y)
svm_reg1 = LinearSVR(epsilon=1.5, random_state=42)
svm_reg2 = LinearSVR(epsilon=0.5, random_state=42)
svm_reg1.fit(X,y)
svm_reg2.fit(X,y)

def find_support_vectors(svm_reg, X, y):
    y_pred = svm_reg.predict(X)
    off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon)
    return np.argwhere(off_margin)

svm_reg1.support_ = find_support_vectors(svm_reg1, X, y)
svm_reg2.support_ = find_support_vectors(svm_reg2, X, y)

eps_x1 = 1
eps_y_pred = svm_reg1.predict([[eps_x1]])

def plot_svm_regression(svm_reg, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, 'k-', linewidth=2, label=r'$\hat{y}$')
    plt.plot(x1s, y_pred + svm_reg.epsilon, 'k--')
    plt.plot(x1s, y_pred - svm_reg.epsilon, 'k--')
    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')
    plt.plot(X, y, 'bo')
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.axis(axes)
    
fig,axes = plt.subplots(ncols=2, figsize=(9,4), sharey=True)
plt.sca(axes[0])
plot_svm_regression(svm_reg1, X, y, [0,2,3,11])
plt.title(r'$\epsilon = {}$'.format(svm_reg1.epsilon), fontsize=18)
plt.ylabel(r'$y$', fontsize=18, rotation=0)
plt.annotate(
        '', xy=(eps_x1, eps_y_pred), xycoords='data',
        xytext=(eps_x1, eps_y_pred - svm_reg1.epsilon),
        textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 1.5}
    )
plt.text(0.91, 5.6, r'$\epsilon$', fontsize=20)
plt.sca(axes[1])
plot_svm_regression(svm_reg2, X, y, [0,2,3,11])
plt.title(r'$\epsilon = {}'.format(svm_reg2.epsilon), fontsize=18)
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/51c996a7425244f69c0fed0f40ffdbc0.png#pic_center)

    



```python
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1) - 1
y = (0.2 + 0.1 * X + 0.5 * X**2 + np.random.randn(m, 1)/10).ravel()
from sklearn.svm import SVR
svm_poly_reg = SVR(kernel='poly', degree=2, C=100, epsilon=0.1, gamma='scale')
svm_poly_reg.fit(X, y)
```




    SVR(C=100, degree=2, kernel='poly')




```python
from sklearn.svm import SVR
svm_poly_reg1 = SVR(kernel='poly', degree=2, C=100, epsilon=0.1, gamma='scale')
svm_poly_reg2 = SVR(kernel='poly', degree=2, C=0.01, epsilon=0.1, gamma='scale')
svm_poly_reg1.fit(X, y)
svm_poly_reg2.fit(X, y)
```




    SVR(C=0.01, degree=2, kernel='poly')




```python
fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
plt.sca(axes[0])
plot_svm_regression(svm_poly_reg1, X, y, [-1, 1, 0, 1])
plt.title(r'$degree={}, C={}, \epsilon = {}$'.format(svm_poly_reg1.degree, svm_poly_reg1.C, svm_poly_reg1.epsilon), fontsize=18)
plt.ylabel(r'$y$', fontsize=18, rotation=0)
plt.sca(axes[1])
plot_svm_regression(svm_poly_reg2, X, y, [-1,1,0,1])
plt.title(r'$degree={}, C={}, \epsilon = {}$'.format(svm_poly_reg2.degree, svm_poly_reg2.C, svm_poly_reg2.epsilon), fontsize=18)
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/9f33079547aa4617bab10dcc547acfd6.png#pic_center)

    



```python
# 作业
from sklearn import datasets

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]
```


```python
from sklearn.linear_model import SGDClassifier
C = 5
alpha = 1 / (C * len(X))
lin_svc = LinearSVC(C=5,loss='hinge', random_state=42)
svc_clf = SVC(kernel='linear',degree=2, gamma=5, C=5)
sgd_clf = SGDClassifier(loss='hinge', learning_rate='constant', eta0=0.001, alpha=alpha,
                       max_iter=1000, tol=1e-3, random_state=42)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
lin_svc.fit(scaled_X, y)
svc_clf.fit(scaled_X, y)
sgd_clf.fit(scaled_X, y)

print('LSVC:', lin_svc.intercept_, lin_svc.coef_)
print('SVC:', svc_clf.intercept_, svc_clf.coef_)
print('SGDC(alpha={:.5f}):'.format(sgd_clf.alpha), sgd_clf.intercept_, sgd_clf.coef_)
```

    LSVC: [0.28475098] [[1.05364854 1.09903804]]
    SVC: [0.31896852] [[1.1203284  1.02625193]]
    SGDC(alpha=0.00200): [0.117] [[0.77714169 0.72981762]]



```python
w1 = -lin_svc.coef_[0, 0] / lin_svc.coef_[0, 1]
b1 = -lin_svc.intercept_[0] / lin_svc.coef_[0, 1]
w2 = -svc_clf.coef_[0, 0] / svc_clf.coef_[0, 1]
b2 = -svc_clf.intercept_[0] / svc_clf.coef_[0, 1]
w3 = -sgd_clf.coef_[0, 0] / sgd_clf.coef_[0, 1]
b3 = -sgd_clf.intercept_[0] / sgd_clf.coef_[0, 1]

line1 = scaler.inverse_transform([[-10, -10*w1+b1], [10, 10*w1+b1]])
line2 = scaler.inverse_transform([[-10, -10*w2+b2], [10, 10*w2+b2]])
line3 = scaler.inverse_transform([[-10, -10*w3+b3], [10, 10*w3+b3]])
```


```python
line1,w1
```




    (array([[-11.56182566,   6.03127548],
            [ 17.28382566,  -4.7506598 ]]),
     -0.9587007015533304)




```python
plt.figure(figsize=(11, 4))
plt.plot(X[:,0][y==0], X[:,1][y==0], 'ro')
plt.plot(X[:,0][y==1], X[:,1][y==1], 'g^')
plt.plot(line1[:,0], line1[:,1], 'k-', label='LinearSVC')
plt.plot(line2[:,0], line2[:,1], 'y--', linewidth=2, label='SVC')
plt.plot(line3[:,0], line3[:,1], 'c:', label='SGDClassifier')
plt.xlabel('长度', fontsize=14)
plt.ylabel('宽度', fontsize=14)
plt.legend(loc='upper center')
plt.axis([0.5, 5.5, 0.0, 2.0])
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/798be0f835ab49e7821a5c38b98928da.png#pic_center)

    



```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']
X_train ,X_test= X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
```


```python
lin_svc = LinearSVC(random_state = 42)
lin_svc.fit(X_train, y_train)
```




    LinearSVC(random_state=42)




```python
from sklearn.metrics import accuracy_score
y_pred = lin_svc.predict(X_train)
accuracy_score(y_train, y_pred)
```




    0.8348666666666666




```python
std_scaler = StandardScaler()
scaled_X_train = std_scaler.fit_transform(X_train.astype(np.float32))
scaled_X_test = std_scaler.fit_transform(X_test.astype(np.float32))
```


```python
lin_clf = LinearSVC(random_state=42)
lin_clf.fit(scaled_X_train, y_train)
```






    LinearSVC(random_state=42)




```python
y_pred = lin_clf.predict(scaled_X_train)
accuracy_score(y_train, y_pred)
```




    0.9214




```python
svm_clf = SVC(gamma='scale')
svm_clf.fit(scaled_X_train[:10000], y_train[:10000])
y_pred = svm_clf.predict(scaled_X_train)
accuracy_score(y_train, y_pred)
```




    0.9455333333333333




```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
param_distributions = {'gamma': reciprocal(0.001, 0.1), 'C': uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2, cv=3)
rnd_search_cv.fit(scaled_X_train[:1000], y_train[:1000])
```

    Fitting 3 folds for each of 10 candidates, totalling 30 fits
    [CV] END ....C=10.539285770025874, gamma=0.06756608892518738; total time=   0.2s
    [CV] END ....C=10.539285770025874, gamma=0.06756608892518738; total time=   0.2s
    [CV] END ....C=10.539285770025874, gamma=0.06756608892518738; total time=   0.2s
    [CV] END ...C=4.701587002554444, gamma=0.0010737748632897968; total time=   0.2s
    [CV] END ...C=4.701587002554444, gamma=0.0010737748632897968; total time=   0.2s
    [CV] END ...C=4.701587002554444, gamma=0.0010737748632897968; total time=   0.2s
    [CV] END ...C=10.283185625877254, gamma=0.007184032636581678; total time=   0.2s
    [CV] END ...C=10.283185625877254, gamma=0.007184032636581678; total time=   0.2s
    [CV] END ...C=10.283185625877254, gamma=0.007184032636581678; total time=   0.2s
    [CV] END ....C=10.666548190436696, gamma=0.08457460033731479; total time=   0.2s
    [CV] END ....C=10.666548190436696, gamma=0.08457460033731479; total time=   0.2s
    [CV] END ....C=10.666548190436696, gamma=0.08457460033731479; total time=   0.2s
    [CV] END .....C=9.530094554673601, gamma=0.00388059021391932; total time=   0.2s
    [CV] END .....C=9.530094554673601, gamma=0.00388059021391932; total time=   0.2s
    [CV] END .....C=9.530094554673601, gamma=0.00388059021391932; total time=   0.2s
    [CV] END .....C=4.850977286019253, gamma=0.05038176096019996; total time=   0.2s
    [CV] END .....C=4.850977286019253, gamma=0.05038176096019996; total time=   0.2s
    [CV] END .....C=4.850977286019253, gamma=0.05038176096019996; total time=   0.2s
    [CV] END ....C=4.169220051562776, gamma=0.002182657003890117; total time=   0.2s
    [CV] END ....C=4.169220051562776, gamma=0.002182657003890117; total time=   0.2s
    [CV] END ....C=4.169220051562776, gamma=0.002182657003890117; total time=   0.2s
    [CV] END ......C=6.568012624583502, gamma=0.0745262979291264; total time=   0.2s
    [CV] END ......C=6.568012624583502, gamma=0.0745262979291264; total time=   0.2s
    [CV] END ......C=6.568012624583502, gamma=0.0745262979291264; total time=   0.2s
    [CV] END .....C=7.96029796674973, gamma=0.013807731717915692; total time=   0.2s
    [CV] END .....C=7.96029796674973, gamma=0.013807731717915692; total time=   0.2s
    [CV] END .....C=7.96029796674973, gamma=0.013807731717915692; total time=   0.2s
    [CV] END ....C=1.9717649377076854, gamma=0.01698300171255907; total time=   0.2s
    [CV] END ....C=1.9717649377076854, gamma=0.01698300171255907; total time=   0.2s
    [CV] END ....C=1.9717649377076854, gamma=0.01698300171255907; total time=   0.2s





    RandomizedSearchCV(cv=3, estimator=SVC(),
                       param_distributions={'C': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7fc549572d10>,
                                            'gamma': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7fc54943af50>},
                       verbose=2)




```python
rnd_search_cv.best_estimator_,
```




    (SVC(C=4.701587002554444, gamma=0.0010737748632897968),)




```python
rnd_search_cv.best_score_
```




    0.8639957322592053




```python
rnd_search_cv.best_estimator_.fit(scaled_X_train, y_train)
```




    SVC(C=4.701587002554444, gamma=0.0010737748632897968)




```python
y_pred = rnd_search_cv.best_estimator_.predict(scaled_X_train)
accuracy_score(y_train, y_pred)
```




    0.99565




```python
y_pred = rnd_search_cv.best_estimator_.predict(scaled_X_test)
accuracy_score(y_test, y_pred)
```




    0.9717




```python
# 作业10
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X = housing['data']
y = housing['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
std_clf = StandardScaler()
scaled_X_train = std_clf.fit_transform(X_train)
```


```python
from sklearn.svm import LinearSVR
lin_svr = LinearSVR(random_state=42)
lin_svr.fit(scaled_X_train, y_train)
```

    /Users/zhangdi/opt/miniconda3/lib/python3.7/site-packages/sklearn/svm/_base.py:1208: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      ConvergenceWarning,





    LinearSVR(random_state=42)




```python
from sklearn.metrics import mean_squared_error
y_pred = lin_svr.predict(scaled_X_train)
mse = mean_squared_error(y_train, y_pred)
```


```python
np.sqrt(mse)
```




    0.9819256687727764




```python
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
```


```python
param_distributions = {'gamma': reciprocal(0.001, 0.1), 'C': uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(SVR(), param_distributions, n_iter=10, verbose=2, cv=3, random_state=42)
rnd_search_cv.fit(scaled_X_train, y_train)
```

    Fitting 3 folds for each of 10 candidates, totalling 30 fits
    [CV] END .....C=4.745401188473625, gamma=0.07969454818643928; total time=   7.7s
    [CV] END .....C=4.745401188473625, gamma=0.07969454818643928; total time=   7.6s
    [CV] END .....C=4.745401188473625, gamma=0.07969454818643928; total time=   7.7s
    [CV] END .....C=8.31993941811405, gamma=0.015751320499779724; total time=   7.6s
    [CV] END .....C=8.31993941811405, gamma=0.015751320499779724; total time=   7.6s
    [CV] END .....C=8.31993941811405, gamma=0.015751320499779724; total time=   7.7s
    [CV] END ....C=2.560186404424365, gamma=0.002051110418843397; total time=   7.6s
    [CV] END ....C=2.560186404424365, gamma=0.002051110418843397; total time=   7.6s
    [CV] END ....C=2.560186404424365, gamma=0.002051110418843397; total time=   7.8s
    [CV] END ....C=1.5808361216819946, gamma=0.05399484409787431; total time=   7.6s
    [CV] END ....C=1.5808361216819946, gamma=0.05399484409787431; total time=   7.5s
    [CV] END ....C=1.5808361216819946, gamma=0.05399484409787431; total time=   7.5s
    [CV] END ....C=7.011150117432088, gamma=0.026070247583707663; total time=   7.7s
    [CV] END ....C=7.011150117432088, gamma=0.026070247583707663; total time=   7.8s
    [CV] END ....C=7.011150117432088, gamma=0.026070247583707663; total time=   7.8s
    [CV] END .....C=1.2058449429580245, gamma=0.0870602087830485; total time=   7.3s
    [CV] END .....C=1.2058449429580245, gamma=0.0870602087830485; total time=   7.4s
    [CV] END .....C=1.2058449429580245, gamma=0.0870602087830485; total time=   7.4s
    [CV] END ...C=9.324426408004218, gamma=0.0026587543983272693; total time=   7.8s
    [CV] END ...C=9.324426408004218, gamma=0.0026587543983272693; total time=   7.7s
    [CV] END ...C=9.324426408004218, gamma=0.0026587543983272693; total time=   7.7s
    [CV] END ...C=2.818249672071006, gamma=0.0023270677083837795; total time=   7.7s
    [CV] END ...C=2.818249672071006, gamma=0.0023270677083837795; total time=   7.7s
    [CV] END ...C=2.818249672071006, gamma=0.0023270677083837795; total time=   7.7s
    [CV] END ....C=4.042422429595377, gamma=0.011207606211860567; total time=   7.6s
    [CV] END ....C=4.042422429595377, gamma=0.011207606211860567; total time=   7.6s
    [CV] END ....C=4.042422429595377, gamma=0.011207606211860567; total time=   7.7s
    [CV] END ....C=5.319450186421157, gamma=0.003823475224675185; total time=   7.8s
    [CV] END ....C=5.319450186421157, gamma=0.003823475224675185; total time=   7.8s
    [CV] END ....C=5.319450186421157, gamma=0.003823475224675185; total time=   7.7s





    RandomizedSearchCV(cv=3, estimator=SVR(),
                       param_distributions={'C': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7fc55ce9dd10>,
                                            'gamma': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7fc55cbf6490>},
                       random_state=42, verbose=2)




```python
rnd_search_cv.best_estimator_
```




    SVR(C=4.745401188473625, gamma=0.07969454818643928)




```python
rnd_search_cv.best_score_
```




    0.7382549678673654




```python
y_pred = rnd_search_cv.best_estimator_.predict(scaled_X_train)
mse = mean_squared_error(y_train, y_pred)
np.sqrt(mse)
```




    0.5727524770785359




```python
scaled_X_test = std_clf.transform(X_test)
```


```python
y_pred = rnd_search_cv.best_estimator_.predict(scaled_X_test)
mse = mean_squared_error(y_test, y_pred)
np.sqrt(mse)
```




    1.1066673077043268

