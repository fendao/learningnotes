# 训练与可视化
法一：  
    设置feature_names、class_names，tree.plot_tree()  
法二：  
    终端进入目录执行dot -o png转换，再export_graphviz()
# 树的预测
干啥的：预测数据属于哪个节点  
怎么干：  
    分类：根据长宽一直切，直到gini=0所有实例都同一类  
    回归：切出区域均值，让最多实例接近该均值（最小化MSE）  
    机制：CART算法，二叉树每次分两枝——Decision Stump看一个特征切一刀  
        每次分枝——b(x)最小化gini指数分枝  
    正则：切刀倾向切到Ein=0，低阶树数据太少必overfit  
实操：  
    训练模型：tree_clf.fit(X, y)  
    估计类概率：tree_clf.predict_proba()，predict()所属类  
    正则：参数max_depth控制深度，min_samples_split样本数、min_samples_leaf小错容忍参数  
        max_leaf_nodes、max_features，95%纯度  
优点：不需要特征缩放，  
缺点：边界附近的实例容易估错；全是横竖刀，对旋转敏感（主成分分析有助于调整方向）  
代码：绘制树的决策边界（切刀刀风）；网格搜索；手动随机森林实现


```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
```


```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
```


```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
```


```python
iris = load_iris()
X = iris.data[:, 2:] # 行全选，第3、4列长宽
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
```




    DecisionTreeClassifier(max_depth=2)




```python
a = np.linspace(0, 5, 10)
b = np.linspace(10, 20, 10)
a
```




    array([0.        , 0.55555556, 1.11111111, 1.66666667, 2.22222222,
           2.77777778, 3.33333333, 3.88888889, 4.44444444, 5.        ])




```python
b
```




    array([10.        , 11.11111111, 12.22222222, 13.33333333, 14.44444444,
           15.55555556, 16.66666667, 17.77777778, 18.88888889, 20.        ])




```python
x1, x2 = np.meshgrid(a, b)
x1.shape
```




    (10, 10)




```python
tree_clf.predict(X)
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2,
           2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])




```python
# matplotlib给tree可视化、将matplotlib图嵌入notebook
from sklearn import tree
%matplotlib inline
fn=['petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (2,2), dpi=300)
tree.plot_tree(tree_clf,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/06b6676e9a5b4b6198008f77a6f92e79.png#pic_center)

    



```python
import os
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```


```python
from sklearn.tree import export_graphviz
# 将模型导出为dot文件，终端进目录转png
tree.export_graphviz(tree_clf,
                     out_file="tree.dot",
                     feature_names = fn, 
                     class_names=cn,
                     filled = True)
```


```python
from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    # 0到7.5 100个点作
    x1s = np.linspace(axes[0], axes[1], 100)
    # 0到3 100个
    x2s = np.linspace(axes[2], axes[3], 100)
    # 生成网格矩阵，一对多生一万个点，将一万个点的xy坐标分别返回
    # 就像meshgrid生成的100*100矩阵，坐标同样100*100，只是由两个数的列表换作单数
    x1, x2 = np.meshgrid(x1s, x2s)
    # 分别拉成一维并组两列，成为10000*2数组
    X_new = np.c_[x1.ravel(), x2.ravel()]
    # 对10000*2数组数据预测分类，并重塑为100*100数组。。。以对应每个实例点的预测
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    # 画每刀的切割线？
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:    # not false才往下走
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contourf(x2, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    # 默认为True，所以默认描绘三类数据给三种颜色
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'yo', label='山鸢尾')
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'bs', label='多色')
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], 'g^', label='弗')
        plt.axis(axes)    # 默认参数刻度
    # 默认为True，所以默认xy轴标签。否则X1X2
    if iris:
        plt.xlabel('长', color='white', fontsize=14)
        plt.ylabel('宽', color='white', fontsize=14)
    else:
        plt.xlabel(r'$x_1$', color='white', fontsize=18)
        plt.ylabel(r'$x_2$', color='white', fontsize=18, rotation=0)
    # 图例False
    if legend:
        plt.legend(loc='lower right', fontsize=14)
```


```python
plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
# 自画决策边界
plt.plot([2.45, 2.45], [0, 3], 'k-', linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], 'k--', linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], 'k:', linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], 'k:', linewidth=2)
# 自定义深度。。
plt.text(1.4, 1.0, '深度0', fontsize=15)
plt.text(3.2, 1.8, '深度1', fontsize=13)
plt.text(4.05, 0.5, '(深度2)', fontsize=11)
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
save_fig('decision_tree_decision_boundaries_plot')
plt.show()
```

    Saving figure decision_tree_decision_boundaries_plot



    
![在这里插入图片描述](https://img-blog.csdnimg.cn/62a41c5203e94d939d5981df2f6143b9.png#pic_center)

    



```python
tree_clf.predict_proba([[5, 1.5]])
```




    array([[0.        , 0.90740741, 0.09259259]])




```python
tree_clf.predict([[5, 1.5]])
```




    array([1])




```python
tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)
tree_clf_tweaked.fit(X, y)
```




    DecisionTreeClassifier(max_depth=2, random_state=40)




```python
plt.figure(figsize=(8,4))
plot_decision_boundary(tree_clf_tweaked, X, y, legend=False)
plt.plot([0, 7.5], [0.8, 0.8], 'k-', linewidth=2)
plt.plot([0, 7.5], [1.75, 1.75], 'k--', linewidth=2)
plt.text(1.0, 0.9, 'depth=0', fontsize=15)
plt.text(1.0, 1.8, 'depth=1', fontsize=13)
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/1e2e063e9e2740aabc9a9223ec0e02a3.png#pic_center)

    



```python
from sklearn.datasets import make_moons
Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)
deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
deep_tree_clf1.fit(Xm, ym)
deep_tree_clf2.fit(Xm, ym)
fig, axes = plt.subplots(ncols=2, figsize=(10,4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.4, -1, 1.5], iris=False)
plt.title('非正则', fontsize=16, color='white')
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
plt.sca(axes[1])
plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.4, -1, 1.5], iris=False)
plt.title('min_samples_leaf={}'.format(deep_tree_clf2.min_samples_leaf), fontsize=14, color='white')
plt.ylabel('')
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/acf39999884b4c039970ae81a18f2a95.png#pic_center)

    



```python
angle = np.pi / 180 * 20
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
Xr = X.dot(rotation_matrix)
tree_clf_r = DecisionTreeClassifier(random_state=42)
tree_clf_r.fit(Xr, y)
plt.figure(figsize=(8,3))
plot_decision_boundary(tree_clf_r, Xr, y, axes=[0.5, 7.5, -1.0, 1], iris=False)
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/a90318e1df024c73a69bd6b95f4310ae.png#pic_center)

    



```python
from sklearn.tree import DecisionTreeRegressor
np.random.seed(42)
m=200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
```




    DecisionTreeRegressor(max_depth=2)




```python
# fn = ['长度', '宽度']
# cn = ['山', '多', '弗']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(2,2), dpi=300)
tree.plot_tree(
    tree_reg,
#     feature_names = fn,
#     class_names = cn,
    rounded=True,
    filled=True
)
```




    [Text(0.5, 0.8333333333333334, 'X[0] <= 0.197\nsquared_error = 0.098\nsamples = 200\nvalue = 0.354'),
     Text(0.25, 0.5, 'X[0] <= 0.092\nsquared_error = 0.038\nsamples = 44\nvalue = 0.689'),
     Text(0.125, 0.16666666666666666, 'squared_error = 0.018\nsamples = 20\nvalue = 0.854'),
     Text(0.375, 0.16666666666666666, 'squared_error = 0.013\nsamples = 24\nvalue = 0.552'),
     Text(0.75, 0.5, 'X[0] <= 0.772\nsquared_error = 0.074\nsamples = 156\nvalue = 0.259'),
     Text(0.625, 0.16666666666666666, 'squared_error = 0.015\nsamples = 110\nvalue = 0.111'),
     Text(0.875, 0.16666666666666666, 'squared_error = 0.036\nsamples = 46\nvalue = 0.615')]




    
![在这里插入图片描述](https://img-blog.csdnimg.cn/7984f4f7bc724aa78c96ffb955e80457.png#pic_center)

    



```python
from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)
# 只是预测各区域的均值
# 数据x轴500个均匀分布点、蓝点数据、红色各预测线
def plot_regression_predictions(tree_reg, X, y, axes=[0,1,-0.2,1], ylabel='$y$'):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel('$x_1$', fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, 'b.')
    plt.plot(x1, y_pred, 'r.-', linewidth=2, label=r'$\hat{y}$')
```


```python
fig,axes = plt.subplots(ncols=2, figsize=(10,4), sharey=True)
plt.sca(axes[0])
plot_regression_predictions(tree_reg1, X, y)
for split, style in ((0.1973, 'k-'), (0.0917, 'k--'), (0.7718, 'k--')):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
plt.text(0.21, 0.65, 'depth=0', fontsize=15)
plt.text(0.01, 0.2, 'depth=1', fontsize=13)
plt.text(0.65, 0.8, 'depth=1', fontsize=18)
plt.legend(loc='upper center', fontsize=18)
plt.title('max_depth=2', fontsize=14)
plt.sca(axes[1])
plot_regression_predictions(tree_reg2, X, y, ylabel=None)
for split, style in ((0.1973, 'k-'), (0.0917, 'k--'), (0.7718, 'k--')):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
for split in (0.0458, 0.1298, 0.2873, 0.9040):
    plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)    
plt.text(0.3, 0.5, 'depth=2', fontsize=13)
plt.title('max_depth=3', fontsize=14)
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/f91e11bd3c0e4f11afbda4b4e8db2d8b.png#pic_center)

    



```python
tree_reg1 = DecisionTreeRegressor(random_state=42)
tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)
x1 = np.linspace(0, 1, 500).reshape(-1, 1)
y_pred1 = tree_reg1.predict(x1)
y_pred2 = tree_reg2.predict(x1)
fig, axes = plt.subplots(ncols=2, figsize=(15,6), sharey=True)
plt.sca(axes[0])
plt.plot(X,y,'b.')
plt.plot(x1,y_pred1, 'r.-',linewidth=2, label=r'$\hat{y}$')
plt.axis([0, 1, -0.2, 1.1])
plt.xlabel('$x_1$', fontsize=18)
plt.ylabel('$y$', fontsize=18, rotation=0)
plt.legend(loc='upper center', fontsize=18)
plt.title('无正则', fontsize=14)
plt.sca(axes[1])
plt.plot(X,y,'b.')
plt.plot(x1,y_pred2,'r.-',linewidth=2,label=r'$\hat{y}$')
plt.axis([0,1,-0.2,1.1])
plt.xlabel('$x_1$',fontsize=18)
plt.title('min_samples_leaf={}'.format(tree_reg2.min_samples_leaf),fontsize=14)
plt.show()
```


    
![在这里插入图片描述](https://img-blog.csdnimg.cn/a9a02e075bce42feb65aa6111ae1c3ae.png#pic_center)

    



```python
# coding=utf-8
import matplotlib.pyplot as plt
a = {'贴片式': 76, '水洗式': 11, '睡眠免洗式': 6, '其他': 7}
b = [key for key in a]
c = [value for value in a.values()]
plt.figure(figsize=(10, 10), dpi=100)
plt.pie(c, labels=b, autopct="%1.2f%%", colors=['c', 'y', 'w','r'],textprops={'fontsize': 24}, labeldistance=1.05)
plt.legend(loc='upper right',fontsize=16)
plt.title("面膜品类占比", fontsize=24)
save_fig('piepie')
plt.show()
```

    Saving figure piepie



    
![在这里插入图片描述](https://img-blog.csdnimg.cn/1a7277775b59449e8cc7b44756c4857b.png#pic_center)

    



```python
data = [60, 390, 317, 58, 36, 140, 7, 2]
labels = ['西班牙/乌拉圭', '葡萄牙/巴西', '英国', '荷兰', '美国', '法国', '丹麦/波罗的海', '其他']
colors = ['lightsteelblue', 'bisque', 'bisque', 'lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue',]
plt.bar(range(len(data)), data, color=colors, alpha=0.6, width=1.0)
plt.xticks(range(len(data)), labels, rotation=45)
plt.ylabel('登船的奴隶总数')
plt.title('各国贩奴数汇总')
save_fig('bar1')
plt.show()
```

    Saving figure bar1



    
![在这里插入图片描述](https://img-blog.csdnimg.cn/1237e8223901495c9c6f560178dea16e.png#pic_center)

    



```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

```


```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
params = {'max_leaf_nodes': list(range(2,100)), 'min_samples_split': [2,3,4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)
```

    Fitting 3 folds for each of 294 candidates, totalling 882 fits





    GridSearchCV(cv=3, estimator=DecisionTreeClassifier(random_state=42),
                 param_grid={'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                31, ...],
                             'min_samples_split': [2, 3, 4]},
                 verbose=1)




```python
grid_search_cv.best_estimator_
```




    DecisionTreeClassifier(max_leaf_nodes=17, random_state=42)




```python
from sklearn.metrics import accuracy_score
y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)
```




    0.8695




```python
from sklearn.model_selection import ShuffleSplit
n_trees = 1000
n_instances = 100
mini_sets = []
rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train)-n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))
```


```python
from sklearn.base import clone
forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]
accuracy_scores = []
for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))
np.mean(accuracy_scores)
```




    0.8054499999999999




```python
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)
for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)
```


```python
from scipy.stats import mode
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
```


```python
accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))
```




    0.872

