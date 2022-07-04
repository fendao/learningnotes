# 投票分类器(单纯投票：累加看正负，给g权重α，给g设条件函数
集成：大量且多种预测可超1/2的弱学习器，可实现出一强学习器  
软投票实现：  
    VotingClassifier(estimators=[('lr', log_clf),('rf', rnd_clf),('svc', svm_clf)],voting='soft')  
    看疗效accuracy_score(y_test, y_pred)  
集成特点：  
    主要啥都能干。欠拟合就特征转换。过拟合就作正则  
代码：随机硬币一万次几率图
# bagging and pasting
bagging:bootstrap有放回抽样聚合；pasting:不放回抽样  
实现：  
    BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,max_samples=100, bootstrap=True, n_jobs=-1)  
    bootstrap=False即pasting  
    oob_score=True自动oob评估  
    bootstrap_features=True或max_features<1.0随机子空间，即对特征抽样（降维  
集成的优点：  
    Eout低泛化性能好  
    会有37%包外实例可像test数据作出验证  
集成的缺点：  
    跟单个预测器的Ein相近  
代码：  
    绘制集成的决策边界
# 随机森林
干嘛的：聚合决策树  
怎么干：bagging+随机子空间CART，每次b(x)分枝的时候随机子空间一次  
实现：  
    BaggingClassifier(DecisionTreeClassifier(max_features='sqrt',max_leaf_nodes=16),  
    n_estimators=500,random_state=42)  
    rnd_clf.feature_importances_输出各特征的重要性  
随机森林特点：  
    在随机子空间搜索特征是在拿高偏差换低方差  
    能测每个特征的重要性：以每个节点相关样本数量作权重，计算所有包含该特征所减少不纯度的均值？  
    玄学性能，不行就多加点树，实在不行就种子改幸运数
## 1.极端随机树
干嘛的：分枝时不找最佳特征，随便上特征  
实现：ExtraTreesClassifier()
# Adaboost
干嘛的：盯住弱鸡的错例，再组合弱鸡就会变强  
怎么干：  
    弱鸡的自我修炼宝典——如何得屡次改错的g  
    对于弱鸡的前提：错误率小于1/2，正确量比错误量>1。只要每轮结束能放大错例缩小正例，一样有不同g  
    使t+1轮永有（错例）/（错+正例） = 1/2，若是错例，就让该例占比\*正例的数量，若正例，则反之  
    这样，每个实例都有不同，绝选不出从前的g。最后α作为权重对所有g聚合  
实现：  
    AdaBoostClassifier(  
        DecisionTreeClassifier(max_depth=1), n_estimators=200,  
        algorithm='SAMME.R', learning_rate=0.5, random_state=42)  
优点：  
    理论上偏差可直达0，由VCBound知且方差还足够小  
缺点：  
    在前一个预测器更新完所有实例的权重之后，才会开始新预测器。无法并行，扩展性能差
# 梯度提升
特点：同样每个预测器改正前序。但不通过调整实例权重，而是拟合之前的残差  
GBDT算法怎么干：  
    使用决策树回归算法对假说h(x)与残差(y-S)的距离作最小化动作，得到最佳h作g  
    使用线性回归最小化残差(y-S)与假说g(x)的距离，得最大步长作g的权重α  
手动实现：  
    训练基础决策树回归模型，由y-基础预测数据 的残差y2  
    对实例X与残差y2训练2号回归器，得残差y3 = y2 - y_pred  
    对实例X于残差y3训练3号回归器  
    集成预测：将3个回归器对新实例的预测数据相加  
Scikit-Learn：  
    GradientBoostingRegressor(  
        max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)  
    学习率参数learning_rate控制树权重、warm_start=True保留树以增量训练  
    subsample=0.5，每棵树随机分配一半实例训练，增强泛化性能  
提前停止法找最佳树量：  
    验证误差：errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]  
代码：绘制不同树下的验证误差
# GBDT的扩展——XGBoost
# 堆叠法stacking
干嘛的：用函数聚合各g有点low，用模型试试  
怎么干：第一子集训练预测器，预测器对第二子集预测，n种预测结果作为n维子集训练混合器  
手动实现：关键是将各预测器的预测数据封装成新训练集还是将训练集分成三份？？？


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
```


```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
```


```python
import os
PROJECT_ROOT_DIR = '.'
CHAPTER_ID = 'ensembles'
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, 'images', CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id+'.'+fig_extension)
    print('saving figure', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```


```python
# 正面朝上概率，小于0.51表示正面
heads_proba = 0.51
# 10000行10列[0,1)均匀分布数组，统计出比0.51小的True or False数组
coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
# 纵向对True累加统计该列有多少True，分别除以一到10000排成一列，表示10000次中各次True出现的概率
cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)

plt.figure(figsize=(8, 3.5))
plt.plot(cumulative_heads_ratio)
plt.plot([0,10000], [0.51,0.51], 'k--', linewidth=2, label='51%') # 一条0.51虚线
plt.plot([0,10000], [0.5, 0.5], 'k-', label='50%') # 一条0.5直线
plt.xlabel('投掷次数')
# plt.ylabel('正面出现几率')
plt.ylabel('正\n面\n出\n现\n几\n率', labelpad=10, rotation=0, position=(-10,0.3))
# plt.ylabel('正面出现几率', fontproperties='Arial Unicode MS',rotation='horizontal', verticalalignment='bottom', horizontalalignment='left')
# plt.ylabel('s\ni\nn',fontsize=10,linespacing=4,position=(-10,0.3),rotation=0)
plt.legend(loc='lower right')
plt.axis([0,10000,0.42,0.58])
plt.show()
```


![](https://github.com/fendao/imgs/blob/main/ml第七章/output_4_0.png)
    



```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```


```python
log_clf = LogisticRegression(solver='lbfgs', random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma='scale', random_state=42)
```


```python
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')
voting_clf.fit(X_train, y_train)
```




    VotingClassifier(estimators=[('lr', LogisticRegression(random_state=42)),
                                 ('rf', RandomForestClassifier(random_state=42)),
                                 ('svc', SVC(random_state=42))])




```python
from sklearn.metrics import accuracy_score
log_clf.fit(X_train, y_train)
y_pred = log_clf.predict(X_test)
print(log_clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

    LogisticRegression 0.864



```python
rnd_clf.fit(X_train, y_train)
y_pred = rnd_clf.predict(X_test)
print(rnd_clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

    RandomForestClassifier 0.896



```python
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
print(svm_clf.__class__.__name__, accuracy_score(y_test, y_pred))
```


    ---------------------------------------------------------------------------




```python
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
print(voting_clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

    VotingClassifier 0.912



```python
log_clf = LogisticRegression(solver='lbfgs', random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma='scale', probability=True, random_state=42)
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf),('rf', rnd_clf),('svc', svm_clf)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
```


```python
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
print(svm_clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

    SVC 0.896



```python
# voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
print(voting_clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

    VotingClassifier 0.92



```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1
)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
```


```python
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
```

    0.912



```python
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))
```

    0.856



```python
from matplotlib.colors import ListedColormap
def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.5, contour=True):
    # x轴y轴之间生成数据
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    # 相应成对成双，返回左x右y坐标
    x1, x2 = np.meshgrid(x1s, x2s)
    # x与y再配对，组成数据坐标以便于预测
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    # 颜色习惯
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    # 填充等高线之间的空隙颜色，呈现出区域的分划状
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    # 对单xy坐标便于生成数据点，绘制预测线（决策边界）、颜色风格
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    # 经典绘制两列数据
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'yo', alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'bs', alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18, rotation=0)
```


```python
fig, axes = plt.subplots(ncols=2, figsize=(10,4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(tree_clf, X, y)
plt.title('Decision Tree', fontsize=14)
plt.sca(axes[1])
plot_decision_boundary(bag_clf, X, y)
plt.title('Bagging Decision Tree', fontsize=14)
plt.ylabel('')
plt.show()
```


![](https://github.com/fendao/imgs/blob/main/ml第七章/output_19_0.png)
    



```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    bootstrap=True, oob_score=True, random_state=40
)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_
```




    0.8986666666666666




```python
from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)
```




    0.912



# 随机森林


```python
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```


```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features='sqrt', max_leaf_nodes=16),
    n_estimators=500, random_state=42
)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
```


```python
np.sum(y_pred==y_pred_rf) / len(y_pred)
```




    1.0




```python
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris['data'], iris['target'])
for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
    print(name, score)
```

    sepal length (cm) 0.11249225099876375
    sepal width (cm) 0.02311928828251033
    petal length (cm) 0.4410304643639577
    petal width (cm) 0.4233579963547682



```python
rnd_clf.feature_importances_
```




    array([0.11249225, 0.02311929, 0.44103046, 0.423358  ])




```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf.fit(mnist['data'], mnist['target'])
```




    RandomForestClassifier(random_state=42)




```python
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.hot,
              interpolation='nearest')
    plt.axis('off')
```


```python
plot_digit(rnd_clf.feature_importances_)
cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['不重要', '很重要'])
plt.show()
```



![](https://github.com/fendao/imgs/blob/main/ml第七章/output_30_0.png)
    


# Adaboost


```python
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm='SAMME.R', learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
```




    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
                       learning_rate=0.5, n_estimators=200, random_state=42)




```python
axes = [-1.5, 2.45, -1, 1.5]
x1s = np.linspace(axes[0], axes[1], 100)
x2s = np.linspace(axes[2], axes[3], 100)
x1, x2 = np.meshgrid(x1s, x2s)
X_new = np.c_[x1.ravel(), x2.ravel()]
y_pred = ada_clf.predict(X_new).reshape(x1.shape)
custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'yo', alpha=0.5)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'bs', alpha=0.5)
plt.axis(axes)
plt.xlabel(r'$x_1$', fontsize=18)
plt.ylabel(r'$x_2$', fontsize=18, rotation=0)
plt.show()
```


    
![](https://github.com/fendao/imgs/blob/main/ml第七章/output_33_0.png)
    



```python
m = len(X_train)
fix, axes = plt.subplots(ncols=2, figsize=(10,4), sharey=True)
for subplot, learning_rate in ((0,1),(1, 0.5)):
    sample_weights = np.ones(m) / m
    plt.sca(axes[subplot])
    for i in range(5):
        svm_clf = SVC(kernel='rbf', C=0.2, gamma=0.6, random_state=42)
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights * m)
        y_pred = svm_clf.predict(X_train)
        r = sample_weights[y_pred != y_train].sum() / sample_weights.sum()
        alpha = learning_rate * np.log((1-r) / r)
        sample_weights[y_pred != y_train] *= np.exp(alpha)
        sample_weights /= sample_weights.sum()
        plot_decision_boundary(svm_clf, X, y, alpha=0.2)
        plt.title('learning_rate = {}'.format(learning_rate), fontsize=16)
    if subplot==0:
        plt.text(-0.75, -0.95, '1', fontsize=14)
        plt.text(-1.05, -0.95, '2', fontsize=14)
        plt.text(1.0, -0.95, '3', fontsize=14)
        plt.text(-1.45, -0.5, '4', fontsize=14)
        plt.text(1.36, -0.95, '5', fontsize=14)
    else:
        plt.ylabel('')
plt.show()
```


    
![](https://github.com/fendao/imgs/blob/main/ml第七章/output_34_0.png)
    



```python
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm='SAMME.R', learning_rate=0.5)
ada_clf.fit(X_train, y_train)
```




    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
                       learning_rate=0.5, n_estimators=200)



# 梯度提升


```python
np.random.seed(42)
# (-0.5_0.5)100行1列均匀分布数组
X = np.random.rand(100, 1) -0.5
# 二次项以及标正噪音
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)
```


```python
from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
# 根据y适用决策树回归训练X
tree_reg1.fit(X, y)
```




    DecisionTreeRegressor(max_depth=2, random_state=42)




```python
# 标签再次减小
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
# 适用小标签再次训练数据
tree_reg2.fit(X, y2)
```




    DecisionTreeRegressor(max_depth=2)




```python
# 利用预测数据再再次减小
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)
```




    DecisionTreeRegressor(max_depth=2)




```python
X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
```


```python
def plot_predictions(regressors, X, y, axes, label=None, style='r-', data_style='b.', data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1,1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=None)
    if label or data_label:
        plt.legend(loc='upper center', fontsize=16)
        plt.axis(axes)
```


```python

```




    array([0.66091233, 0.66091233, 0.66091233, 0.66091233, 0.66091233,
           0.66091233, 0.66091233, 0.66091233, 0.66091233, 0.66091233,
           0.66091233, 0.66091233, 0.66091233, 0.66091233, 0.66091233,
           0.66091233, 0.66091233, 0.66091233, 0.66091233, 0.66091233,
           0.66091233, 0.66091233, 0.66091233, 0.66091233, 0.66091233,
           0.66091233, 0.66091233, 0.66091233, 0.66091233, 0.66091233,
           0.66091233, 0.66091233, 0.66091233, 0.66091233, 0.66091233,
           0.48779682, 0.48779682, 0.48779682, 0.48779682, 0.48779682,
           0.48779682, 0.48779682, 0.48779682, 0.48779682, 0.48779682,
           0.48779682, 0.48779682, 0.48779682, 0.48779682, 0.48779682,
           0.48779682, 0.48779682, 0.48779682, 0.48779682, 0.48779682,
           0.48779682, 0.48779682, 0.48779682, 0.48779682, 0.48779682,
           0.48779682, 0.48779682, 0.48779682, 0.48779682, 0.48779682,
           0.48779682, 0.48779682, 0.48779682, 0.48779682, 0.48779682,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.12356613, 0.12356613, 0.12356613, 0.12356613,
           0.12356613, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846,
           0.52856846, 0.52856846, 0.52856846, 0.52856846, 0.52856846])




```python
axes = [-0.5, 0.5, -0.1, 0.8]
plt.figure(figsize=(11,11))
plt.subplot(321)
x1 = np.linspace(axes[0], axes[1], 500)
# 循环才能500次预测数据？？？
y_pred1 = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in [tree_reg1])
plt.plot(X[:, 0], y, 'b.', label='Training Set')
plt.plot(x1, y_pred1, 'g-', linewidth=2, label='$h_1(x_1)$')
plt.legend(loc='upper center', fontsize=16)
plt.axis(axes)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.title('残差与树预测', fontsize=16)
plt.subplot(322)
plt.plot(X[:, 0], y, 'y.', label='Training Set')
plt.plot('r-', linewidth=2, label='$h(x_1)=h_1(x_1)$')
plt.legend(loc='upper center', fontsize=16)
plt.axis(axes)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.title('集成预测', fontsize=16)

plt.subplot(323)
y_pred2 = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in [tree_reg2])
plt.plot(X[:, 0], y2, 'k+', label='残差')
plt.plot(x1, y_pred2, 'r-', linewidth=2, label='$h_2(x_1)$')
plt.legend(loc='upper center', fontsize=16)
plt.axis([-0.5, 0.5, -0.5, 0.5])
plt.ylabel('$y - h_1(x_1)$', fontsize=16)
plt.subplot(324)
y_pred5 = sum(a.predict(x1.reshape(-1, 1)) for a in [tree_reg1, tree_reg2])
plt.plot(X[:, 0], y, 'y.',)
plt.plot(x1, y_pred5, 'r-', linewidth=2, label='$h(x_1)=h_1(x_1)+h_2(x_1)$')
plt.axis(axes)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.legend(loc='upper center', fontsize=16)

plt.subplot(325)
y_pred3 = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in [tree_reg3])
plt.plot(X[:, 0], y3, 'k+')
plt.plot(x1, y_pred3, 'r-', linewidth=2, label='$h_3(x_1)$')
plt.axis([-0.5, 0.5, -0.5, 0.5])
plt.ylabel('$y-h_1(x_1)-h_2(x_1)$', fontsize=16)
plt.xlabel('$x_1$', fontsize=16)
plt.legend(loc='upper center', fontsize=16)
plt.subplot(326)
y_pred4 = sum(a.predict(x1.reshape(-1, 1)) for a in [tree_reg1, tree_reg2, tree_reg3])
plt.plot(X[:, 0], y, 'y.')
plt.plot(x1, y_pred4, 'r-', linewidth=2, label='$h(x_1)=h_1(x_1)+h_2(x_1)+h_3(x_1)$')
plt.xlabel('$x_1$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.show()
```


    
![](https://github.com/fendao/imgs/blob/main/ml第七章/output_44_0.png)
    



```python
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X, y)
```




    GradientBoostingRegressor(learning_rate=1.0, max_depth=2, n_estimators=3,
                              random_state=42)




```python
gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200)
gbrt_slow.fit(X, y)
```




    GradientBoostingRegressor(max_depth=2, n_estimators=200)




```python
fix,axes = plt.subplots(ncols=2, figsize=(10,4), sharey=True)
plt.sca(axes[0])
ax=[-0.5, 0.5, -0.1, 0.8]
x1 = np.linspace(ax[0], ax[1], 500)
y_pred = sum(regressor.predict(x1.reshape(-1,1)) for regressor in [gbrt])
plt.plot(X[:, 0], y,'b.')
plt.plot(x1, y_pred, 'r-', linewidth=2, label='集成预测')
plt.legend(loc='upper center', fontsize=16)
plt.title('learning_rate={}, n_estimators={}'.format(gbrt.learning_rate, gbrt.n_estimators), fontsize=14)
plt.xlabel('$x_1$', fontsize=16)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.axis(ax)

plt.sca(axes[1])
x1 = np.linspace(ax[0], ax[1], 500)
y_pred2 = sum(regressor.predict(x1.reshape(-1,1)) for regressor in [gbrt_slow])
plt.plot(X[:, 0], y, 'b.')
plt.plot(x1, y_pred2, 'r-', linewidth=2)
plt.title('learning_rate={}, n_estimators={}'.format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=14)
plt.xlabel('$x_1$', fontsize=16)
plt.show()
```


    
![](https://github.com/fendao/imgs/blob/main/ml第七章/output_47_0.png)
    



    
![](https://github.com/fendao/imgs/blob/main/ml第七章/output_47_1.png)
    



```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)
errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors) + 1
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train)
```




    GradientBoostingRegressor(max_depth=2, n_estimators=56, random_state=42)




```python
min_error = np.min(errors)
```


```python
def plot_predictions(regressors, X, y, axes, label=None, style='r-', data_style='b.', data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc='upper center', fontsize=16)
    plt.axis(axes)
```


```python
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.plot(np.arange(1, len(errors) +1), errors, 'b.-')
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], 'k--')
plt.plot([0, 120], [min_error, min_error], 'k--')
plt.plot(bst_n_estimators, min_error, 'ko')
plt.text(bst_n_estimators, min_error*1.2, 'Minimum', ha='center', fontsize=14)
plt.axis([0, 120, 0, 0.01])
plt.xlabel('树数')
plt.ylabel('误差', fontsize=16)
plt.title('验证误差', fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title('Best model (%d trees)' % bst_n_estimators, fontsize=14)
plt.ylabel('$y$', fontsize=16, rotation=0)
plt.xlabel('$x_1$', fontsize=16)
plt.show()
```


    
![](https://github.com/fendao/imgs/blob/main/ml第七章/output_51_0.png)
    



```python
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)
min_val_error = float('inf')
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break
```


```python
print(gbrt.n_estimators)
print('Minimum validation MSE:', min_val_error)
```

    61
    Minimum validation MSE: 0.002712853325235463



```python
try:
    import xgboost
except ImportError as ex:
    print('Error: the xgboost library is not installed.')
    xgboost = None
```


```python
if xgboost is not None:
    xgb_reg = xgboost.XGBRegressor(random_state=42)
    xgb_reg.fit(X_train, y_train,
               eval_set=[(X_val, y_val)], early_stopping_rounds=2)
    y_pred = xgb_reg.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    print('Vallidation MSE:', val_error)
```

    [0]	validation_0-rmse:0.22834
    [1]	validation_0-rmse:0.16224
    [2]	validation_0-rmse:0.11843
    [3]	validation_0-rmse:0.08760
    [4]	validation_0-rmse:0.06848
    [5]	validation_0-rmse:0.05709
    [6]	validation_0-rmse:0.05297
    [7]	validation_0-rmse:0.05129
    [8]	validation_0-rmse:0.05155
    [9]	validation_0-rmse:0.05211
    Vallidation MSE: 0.002630868681577655


   


```python
%timeit xgboost.XGBRegressor().fit(X_train, y_train) if xgboost is not None else None
```

    23.2 ms ± 4.12 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%timeit GradientBoostingRegressor().fit(X_train, y_train)
```

    8.7 ms ± 14.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
# 作业8
from sklearn.model_selection import train_test_split
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)
# X_train_val、y_train_val均50000
X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42)
# X_train、y_train均40000
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42)
```


```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
```


```python
random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(max_iter=100, tol=20, random_state=42)
mlp_clf = MLPClassifier(random_state=42)
```


```python
estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print('Training the', estimator)
    estimator.fit(X_train, y_train)
```

    Training the RandomForestClassifier(random_state=42)
    Training the ExtraTreesClassifier(random_state=42)
    Training the LinearSVC(max_iter=100, random_state=42, tol=20)
    Training the MLPClassifier(random_state=42)



```python
[estimator.score(X_val, y_val) for estimator in estimators]
```




    [0.9692, 0.9715, 0.859, 0.9634]




```python
from sklearn.ensemble import VotingClassifier
named_estimators = [
    ('random_forest_clf', random_forest_clf),
    ('extra_trees_clf', extra_trees_clf),
    ('svm_clf', svm_clf),
    ('mlp_clf', mlp_clf)
]
```


```python
voting_clf = VotingClassifier(named_estimators)
voting_clf.fit(X_train, y_train)
```




    VotingClassifier(estimators=[('random_forest_clf',
                                  RandomForestClassifier(random_state=42)),
                                 ('extra_trees_clf',
                                  ExtraTreesClassifier(random_state=42)),
                                 ('svm_clf',
                                  LinearSVC(max_iter=100, random_state=42, tol=20)),
                                 ('mlp_clf', MLPClassifier(random_state=42))])




```python
voting_clf.score(X_val, y_val)
```




    0.9708




```python
[estimator.score(X_val, y_val) for estimator in voting_clf.estimators_]
```




    [0.9692, 0.9715, 0.859, 0.9634]




```python
voting_clf.set_params(svm_clf=None)
```




    VotingClassifier(estimators=[('random_forest_clf',
                                  RandomForestClassifier(random_state=42)),
                                 ('extra_trees_clf',
                                  ExtraTreesClassifier(random_state=42)),
                                 ('svm_clf', None),
                                 ('mlp_clf', MLPClassifier(random_state=42))])




```python
voting_clf.estimators
```




    [('random_forest_clf', RandomForestClassifier(random_state=42)),
     ('extra_trees_clf', ExtraTreesClassifier(random_state=42)),
     ('svm_clf', None),
     ('mlp_clf', MLPClassifier(random_state=42))]




```python
voting_clf.estimators_
```




    [RandomForestClassifier(random_state=42),
     ExtraTreesClassifier(random_state=42),
     LinearSVC(max_iter=100, random_state=42, tol=20),
     MLPClassifier(random_state=42)]




```python
del voting_clf.estimators_[2]
```


```python
voting_clf.score(X_val, y_val)
```




    0.9735




```python
voting_clf.voting = 'soft'
voting_clf.score(X_val, y_val)
```




    0.9692




```python
voting_clf.voting = 'hard'
voting_clf.score(X_test, y_test)
```




    0.9705




```python
[estimator.score(X_test, y_test) for estimator in voting_clf.estimators_]
```




    [0.9645, 0.9691, 0.9611]



# 作业9


```python
# 一万行四列0矩阵
X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)
# 索引0123和预测器名对照
for index, estimator in enumerate(estimators):
    # 四个预测器预测的数据分别放到矩阵四列，从而创建了新的训练集
    X_val_predictions[:, index] = estimator.predict(X_val)
```


```python
# 新训练集训练RF混合器
rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender.fit(X_val_predictions, y_val)
```




    RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)




```python
rnd_forest_blender.oob_score_
```




    0.9698




```python
# 对测试集的图片作出预测：各分类器的预测结果交给混合器，得到集成预测
# 1000行四列矩阵，方便放置4个分类器的预测结果
X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)
for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)
```


```python
# 混合器对各分类器在test集预测结果进行预测
y_pred = rnd_forest_blender.predict(X_test_predictions)
```


```python
# 混合器集成预测的得分
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```




    0.9686




```python

```
