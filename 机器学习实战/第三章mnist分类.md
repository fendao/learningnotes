# 整理主要流程——MNIST
mnist = fetch_openml('mnist_784', version=1)  
第一个实例的特征向量重组    some_digit_image = some_digit.values.reshape(28, 28)  
切分训练测试集 X_train\X_test\y_train\y_test
# 二元分类
目标与非目标布尔向量组    y_train_5 = (y_train == 5)   y_test_5 = (y_test == 5)  
上SGD分类器训练    实例（种子），数据训练，predict True or False  
评估分类器——交叉验证  
    StratifiedKFold（组数，种子），split将D（X_train）按y_train_5布尔组的T、F比例分层采样  
    按长度对训练数据评估正确率sum(y_pred == y_test_fold) / len(y_pred)，上cross_val_score评估准确率  
    但类别多数据少，即使猜数据中全是False也会很高正确率——正确率不能评估分类器  
混淆矩阵（行是实际，列是预测，得混淆次数）  
    上干净预测数据    **y_train_pred = cross_val_predict(分类器, D, 布尔向量, cv3)**  
    上实际f与预测h的混淆矩阵    **confusoin_matrix(y_train_5, y_train_pred)**  
    计算精度召回率\谐波均值    **precision_score(布尔向量f, 预测数据pred)**    recall_score(f, pred)    f1_score(f, pred)  
    调整判断正类的阙值：提高阙值，假正会变真负，所以精度提高，真正会变假负，所以召回率降低  
    阙值决定：返回所有实例决策分数  y_scores = cross_val_predict(sgd_clf, D, 布尔向量f, cv3, method='decision_function')  
            绘制精度召回阙值函数图  precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)  
            绘制精度召回函数图（PR曲线）  
            查精度90的阙值  threshold_90_precision = thresholds[np.argmax(precisions >= 0.9)]  
    ROC曲线：绘制FPR对TPR曲线  fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)  
            查ROC AUC（越接近1越准）    roc_auc_score(y_train_5, y_scores)  
    经验法则：关注假正（错误率），选择PR；；；关注假负（召回率），选择ROC  
分类器比较——RandomForest分类器  
    创建实例、获取概率矩阵  y_probas_forest = cross_val_predict(forest_clf, D, f, cv3, method='predict_proba')  
    获取正类分数、绘图     y_scores_forest = y_probas_forest[:, 1]  
                        fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)  
    查roc auc        roc_auc_score(布尔数组f, 决策分数score)
# 多类分类
使用二元分类分类多类时，Scikit-Learn自动Ovr、OvO  
SVM分类器：  
    可强制策略： ovr_clf = OneVsRestClassfier(SVC())  
    实例、训练数据（D, 标签向量f）、利用特征进行预测svm_clf.predict([some_digit])  
    获取决策分数  some_digit_scores = svm_clf.decision_function([some_digit])、class_获取原始类别  
    性能提升：特征缩放、超参微调、
# 误差分析
多类分类预测数据    y_train_pred = cross_val_predict(分类器, D, 标签向量f, cv3)  
混淆矩阵          conf_mx = confusion_matrix(实际标签y_train, 预测数据y_train_pred)  
matshow直接看矩阵，暗表示不太好  
matshow看错误率矩阵  conf_mx / conf_mx.sum(axis=1, keepdims=True)  
分析错误并改进：增加数据、搞新特征区分、上实例图看哪有问题使图片突出模式
# 多标签分类（）
多标签布尔数组    y_multilabel = np.c_[判定布尔向量f1, 判定布尔向量f2]  
KNeighborsClassfier().fit(D, y_multilabel)  
KNeighborsClassfier().predict([some_digit])  
计算f1分数：  
    上预测数据    pred = cross_val_predict(分类器, D, y_multilabel, cv3)  
    计算平均f1:   f1_score(y_multilabel, pred, average='macro')
# 多输出分类
给D每个像素加入噪音，将干净数据作为标签y，训练分类器  KNeighborsClassfier().fit(D, y)  
使用含噪测试集预测数据  KNeighborsClassfier().predict([X_test_mod[some_index]])


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
```


```python
from sklearn.datasets import fetch_openml
```


```python
import matplotlib as mpl
import matplotlib.pyplot as plt
```


```python
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
```




    dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])




```python
X, y = mnist['data'], mnist['target']
```


```python
X.shape
```




    (70000, 784)




```python
y.shape
```




    (70000,)




```python
some_digit = X.iloc[0]
```


```python
some_digit_image = some_digit.values.reshape(28, 28)
```


```python
plt.imshow(some_digit_image, cmap='binary')
plt.axis('off')
plt.show()
```


    
![](https://img-blog.csdnimg.cn/3ed208048dc4402a855433d8ce68327b.png#pic_center)




```python
y[0]
```




    '5'




```python
y = y.astype(np.uint8)
```


```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```


```python
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
```


```python
from sklearn.linear_model import SGDClassifier
```


```python
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```




    SGDClassifier(random_state=42)




```python
sgd_clf.predict([some_digit])
```




    array([ True])




```python
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
```


```python
skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
```


```python
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train.iloc[train_index]
    y_train_folds = y_train_5.iloc[train_index]
    X_test_fold = X_train.iloc[test_index]
    y_test_fold = y_train_5.iloc[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
```

    0.9669
    0.91625
    0.96785



```python
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, scoring='accuracy', cv=3)
```




    array([0.95035, 0.96035, 0.9604 ])




```python
from sklearn.base import BaseEstimator
```


```python
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
```


```python
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring='accuracy')
```




    array([0.91125, 0.90855, 0.90915])




```python
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```


```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
```




    array([[53892,   687],
           [ 1891,  3530]])




```python
y_train_predictions = y_train_5
confusion_matrix(y_train_5, y_train_predictions)
```




    array([[54579,     0],
           [    0,  5421]])




```python
from sklearn.metrics import precision_score, recall_score
```


```python
precision_score(y_train_5, y_train_pred)
```




    0.8370879772350012




```python
recall_score(y_train_5, y_train_pred)
```




    0.6511713705958311




```python
from sklearn.metrics import f1_score
```


```python
f1_score(y_train_5, y_train_pred)
```




    0.7325171197343846




```python
y_scores = sgd_clf.decision_function([some_digit])
y_scores
```

  

    array([2164.22030239])




```python
threshold = 0
y_some_digit_pred = (y_scores > threshold)
```


```python
y_some_digit_pred
```




    array([ True])




```python
threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
```




    array([False])




```python
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
```


```python
from sklearn.metrics import precision_recall_curve
```


```python
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```


```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlabel('阙值', fontsize=16)
plt.legend(loc='center right', fontsize=16)
plt.grid(True)
plt.axis([-50000, 50000, 0, 1])
plt.show()
```


    
![](https://img-blog.csdnimg.cn/21f0746cf0564a2fbbabc151c506882f.png#pic_center)

    



```python
threshold_90_precision = thresholds[np.argmax(precisions >= 0.9)]
```


```python
y_train_pred_90 = (y_scores >= threshold_90_precision)
```


```python
precision_score(y_train_5, y_train_pred_90)
```




    0.9000345901072293




```python
recall_score(y_train_5, y_train_pred_90)
```




    0.4799852425751706




```python
recall_90_prcision = recalls[np.argmax(precisions >= 0.9)]
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.xlabel('召回率', fontsize=16)
    plt.ylabel('精度', fontsize=16)
    plt.axis([0,1,0,1])
    plt.grid(True)
plt.figure(figsize=(8,6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_prcision, recall_90_prcision], [0, 0.9], 'r:')
plt.plot([0, recall_90_prcision], [0.9, 0.9], 'r:')
plt.plot([recall_90_prcision], [0.9], 'ro')
plt.show()
```


    <Figure size 800x600 with 0 Axes>



    


![](https://img-blog.csdnimg.cn/263c888ff3484fe0b3f3e39962b19478.png#pic_center)


```python
from sklearn.metrics import roc_curve
```


```python
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```


```python
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
plot_roc_curve(fpr, tpr)
plt.axis([0,1,0,1])
plt.xlabel('假正类率FPR')
plt.ylabel('真正类率TPR')
plt.grid(True)
plt.show()
```


    
![](https://img-blog.csdnimg.cn/66f3f06ce6184f478c751a6a84d213a9.png#pic_center)

    



```python
from sklearn.metrics import roc_auc_score
```


```python
roc_auc_score(y_train_5, y_scores)
```




    0.9604938554008616




```python
from sklearn.ensemble import RandomForestClassifier
```


```python
forect_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forect_clf, X_train, y_train_5, cv=3, method='predict_proba')
```


```python
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
```


```python
plt.plot(fpr, tpr, 'b:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
plt.legend(loc='lower right')
plt.show()
```


    
![](https://img-blog.csdnimg.cn/4148f8b94fbd403facbc9fcfa61ca5bc.png#pic_center)

    



```python
roc_auc_score(y_train_5, y_scores_)
```




    0.9983436731328145




```python
precisions2, recalls2, thresholds2 = precision_recall_curve(y_train_5, y_scores_forest)
```


```python
precision_score(y_train_5, (y_scores_forest >= threshold_90_precision))
```







    0.0




```python
y_train_pred_forest = cross_val_predict(forect_clf, X_train, y_train_5, cv=3)
precision_score(y_train_5, y_train_pred_forest)
```




    0.9905083315756169




```python
recall_score(y_train_5, y_train_pred_forest)
```




    0.8662608374838591




```python
from sklearn.svm import SVC
```


```python
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_clf.predict([some_digit])
```






    array([5], dtype=uint8)




```python
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores
```





    array([[ 1.72501977,  2.72809088,  7.2510018 ,  8.3076379 , -0.31087254,
             9.3132482 ,  1.70975103,  2.76765202,  6.23049537,  4.84771048]])




```python
np.argmax(some_digit_scores)
```




    5




```python
svm_clf.classes_
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)




```python
svm_clf.classes_[5]
```




    5




```python
from sklearn.multiclass import OneVsRestClassifier
```


```python
# 处理X does not have valid feature names, but，可将驯良数据df.to_numpy()
%%time
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
ovr_clf.predict([some_digit])
```

    CPU times: user 10min 24s, sys: 5.99 s, total: 10min 30s
    Wall time: 10min 31s



    array([5], dtype=uint8)




```python
len(ovr_clf.estimators_)
```




    10




```python
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
```

    /Users/zhangdi/opt/miniconda3/lib/python3.7/site-packages/sklearn/base.py:451: UserWarning: X does not have valid feature names, but SGDClassifier was fitted with feature names
      "X does not have valid feature names, but"





    array([3], dtype=uint8)




```python
sgd_clf.decision_function([some_digit])
```


    array([[-31893.03095419, -34419.69069632,  -9530.63950739,
              1823.73154031, -22320.14822878,  -1385.80478895,
            -26188.91070951, -16147.51323997,  -4604.35491274,
            -12050.767298  ]])




```python
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')
```




    array([0.87365, 0.85835, 0.8689 ])




```python
%%time
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')
```

    CPU times: user 8min 31s, sys: 4.81 s, total: 8min 36s
    Wall time: 8min 33s





    array([0.8983, 0.891 , 0.9018])




```python
%%time
y_trian_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
```

    CPU times: user 8min 25s, sys: 4.36 s, total: 8min 29s
    Wall time: 1h 4min 1s





    array([[5860,   63,    0,    0,    0,    0,    0,    0,    0,    0],
           [6675,   67,    0,    0,    0,    0,    0,    0,    0,    0],
           [5932,   26,    0,    0,    0,    0,    0,    0,    0,    0],
           [5913,  218,    0,    0,    0,    0,    0,    0,    0,    0],
           [5821,   21,    0,    0,    0,    0,    0,    0,    0,    0],
           [1891, 3530,    0,    0,    0,    0,    0,    0,    0,    0],
           [5796,  122,    0,    0,    0,    0,    0,    0,    0,    0],
           [6251,   14,    0,    0,    0,    0,    0,    0,    0,    0],
           [5741,  110,    0,    0,    0,    0,    0,    0,    0,    0],
           [5903,   46,    0,    0,    0,    0,    0,    0,    0,    0]])




```python
def plot_confusion_matrix(matrix):#对个数大小绘图
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
```


    
![](https://img-blog.csdnimg.cn/e5bc99d6220848ebb48b2177f8badb82.png#pic_center)

    



```python
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
```


```python
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
```


    
![](https://img-blog.csdnimg.cn/67692c13a1cd409f9f90abdd5d41d5a2.png#pic_center)

    



```python
import matplotlib as mpl
def plot_digit(data):
    image = data.values.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary,
              interpolation='nearest')
    plt.axis('off')
def plot_digit2(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary,
              interpolation='nearest')
    plt.axis('off')
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    # 每行一个
    image_pre_row = min(len(instances), images_per_row)
    images=[instances.values.reshape(size, size) for instances in instances]
    # 定义行
    n_rows = (len(instances)-1) // image_pre_row+1
    row_images=[]
    n_empty = n_rows * image_pre_row-len(instances)
    images.append(np.zeros((size, size*n_empty)))
    for row in range(n_rows):
        # 每次添加一行
        rimages = images[row * image_pre_row:(row+1)*image_pre_row]
        # 左右连接
        row_images.append(np.concatenate(rimages, axis=1))
    # 上下连接
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, ** options)
    plt.axis('off')
```


```python
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_aa[:25], images_per_row=5)
plt.show()
```


    ---------------------------------------------------------------------------

 
    array([[False,  True]])




```python
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average='macro')
```




    0.976410265560605




```python
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
```


```python
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
```


```python
X_test_mod.shape
```




    (10000, 784)




```python
some_index = 5500
plt.subplot(121);plot_digit(X_test_mod.iloc[some_index])
plt.subplot(122);plot_digit(y_test_mod.iloc[some_index])
plt.show()
```


    
![](https://img-blog.csdnimg.cn/afd0b17647bc425d8a0a8eda5ce32bcc.png#pic_center)

    



    
![](https://img-blog.csdnimg.cn/ce12bf411f8b40a99ae58ee5130ce250.png#pic_center)

    



    
![](https://img-blog.csdnimg.cn/d1b568b403914bea827cea0959a6f341.png#pic_center)

    



    
![](https://img-blog.csdnimg.cn/1912ce19b5ff4649bb2f54d1fc9abce5.png#pic_center)

    



    
![](https://img-blog.csdnimg.cn/54779ee9953c4963a66f78feadfdb0dc.png#pic_center)

    



    
![](https://img-blog.csdnimg.cn/33f2b96385aa40aea94b35fc06695841.png#pic_center)

    



    
![](https://img-blog.csdnimg.cn/fb358a8b6ed145a2a43ac33953c85ad0.png#pic_center)

    



    
![](https://img-blog.csdnimg.cn/8891cf4f9ebf43f69fb165d1deb4fd22.png#pic_center)

    



    
![](https://img-blog.csdnimg.cn/64175905b66f4b38a7d920a4ff8145a2.png#pic_center)

    



    
![](https://img-blog.csdnimg.cn/4fc569ae5efe4169ae395e1b5722095f.png#pic_center)

    



    
![](https://img-blog.csdnimg.cn/60dd04cf4cf749ca82c1178f529d0a00.png#pic_center)

    



    
![](https://img-blog.csdnimg.cn/5bfe71d06c8248128e23c8ffbb84ce63.png#pic_center)

    



```python
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod.iloc[some_index]])
plot_digit2(clean_digit)
```



```python

```
