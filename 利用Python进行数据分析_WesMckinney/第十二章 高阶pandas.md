```python
import numpy as np
import pandas as pd
```


```python
values = pd.Series([0,1,0,0] * 2)
```


```python
dim = pd.Series(['apple','orange'])
```


```python
dim.take(values)
```




    0     apple
    1    orange
    0     apple
    0     apple
    0     apple
    1    orange
    0     apple
    0     apple
    dtype: object




```python
fruits = ['apple','orange','apple','apple'] * 2
```


```python
N = len(fruits)
```


```python
df = pd.DataFrame({'fruit':fruits,
                  'basket_id':np.arange(N),
                  'count':np.random.randint(3,15,size=N),
                  'weight':np.random.uniform(0,4,size=N),
                  },columns=['basket_id','fruit','count','weight'])
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>basket_id</th>
      <th>fruit</th>
      <th>count</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>apple</td>
      <td>12</td>
      <td>1.663990</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>orange</td>
      <td>6</td>
      <td>3.024129</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>apple</td>
      <td>5</td>
      <td>0.878946</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>apple</td>
      <td>8</td>
      <td>3.433659</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>apple</td>
      <td>4</td>
      <td>1.436422</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>orange</td>
      <td>10</td>
      <td>1.804335</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>apple</td>
      <td>10</td>
      <td>2.195085</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>apple</td>
      <td>3</td>
      <td>1.893930</td>
    </tr>
  </tbody>
</table>
</div>




```python
fruit_cat = df['fruit'].astype('category')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-6-4c297ee39197> in <module>
    ----> 1 fruit_cat = df['fruit'].astype('category')
    

    NameError: name 'df' is not defined



```python
fruit_cat
```




    0     apple
    1    orange
    2     apple
    3     apple
    4     apple
    5    orange
    6     apple
    7     apple
    Name: fruit, dtype: category
    Categories (2, object): ['apple', 'orange']




```python
c = fruit_cat.values
```


```python
type(c)
```




    pandas.core.arrays.categorical.Categorical




```python
c.categories
```




    Index(['apple', 'orange'], dtype='object')




```python
c.codes
```




    array([0, 1, 0, 0, 0, 1, 0, 0], dtype=int8)




```python
my_cat_2 = pd.Categorical.from_codes([0,1,2,0,0,1],['foo','bar','baz'],ordered=True)
```


```python
my_cat_2
```




    ['foo', 'bar', 'baz', 'foo', 'foo', 'bar']
    Categories (3, object): ['foo' < 'bar' < 'baz']




```python
my_cat_2.as_ordered()
```




    ['foo', 'bar', 'baz', 'foo', 'foo', 'bar']
    Categories (3, object): ['foo' < 'bar' < 'baz']




```python
np.random.seed(12345)
```


```python
draws = np.random.randn(1000)
```


```python
draws[:5]
```




    array([-0.20470766,  0.47894334, -0.51943872, -0.5557303 ,  1.96578057])




```python
bins = pd.qcut(draws,4)
```


```python
bins
```




    [(-0.684, -0.0101], (-0.0101, 0.63], (-0.684, -0.0101], (-0.684, -0.0101], (0.63, 3.928], ..., (-0.0101, 0.63], (-0.684, -0.0101], (-2.9499999999999997, -0.684], (-0.0101, 0.63], (0.63, 3.928]]
    Length: 1000
    Categories (4, interval[float64, right]): [(-2.9499999999999997, -0.684] < (-0.684, -0.0101] < (-0.0101, 0.63] < (0.63, 3.928]]




```python
bins = pd.qcut(draws,4,labels=['Q1','Q2','Q3','Q4'])
```


```python
bins
```




    ['Q2', 'Q3', 'Q2', 'Q2', 'Q4', ..., 'Q3', 'Q2', 'Q1', 'Q3', 'Q4']
    Length: 1000
    Categories (4, object): ['Q1' < 'Q2' < 'Q3' < 'Q4']




```python
bins.codes[:10]
```




    array([1, 2, 1, 1, 3, 3, 2, 2, 3, 3], dtype=int8)




```python
bins = pd.Series(bins, name='quartile')
```


```python
results = (pd.Series(draws)).groupby(bins).agg(['count','min','max']).reset_index()
```


```python
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>quartile</th>
      <th>count</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Q1</td>
      <td>250</td>
      <td>-2.949343</td>
      <td>-0.685484</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Q2</td>
      <td>250</td>
      <td>-0.683066</td>
      <td>-0.010115</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Q3</td>
      <td>250</td>
      <td>-0.010032</td>
      <td>0.628894</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Q4</td>
      <td>250</td>
      <td>0.634238</td>
      <td>3.927528</td>
    </tr>
  </tbody>
</table>
</div>




```python
results['quartile']
```




    0    Q1
    1    Q2
    2    Q3
    3    Q4
    Name: quartile, dtype: category
    Categories (4, object): ['Q1' < 'Q2' < 'Q3' < 'Q4']




```python
with open('test.txt', 'a') as h:
    for i in range(15):
        h.write(str(np.random.randint(0,10,15)).replace(' ','').replace('[','').replace(']','')+'\n')
```


```python
!cat test.txt
```

    965331196938919
    563821670956390
    694684850985885
    663639108454974
    673422860237771
    768421189642460
    625700406340292
    385090077988369
    279429395892573
    575893954807037
    758213273585872
    653868641392453
    951123826255724
    335653250244228
    296935509798947



```python
count = 0
with open('test.txt', 'r+') as f:
    for line in f:
        count += 1
        print(line)
        str.replace(line.read[14][-2:-1],'中国') if count = 1
```


      File "<ipython-input-101-c7af1435dbae>", line 6
        str.replace(line.read[14][-2:-1],'中国') if count = 1
                                                        ^
    SyntaxError: invalid syntax




```python
with open('test.txt', 'r') as f:
    print(f.readlines(20))
#     str.replace((f.read(14)[-2:-1]),'中国')
```

    ['965331196938919\n', '563821670956390\n']



```python
lines = []
with open('test.txt', 'r+') as f:
    f.seek(11)
    f.write('中')
    print(f.readlines())
#     f.read(14).replace(f.read(14)[12:14],'中国')
#     print(f.read()[:80])
#     if '\n' not in f.read()[:14]:
        
#     print(type(f.read()[12:14]))
#     for line in f.readlines():
#         line += line.rstrip("\n")
#     line.replace(line[12:14],'中国')
#     print(line)
```

    ['89中563821670956390\n', '694684850985885\n', '663639108454974\n', '673422860237771\n', '768421189642460\n', '625700406340292\n', '385090077988369\n', '279429395892573\n', '575893954807037\n', '758213273585872\n', '653868641392453\n', '951123826255724\n', '335653250244228\n', '296935509798947\n', '中']



```python
[2,3,4].count()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-156-98bcd389da34> in <module>
    ----> 1 [2,3,4].count()
    

    TypeError: count() takes exactly one argument (0 given)



```python
with open('test.txt', 'r') as f:
    for line in f.readlines():
        print(line)
```

    要密切监测工业经济运行情况高度关注工业企业的运营状况
    
    强化服务保障，促进企业健康发展
    
    要抓好平台建设，加快推进咸阳市新材料产业园、工业集中区、灵源建材产业园、五峰山氧化钙产业园建设，切实发挥园区聚集效应。要加大规模以上工业企业培育力度，苟晓瑜同志牵头，加快建立园区入驻企业审核制度
    
    对工业集中区、灵源建材产业园等园区入驻企业规模进行排查摸底，未达到规上标准的原则上不得入驻园区。



```python
a = '要密切监测工业经济运行情况高度关注工业企业的运营状况\n强化服务保障，促进企业健康发展\n要抓好平台建设，加快推进新材料产业园、工业集中区、建材产业园建设，切实发挥园区聚集效应'
with open('test2.txt','w') as f:
    f.write(a)
```


```python
# 作业1
with open('test2.txt', 'r') as f:
    for line in f.readlines():
        print(line)
```

    要密切监测工业经济运行情况高度关注工业企业的运营状况
    
    强化服务保障，促进企业健康发展
    
    要抓好平台建设，加快推进新材料产业园、工业集中区、建材产业园建设，切实发挥园区聚集效应



```python
# 作业2
with open('test2.txt','r') as f1:
    with open('test.txt','w') as f2:
        fx = f1.read()
        if '\n' in fx[:14]:    # 去除字符串行末换行符影响
            a = 2 * fx[:14].count('\n')
            b = fx[:14+a].replace(fx[12+a:14+a],'中国')
            b2 = b + fx[14+a:]
            f2.write(b2)
        else:
            b = fx[:14].replace(fx[12:14],'中国')
            b2 = b + fx[14:]
            f2.write(b2)
with open('test.txt', 'r') as f:
    for line in f.readlines():
        print(line)
```

    要密切监测工业经济运行情中国度关注工业企业的运营状况
    
    强化服务保障，促进企业健康发展
    
    要抓好平台建设，加快推进新材料产业园、工业集中区、建材产业园建设，切实发挥园区聚集效应



```python
2 * '\n\n\nsdfs'.count('\n')
```




    6




```python
a = 2
b='asfsdfsdfg'
c = b.replace(b[0:a],'11')
'99'.join(c)
```




    '199199f99s99d99f99s99d99f99g'




```python

```




    'asfsdfsdfg'




```python
ee = '要密切监测工业经济运行情况高度关注工业企业的运营状况\n强化服务保障，促进企业健康发展\n要抓好平台建设，加快推进新材料产业园、工业集中区、建材产业园建设，切实发挥园区聚集效应'
with open('test1.txt','w') as f:
    f.write(ee)
# 作业1
with open('作业.txt', 'r') as f:
    for line in f.readlines():
        print(line)
# 作业2
with open('作业.txt','r') as f1:
    with open('作业2.txt','w') as f2:
        fx = f1.read()
        if '\n' in fx[:14]:    # 去除字符串行末换行符影响
            a = 2 * fx[:14].count('\n')
            b = fx.replace(fx[12+a:14+a],'中国')
            f2.write(b)
        else:
            b = fx.replace(fx[12:14],'中国')
            f2.write(b)
with open('作业2.txt', 'r') as f:
        print(f.read())
```


```python
# 作业1
with open('test1.txt', 'r') as f:
    for line in f.readlines():
        print(line)
```

    要密切监测工业经济运行情况高度关注工业企业的运营状况
    
    强化服务保障，促进企业健康发展
    
    要抓好平台建设，加快推进新材料产业园、工业集中区、建材产业园建设，切实发挥园区聚集效应



```python
# 作业2
with open('test1.txt','r') as f1:
    with open('test2.txt','w') as f2:
        fx = f1.read()
        if '\n' in fx[:14]:    # 去除字符串行末换行符影响
            a = 2 * fx[:14].count('\n')
            b = fx.replace(fx[12+a:14+a],'中国')
            f2.write(b)
        else:
            b = fx.replace(fx[12:14],'中国')
            f2.write(b)
with open('test2.txt', 'r') as f:
        print(f.read())
```

    要密切监测工业经济运行情中国度关注工业企业的运营状况
    强化服务保障，促进企业健康发展
    要抓好平台建设，加快推进新材料产业园、工业集中区、建材产业园建设，切实发挥园区聚集效应



```python

```
