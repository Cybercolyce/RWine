<center><h1>一个红酒品质分析的实例</h1></center>

# 引言

数据来源于[kaggle](https://www.kaggle.com/piyushgoyal443/red-wine-dataset).

# 数据引入

我们先来导入数据：

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn import model_selection
data=pd.read_csv('winequality-red.csv')
```

我们先来看看这个数据的大致模样：

```python
data.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>


再看看汇总的信息：

```python
data.info()
```



```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1599 entries, 0 to 1598
Data columns (total 12 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   fixed acidity         1599 non-null   float64
 1   volatile acidity      1599 non-null   float64
 2   citric acid           1599 non-null   float64
 3   residual sugar        1599 non-null   float64
 4   chlorides             1599 non-null   float64
 5   free sulfur dioxide   1599 non-null   float64
 6   total sulfur dioxide  1599 non-null   float64
 7   density               1599 non-null   float64
 8   pH                    1599 non-null   float64
 9   sulphates             1599 non-null   float64
 10  alcohol               1599 non-null   float64
 11  quality               1599 non-null   int64  
dtypes: float64(11), int64(1)
memory usage: 150.0 KB
```



可以看到我们一共有11个指标，一个待预测的结果。 具体含义在此不探究，只关心指标与结果的关系，先来查看各个指标与因变量的关系。 在`data.info()`中我们可以查看到所有的指标都是浮点型数据，不存在特殊类型如字符、布尔型的数据，故我们选用箱型图进行初步的关系探究：

首先先研究分类结果的分布情况：

```python
sns.countplot(x='quality',data=data)#探究结果的分布情况
```

![](https://i.loli.net/2021/05/25/F9gjoPYESVDMKUw.png)

# 指标与预测结果的相关的初步分析

接下来可以对不同指标与分类结果进行初步的相关性判断：

```python
#探究 fixed acidity与指标的关系
sns.boxplot('quality', 'fixed acidity', data = data)
```

![image-20210525192808821](https://i.loli.net/2021/05/25/tHckKpTQnz7heaA.png)

这个指标的分布成一个上升的趋势，但在品质8的时候出现了回落，同时在品质5的地方出现比较多的异常值，后续需要注意该段的处理。

```python
#volatile acidity
sns.boxplot('quality', 'volatile acidity', data = data)
```

![image-20210525193915603](https://i.loli.net/2021/05/25/7bvsOBFjo8m2M6z.png)

一个非常经典的递减型！有点指数衰减的样子，同样在品质5的分类中出现了较多的异常值，需要注意。

```python
#citric acid
sns.boxplot('quality', 'citric acid', data = data)
```

![image-20210525193941973](https://i.loli.net/2021/05/25/yln7Eepb8Zg2sGP.png)

递增型，有点指数递增的味道了，这个时候的数据都比较理想。

```python
#residual sugar
sns.boxplot('quality','residual sugar',data=data)
```

![image-20210525194033676](https://i.loli.net/2021/05/25/BHyQwIG65lYPnht.png)

这个应该是最难看出特征的数据了，所以这一块要重点处理，考虑到比较多的异常值以及过窄的箱体，我打算使用Z-score进行标准化再来检验下效果如何。

```python
#chlorides
sns.boxplot('quality','chlorides',data=data)
```

![image-20210525194110736](https://i.loli.net/2021/05/25/WUHSBGYhcZXbMy4.png)

依然同上情况，标准化的选择要多试验。

```python
#free sulfur dioxide
sns.boxplot('quality','free sulfur dioxide',data=data)
```

![image-20210525194131731](https://i.loli.net/2021/05/25/QKRiwXeyGqu3IJZ.png)

先增后减，类似正态分布，暂时没有想法。

```python
#total sulfur dioxide
sns.boxplot('quality','total sulfur dioxide',data=data)
```

![image-20210525194204693](https://i.loli.net/2021/05/25/QXLpn1M4W9aDrhi.png)

情况也是先增后减，类似正态分布。

```python
#density
sns.boxplot('quality','density',data=data)
```

![image-20210525194403929](https://i.loli.net/2021/05/25/VYgk3BNsDyUWrPm.png)

品质5出现较大偏差，整体其实是一个递减的结构，但异常值较多的5品质也许影响了整体。

```python
#pH
sns.boxplot('quality','pH',data=data)
```

![image-20210525194423754](https://i.loli.net/2021/05/25/KAz7ep4SDVwXxlh.png)

同上，不过这次异常值都是6品质。

```python
#sulphates
sns.boxplot('quality','sulphates',data=data)
```

![image-20210525194442712](https://i.loli.net/2021/05/25/WvLxRJH9mbjSTeP.png)

较多的异常值出现在5，箱体也偏狭窄，整体是一个递增结构。

```python
#alcohol
sns.boxplot('quality','alcohol',data=data)
```

![image-20210525194508183](https://i.loli.net/2021/05/25/cIBm34DWNilw1ev.png)

品质5的异常值太多，整体是递增趋势。

从上面初步的分析结果来看，品质5的异常情况较多，大部分数据的分布是比较理想的（正态、递增以及递减型）。因此我们要着重关心品质5以及标准化的选择。品质5与6的分类可能比较困难，因此如果不做一些处理的话分类结果可能会很不理想。

# 第一次分类的尝试

首先我们对数据的指标进行标准化，利用sklearn中的`StandardScaler`方法：

```python
x=data.iloc[:,:11]
y=data['quality']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
```

接着我们进行SVM的分类初尝试：

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
svc = SVC()
svc.fit(x_train, y_train)
print (svc.score(x_train, y_train))
```

输出的结果：

```
0.64634675346432149
```

在这里我们可以看出，这样的分类结果并不是很理想。其实问题主要出在品质5与品质6的分类上。如果我们采用主成分分析法，筛掉一些影响不是很大的指标，或许能有所改变。

# 第二次分类的想法

这一次我们加入主成分分析进行降维：

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
pca = PCA()
x_pca = pca.fit_transform(x)
#plot the graph to find the principal components
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()
```

![image-20210525195623236](https://i.loli.net/2021/05/25/xtSCX9LsubBWyFG.png)

可以看到在8个维度的时候效果已经达到了0.9，我们采用8个维度进行分析：

```python
pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(x)
print(x_new)
```

输出的结果如下所示：

```
[[-1.61952988  0.45095009 -1.77445415 ... -0.91392069 -0.16104319
  -0.28225828]
 [-0.79916993  1.85655306 -0.91169017 ...  0.92971392 -1.00982858
   0.76258697]
 [-0.74847909  0.88203886 -1.17139423 ...  0.40147313 -0.53955348
   0.59794606]
 ...
 [-1.45612897  0.31174559  1.12423941 ... -0.50640956 -0.23108221
   0.07938219]
 [-2.27051793  0.97979111  0.62796456 ... -0.86040762 -0.32148695
  -0.46887589]
 [-0.42697475 -0.53669021  1.6289552  ... -0.49615364  1.18913227
   0.04217568]]
```

接下来就是第二次分类的尝试：

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.25)
svc = SVC()
svc.fit(x_train, y_train)
print (svc.score(x_train, y_train))
```

输出结果：

```
0.6763969974979149
```

拟合度尽管有所提升，但依然没有很理想。我们需要牺牲一些分类的精度，引入一定的模糊性。换句话说即将六个分类级别缩减（尤其是合并456级别），从而提高分类的准确性。

# 第三次分类的尝试

不妨设品质小于3的设置为低品质（标号为1），品质为4-7的设为中等品质（标号为2），品质大于8的设置为高品质（标号3）。

```python
reviews = []
for i in data['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
data['Reviews'] = reviews
```

此时同样重复主成分分析法进行降维：

```python
x = data.iloc[:,:11]
y = data['Reviews']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(x)

#plot the graph to find the principal components
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()
```

输出结果如图所示：

![image-20210525200251891](https://i.loli.net/2021/05/25/DRfGH3UWqEXZlmB.png)

根据这次的降维结果，我们可以选择6个维度进行分析：

```python
from sklearn.model_selection import train_test_split
pca_new = PCA(n_components=6)
x_new = pca_new.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.25)
lin_svc = SVC()
lin_svc.fit(x_train, y_train)
print(lin_svc.score(x_train,y_train))
```

输出结果：

```
0.9808173477898249
```

拟合度达到了98%左右，效果还是比较理想的。
