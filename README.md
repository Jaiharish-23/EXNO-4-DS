# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df=pd.read_csv("/content/income1.csv.csv",na_values=[ " ?"])
df
```

<img width="1770" height="524" alt="{C4383450-F1B3-47BB-AECD-B98F17164DE8}" src="https://github.com/user-attachments/assets/4dd2635e-22ce-42a0-abcf-d71f84f80b54" />

```python
df.isnull().sum()
```

<img width="448" height="614" alt="{16588915-A0B4-43E6-BB55-13AA3DC5E165}" src="https://github.com/user-attachments/assets/5038ca77-efca-411a-93bb-ffe98edf75fc" />

```python
ms=df[df.isnull().any(axis=1)]
ms
```

<img width="1774" height="518" alt="{658FE98F-DE94-4CD9-A733-21F491EB113B}" src="https://github.com/user-attachments/assets/af43dc56-8814-43c6-975b-afa8e65cb036" />

```python
df2=df.dropna(axis=0)
df2
```

<img width="1754" height="527" alt="{647AA412-7121-4381-A5B1-F96C748C4DD1}" src="https://github.com/user-attachments/assets/a0ed5dba-f8de-46ae-9107-eab43a56384b" />

```python
s=df["SalStat"]
df2["SalStat"]=df["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(df2['SalStat'])
```

<img width="1497" height="403" alt="{CD945FED-9688-428C-85AA-BFE2348F6569}" src="https://github.com/user-attachments/assets/0684674c-8e89-4ff9-b519-30d77178a216" />

```python
s2=df2['SalStat']
dfs=pd.concat([s,s2],axis=1)
dfs
```

<img width="554" height="531" alt="{33ABE9ED-60A4-4066-A798-C31A80247F5F}" src="https://github.com/user-attachments/assets/55a357b2-1f08-407f-ac11-b3a3079ce9df" />

```python
new_df2=pd.get_dummies(df2, drop_first=True)
new_df2
```

<img width="1755" height="606" alt="{DB0045E6-D03C-46FD-8799-396356CA4103}" src="https://github.com/user-attachments/assets/6173b0e9-8ebc-4bef-ba93-fede9cfff2bb" />

```python
col=list(new_df2.columns)
print(col)
```

<img width="1776" height="52" alt="{091C67A2-43BC-4C7B-A9FE-D56063AF9D11}" src="https://github.com/user-attachments/assets/4b55d917-7611-4805-83af-79ba398a8f28" />

```python
fea=list(set(col)-set(['SalStat']))
print(fea)
```

<img width="1780" height="42" alt="{C5953825-B758-4C5C-A4B9-9DE3AB92D516}" src="https://github.com/user-attachments/assets/13bd6799-1cd8-4040-9372-27a376805a95" />

```python
y=new_df2['SalStat'].values
print(y)
```

<img width="1296" height="39" alt="{AA02F79C-12CE-4C08-BD8F-2EA23E90DA2A}" src="https://github.com/user-attachments/assets/a6f84c25-bbe0-46c6-be1c-f583741caaeb" />

```python
x=new_df2[fea].values
print(x)
```

<img width="879" height="182" alt="{BC8E41A8-C7A2-49F4-8A8C-5E62DA7BDC60}" src="https://github.com/user-attachments/assets/b7622306-89b9-491c-bb02-353c4790bc95" />

```python
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

<img width="689" height="87" alt="image" src="https://github.com/user-attachments/assets/bdb385bf-869a-45b1-8691-b21476d5be3a" />

```python
pred=KNN_classifier.predict(test_x)
con=confusion_matrix(test_y, pred)
print(con)
```

<img width="415" height="73" alt="{48929B05-2E00-4038-B952-85DB8FBE28B0}" src="https://github.com/user-attachments/assets/39d08ca2-b8ff-4064-88e8-326d5e6fd5b1" />

```python
acc=accuracy_score(test_y,pred)
print(acc)
```

<img width="404" height="45" alt="{ABCE1E02-ADC8-4DBE-8B1C-05694562A2EE}" src="https://github.com/user-attachments/assets/e13082a4-5082-448d-842d-073f7e6ca71b" />

```python
print("Misclassified Samples : %d" % (test_y !=pred).sum())
```

<img width="571" height="41" alt="{12F22301-AC4F-449A-8B59-A03287D3350F}" src="https://github.com/user-attachments/assets/ab07406f-367b-48b1-ac55-dbe3a4d100da" />

```python
df.shape
```

<img width="288" height="48" alt="{E066439E-C5E9-4A87-BFBB-76DA557FCF68}" src="https://github.com/user-attachments/assets/27ff6a09-99aa-40fd-87f8-c7dea5aecdb6" />

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 }
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="1772" height="121" alt="{B57B29EC-73A5-4D82-A9AF-3C12E6ED294A}" src="https://github.com/user-attachments/assets/9053555e-a5b8-46d5-8eb2-1e495fdf1da9" />

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="856" height="252" alt="{F79EFBE3-F12B-436F-8E3B-5B12DFA0C848}" src="https://github.com/user-attachments/assets/dbdb29ec-9b4f-42d6-a016-a83f26339339" />

```python
tips.time.unique()
```

<img width="709" height="77" alt="{083F8C69-5A12-41B6-BBCE-0DB0E2A39B13}" src="https://github.com/user-attachments/assets/10082af3-3791-417c-ae3b-282996dc4eba" />

```python
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="642" height="130" alt="{20FD970C-6876-4972-9045-EA9D0C214B52}" src="https://github.com/user-attachments/assets/d9c5dd93-7278-4b81-a201-a1d2bb27f5c1" />

```python
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="617" height="71" alt="{B8EB922F-CC9F-462C-8F14-8FD047201776}" src="https://github.com/user-attachments/assets/abe70fec-7df9-4389-b941-d5b7ea77e8ac" />


# RESULT:
       # INCLUDE YOUR RESULT HERE
