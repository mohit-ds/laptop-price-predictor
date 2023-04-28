#!/usr/bin/env python
# coding: utf-8

# # Laptop Price Predictor - Kaggle

# In[1]:


# Importin all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('laptop_data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.duplicated().sum()


# In[7]:


df.isnull().sum()


# In[8]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# In[9]:


df.head()


# In[10]:


df['Ram']=df['Ram'].str.replace('GB','')
df['Weight']=df['Weight'].str.replace('kg','')


# In[11]:


df.head()


# In[12]:


df['Ram']=df['Ram'].astype('int32')
df['Weight']=df['Weight'].astype('float32')


# In[13]:


df.info()


# In[14]:


sns.distplot(df['Price'])


# In[15]:


df.Company.value_counts().plot(kind='bar')


# - Dell,Lenovo and HP are the top3 most selling laptop brands.

# In[16]:


plt.rcParams['figure.figsize']=(16,4)

sns.barplot(x=df['Company'],y=df['Price'])


# - Razer, Google and LG are the top 3 most expensive laptop brands.

# In[17]:


df['TypeName'].value_counts().plot(kind='bar')
plt.rcParams['figure.figsize']=(12,6)


# - Notebook is the most sold type of laptop
# - Netbook is the least sold type of Laptop

# In[18]:


plt.rcParams['figure.figsize']=(12,6)

sns.barplot(x=df['TypeName'],y=df['Price'])


# In[19]:


df.ScreenResolution.value_counts()


# In[20]:


df.head()


# In[21]:


df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[22]:


df['Touchscreen'].value_counts()


# In[23]:


df.sample(9)


# In[24]:


df['Touchscreen'].value_counts().plot(kind='bar')


# In[25]:


plt.rcParams['figure.figsize']=(12,6)

sns.barplot(x=df['Touchscreen'],y=df['Price'])


# - It concludes that Touchscreen laptops are much expensive than others.

# In[26]:


df.head()


# In[27]:


new=df['ScreenResolution'].str.split('x',n=1,expand=True)


# In[28]:


df['X_res']=new[0]
df['Y_res']=new[1]


# In[29]:


df.head()


# In[30]:


df['X_res']=df['X_res'].str.strip('IPS Panel Retina Display').str.strip('Full HD').str.strip('Quad HD+ / Touchscreen')


# In[31]:


df.head()


# In[32]:


df['X_res'].unique()


# In[33]:


df['X_res']=df['X_res'].str.strip('4K Ultra HD / Touchscreen').str.strip('Full HD')


# In[34]:


df['X_res'].unique()


# In[35]:


df['X_res']=df['X_res'].astype('int')
df['Y_res']=df['Y_res'].astype('int')


# In[36]:


df.info()


# In[37]:


df['ppi']=(((df['X_res']**2)+ (df['Y_res']**2))**0.5/df['Inches']).astype('float') # ppi-pixels per inches


# In[38]:


df['ppi']=(((df['X_res']**2)+ (df['Y_res']**2))**0.5/df['Inches']).astype('float').astype('float')


# In[39]:


df.corr()['Price']


# In[40]:



df.drop('Inches',axis=1,inplace=True)
df.drop('X_res',axis=1,inplace=True)
df.drop('Y_res',axis=1,inplace=True)


# In[41]:


df.head()


# In[42]:


df['Cpu'].value_counts()


# In[43]:


df['Cpu Name']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[44]:


df.head()


# In[45]:


def processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# In[46]:


df['Cpu brand']=df['Cpu Name'].apply(processor)


# In[47]:


df.head()


# In[48]:


df['Cpu brand'].value_counts()


# In[49]:


df['Cpu brand'].value_counts().plot(kind='bar')


# In[50]:


plt.rcParams['figure.figsize']=(12,6)

sns.barplot(x=df['Cpu brand'],y=df['Price'])


# In[51]:


df.drop(columns=['Cpu','Cpu Name'],axis=1,inplace=True)


# In[52]:


df.head()


# In[53]:


df.Ram.value_counts().plot(kind='bar')


# - 8Gb is the most preffered Ram followed by 4 and 16.
# - 64 gb is the least preffered Ram.

# In[54]:




sns.barplot(x=df['Ram'],y=df['Price'])


# - It is a linear relationship as the ram size increases price also increases

# In[55]:


df.Memory.value_counts()


# In[56]:


df['Memory']=df['Memory'].astype(str).replace('\.0','',regex=True)


# In[57]:


df['Memory']=df['Memory'].str.replace('GB','')


# In[58]:


df['Memory']=df['Memory'].str.replace('TB','000')    # as 1tb is 1000


# In[59]:


new=df['Memory'].str.split('+',n=1,expand= True)  # here we split the combinations of memory into 2


# In[60]:


new.value_counts()


# In[61]:


df['first']=new[0]


# In[62]:


df['first']=df['first'].str.strip()


# In[63]:


df['second']=new[1]


# In[64]:


df.second.value_counts()


# In[65]:


df['Layer1HDD']=df['first'].apply(lambda x: 1 if 'HDD' in x else 0)
df['Layer1SSD']=df['first'].apply(lambda x: 1 if 'SSD' in x else 0)
df['Layer1Hybrid']=df['first'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer1Flash_Storage']=df['first'].apply(lambda x: 1 if 'Flash Storage' in x else 0)


# In[66]:


df['first'].value_counts()


# In[67]:


df['first']=df['first'].str.replace(r'\D','')


# In[68]:


df['second'].isnull().sum()    # there are 1095 null rows we need to fill them


# In[69]:


df['second'].fillna('0',inplace=True)


# In[70]:


df['second'].isnull().sum() 


# In[71]:


# now we need to repeat the process for the second split


# In[72]:


df['Layer2HDD']=df['second'].apply(lambda x: 1 if 'HDD' in x else 0)
df['Layer2SSD']=df['second'].apply(lambda x: 1 if 'SSD' in x else 0)
df['Layer2Hybrid']=df['second'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer2Flash_Storage']=df['second'].apply(lambda x: 1 if 'Flash Storage' in x else 0)


# In[73]:


df['second']=df['second'].str.replace(r'\D','')


# In[74]:


df['first']=df['first'].astype(int)
df['second']=df['second'].astype(int)


# In[75]:


df['HDD']=(df['first']*df['Layer1HDD']+df['second']*df['Layer2HDD'])
df['SSD']=(df['first']*df['Layer1SSD']+df['second']*df['Layer2SSD'])
df['Hybrid']=(df['first']*df['Layer1Hybrid']+df['second']*df['Layer2Hybrid'])
df['Flash_Storage']=(df['first']*df['Layer1Flash_Storage']+df['second']*df['Layer2Flash_Storage'])


# In[79]:


# now we drop the columns that we created for the above code
df.drop(columns=['first','second','Layer1HDD','Layer1SSD','Layer1Hybrid','Layer1Flash_Storage',
                'Layer2HDD','Layer2SSD','Layer2Hybrid','Layer2Flash_Storage'],inplace=True)


# In[80]:


df.sample(5)


# In[82]:


df.drop('Memory',axis=1,inplace=True)


# In[83]:


df.head()


# In[84]:


df.corr()['Price']


# - Hybrid and Flash storage is not strongly correlated to Price hence we drop it 

# In[85]:


df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)


# In[86]:


df.head()


# In[88]:


df.Gpu.value_counts()


# In[92]:


df['Gpu brand']=df['Gpu'].apply(lambda x: x.split()[0])


# In[94]:


df.head()


# In[95]:


df['Gpu brand'].value_counts()


# In[100]:


df=df[df['Gpu brand'] != 'ARM']


# In[101]:


df['Gpu brand'].value_counts()


# In[103]:


sns.barplot(x=df['Gpu brand'],y=df['Price'])


# - Nvidia is the most used Graphic card

# In[104]:


df.drop('Gpu',axis=1,inplace=True)


# In[105]:


df.head()


# In[106]:


df.drop('ScreenResolution',axis=1,inplace=True)


# In[108]:


df.OpSys.value_counts()


# In[111]:


def cat(os):
    if os == 'Windows 10' or os == 'Windows 7' or os== 'Windows 10 S':
        return 'Windows'
    elif os == 'macOS' or os=='Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[112]:


df['os']=df['OpSys'].apply(cat)


# In[113]:


df.head()


# In[114]:


df.drop('OpSys',axis=1,inplace=True)


# In[115]:


sns.barplot(x=df['os'],y=df['Price'])


# - this analysis shows that Mac os is the most expensive os.

# In[116]:


sns.heatmap(df.corr())


# In[118]:


sns.distplot(df['Price'])


# In[119]:


sns.distplot(np.log(df['Price']))


# In[121]:


X= df.drop(columns=['Price'])
y= np.log(df['Price'])


# In[122]:


X


# In[123]:


y


# In[124]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[133]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error


# In[156]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# In[138]:


step1= ColumnTransformer(transformers=[
      ('col_tnf',OneHotEncoder(sparse=False,drop='first'),['Company','TypeName','Cpu brand','Gpu brand','os'])  # columns on which we want to apply OHe
],remainder = 'passthrough')   # remainder = passthrough as we dont want any changes on other numerical columns

step2 = LinearRegression()

pipe= Pipeline([('step1',step1),
                ('step2',step2)
               ])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[146]:


step1= ColumnTransformer(transformers=[
      ('col_tnf',OneHotEncoder(sparse=False,drop='first'),['Company','TypeName','Cpu brand','Gpu brand','os'])  
],remainder = 'passthrough')   

step2 = KNeighborsRegressor(n_neighbors=5)

pipe= Pipeline([('step1',step1),
                ('step2',step2)
               ])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[150]:


step1= ColumnTransformer(transformers=[
      ('col_tnf',OneHotEncoder(sparse=False,drop='first'),['Company','TypeName','Cpu brand','Gpu brand','os'])  
],remainder = 'passthrough')   

step2 = DecisionTreeRegressor(max_depth=8)

pipe= Pipeline([('step1',step1),
                ('step2',step2)
               ])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[153]:


step1= ColumnTransformer(transformers=[
      ('col_tnf',OneHotEncoder(sparse=False,drop='first'),['Company','TypeName','Cpu brand','Gpu brand','os'])  
],remainder = 'passthrough')   

step2 = SVR(kernel='rbf',epsilon=0.1,C=10000)

pipe= Pipeline([('step1',step1),
                ('step2',step2)
               ])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[169]:


step1= ColumnTransformer(transformers=[
      ('col_tnf',OneHotEncoder(sparse=False,drop='first'),['Company','TypeName','Cpu brand','Gpu brand','os'])  
],remainder = 'passthrough')   

step2 = RandomForestRegressor(n_estimators=100,random_state=42,max_depth=15)


pipe= Pipeline([('step1',step1),
                ('step2',step2)
               ])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[160]:


step1= ColumnTransformer(transformers=[
      ('col_tnf',OneHotEncoder(sparse=False,drop='first'),['Company','TypeName','Cpu brand','Gpu brand','os'])  
],remainder = 'passthrough')   

step2 = GradientBoostingRegressor(n_estimators=500,learning_rate=0.1)

pipe= Pipeline([('step1',step1),
                ('step2',step2)
               ])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[165]:


step1= ColumnTransformer(transformers=[
      ('col_tnf',OneHotEncoder(sparse=False,drop='first'),['Company','TypeName','Cpu brand','Gpu brand','os'])  
],remainder = 'passthrough')   



step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

pipe= Pipeline([('step1',step1),
                ('step2',step2),
                
               ])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




