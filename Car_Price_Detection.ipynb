#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction using Regression

# ### Import necessary libraries

# In[71]:


import pandas as pd #dataframe
import numpy as np #mathematical computations
import matplotlib.pyplot as plt #visualization
import matplotlib
import seaborn as sns #visualization
import json #exporting columns
import pickle #saving the model
from sklearn.linear_model import LinearRegression #Linear Regression
from sklearn.linear_model import Lasso #Lasso Regression
from sklearn.tree import DecisionTreeRegressor #Decision Tree Regression
from sklearn.ensemble import RandomForestRegressor #Random Forest Regression
from sklearn.model_selection import train_test_split #Splitting the dataset into training and testing
from sklearn.model_selection import ShuffleSplit #Random shuffling
from sklearn.model_selection import cross_val_score #Score cross validation
from sklearn.model_selection import GridSearchCV #Hyper parameter tuning
from warnings import simplefilter #Filtering warnings

simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load data into DataFrame

# In[80]:


df1 = pd.read_csv('car dataset.csv', dtype={'Year': int})
df1.head()


# In[81]:


# shape
df1.shape


# In[82]:


# data columns
df1.columns


# ### Dropping feature that is not required in building our model

# In[83]:


# I also consider reindexing the columns, putting the dependent variable at the end
# Dropping the name of the car, the price in ruppes and seller type
# Retained the columns transmission,number of previous owner, fuel, year and kilometer driven for these features might greatly affect the selling price

df1 = df1.drop(['Car_Name', 'Seller_Type', 'Original_Price'], axis=1)

# Reindexing the columns
df2 = df1[['Transmission', 'Fuel_Type', 'Manufacturer','Kms_Driven', 'Year', 'Selling_Price','Owner']]

df2.head()


# ### Check for null values

# In[84]:


# method 1
df2.isnull().sum()


# In[86]:


# method 2
df2.isnull().any()


# ### Check for data consistency

# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
# You might want to check the integrity of the data whether the columns contains mispelled data or that data pertains to same idea.
# </div>

# In[87]:


df2.Transmission.unique()


# In[88]:


df2.Fuel_Type.unique()


# In[89]:


df2.Year.unique()


# ### Adding new feature

# In[95]:


# adding a converted price for philippine peso

df2['Selling_Price_Birr'] = round(df2['Selling_Price'] * 2250,2)

# dropping selling price in thousand rdollars

df3 = df2.drop('Selling_Price',axis=1)
df3.head()


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
# Let see the data distribution based on each features.
# </div>

# ### Exploratory Data Analysis

# ### Manual and Automatic Cars

# In[98]:


sns.pairplot(df3,hue = 'Transmission',diag_kind = "kde",kind = "scatter",palette = "husl",height=3.5)
plt.show()


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
#     We can see that automatic cars have higher price range than manual type cars though the distribution of automatic cars were skewed to the right. We can also see the increase of automatic cars between 2015 and 2020. In this instance, we can clearly see some outliers in selling price and kilometers driven.
# </div>

# ### Fuel types

# In[100]:


df3['Fuel_Type'].value_counts()


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
# Since most of the data points are Diesel and Petrol, we can label other fuel types as Others.
# </div>

# In[103]:


df4 = df3
df4.fuel = df4.Fuel_Type.map(lambda x: x if x in ['Diesel','Petrol'] else 'Other')
df4.fuel.unique()


# In[104]:


manual = df4[df4['Transmission']=='Manual']
automatic = df4[df4['Transmission']=='Automatic']


# In[105]:


print('Manual type car')
sns.pairplot(manual,hue = 'Fuel_Type',diag_kind = "kde",kind = "scatter",palette = "husl",height=3.5)
plt.show()


# In[107]:


print('Automatic type car')
sns.pairplot(automatic,hue = 'Fuel_Type',diag_kind = "kde",kind = "scatter",palette = "husl",height=3.5)
plt.show()


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
#     We can see that the selling price of diesel type cars in both manual and automatic were more spread than petrol and other fuel hence getting higher average and range of selling price.
# </div>

# ### Owner

# In[108]:


df4.groupby('Owner')[['Kms_Driven','Selling_Price_Birr']].agg(['count','mean']).applymap(lambda x: format(x,'.0f'))


# In[110]:


# we will be dropping test drive cars and will merge third owner and 
# forth & above owner as third owner and above

df5 = df4[~(df4['Owner']=='Test Drive Car')]
df5.Owner = df5.Owner.map(lambda x: x if x in ['First Owner','Second Owner'] else 'Third Owner & Above')

df5['Owner'].value_counts()


# In[111]:


sns.pairplot(df5,hue = 'Owner',diag_kind = "kde",kind = "scatter",palette = "husl",height=3.5)
plt.show()


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
#     Cars sold by first owner have higher prices than the rest. Test drive cars tends to have a higher price as as well though there is a distinction between test drive cars in manual and automatic.
# </div>

# ### Year and Kilometer Driven

# In[112]:


km_mean = df5.groupby('Year')['Kms_Driven'].mean()


# In[114]:


fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(16,8))

ax[0].bar(km_mean.index,km_mean)
sns.distplot(manual['Kms_Driven'],ax=ax[1])
sns.distplot(automatic['Kms_Driven'],ax=ax[1])

ax[0].set_title('Average kilometer driven each year')
ax[0].set_xlabel('Kilometer Driven')
ax[0].set_ylabel('Year')

ax[1].set_title('Kilometers driven distribution')
ax[1].legend(['Manual','Automatic'])

plt.show()


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
#     We can see that average kilometers driven rises up from 1995 until 2005 and linearly goes down until 2020. We can also see some outliers present in the distribution plot.
# </div>

# ### Year and Selling Price

# In[116]:


year_mean_manual = df5[df5['Transmission']=='Manual'].groupby('Year')['Selling_Price_Birr'].mean()
year_mean_automatic = df5[df5['Transmission']=='Automatic'].groupby('Year')['Selling_Price_Birr'].mean()


# In[123]:


fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,8))

ax[0].bar(year_mean_manual.index,year_mean_manual)
ax[1].bar(year_mean_automatic.index,year_mean_automatic)

ax[0].set_title('Average Selling Price of Manual Cars every Year')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('Selling Price')

ax[1].set_title('Average Selling Price of Automatic Cars every Year')
ax[1].set_xlabel('Year')
ax[1].set_ylabel('Selling Price')

plt.show()


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
#     We can see that selling price of manual cars grows linearly each year whereas automatic cars have wavy averages in each year but we can clearly see that selling price grows linearly as well. 
# </div>	

# ### Removing Outliers

# In[126]:


df5.groupby('Transmission').agg(['mean','std','min','max']).applymap(lambda x: format(x,'.0f')).drop(['Year'],axis=1)


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
# Clearly we have some outliers present in kilometers driven and selling price. We need to remove these outliers by using IQR method. 
# </div>	

# In[129]:


def remove_outlier_km_driven(df):
    temp = pd.DataFrame()
    
    df_km_driven = df['Kms_Driven']
    Q1 = df_km_driven.quantile(0.25)
    Q3 = df_km_driven.quantile(0.75)
    IQR = Q3 - Q1
    df_outlier = df_km_driven[(df_km_driven < (Q1 - 1.5 * IQR)) | (df_km_driven > (Q3 + 1.5 * IQR))]
    temp = pd.concat([temp,df_outlier])
        
    return df.drop(temp.index)

df6 = remove_outlier_km_driven(df5)


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
# Removing outliers in selling price would be separated by transmission type and will be done in each year.
# </div>	

# In[137]:


def remove_outlier_selling_price(df):
    temp = pd.DataFrame()
    for year in sorted(df.Year.unique()):
        year_price_manual = df[(df['Year']==year) & (df['Transmission']=='Manual')]['Selling_Price_Birr']
        manual_Q1 = year_price_manual.quantile(0.25)
        manual_Q3 = year_price_manual.quantile(0.75)
        manual_IQR = manual_Q3 - manual_Q1
        outlier_manual = year_price_manual[(year_price_manual < (manual_Q1 - 1.5 * manual_IQR)) | (year_price_manual > (manual_Q3 + 1.5 * manual_IQR))]
        temp = pd.concat([temp,outlier_manual])
        
        year_price_automatic = df[(df['Year']==year) & (df['Transmission']=='Automatic')]['Selling_Price_Birr']
        automatic_Q1 = year_price_automatic.quantile(0.25)
        automatic_Q3 = year_price_automatic.quantile(0.75)
        automatic_IQR = automatic_Q3 - automatic_Q1
        outlier_automatic = year_price_automatic[(year_price_automatic < (automatic_Q1 - 1.5 * automatic_IQR)) | (year_price_automatic > (automatic_Q3 + 1.5 * automatic_IQR))]
        temp = pd.concat([temp,outlier_automatic])
    return df.drop(temp.index)

df7 = remove_outlier_selling_price(df6)


# In[139]:


year_mean_manual = df7[df7['Transmission']=='Manual'].groupby('Year')['Selling_Price_Birr'].mean()
year_mean_automatic = df7[df7['Transmission']=='Automatic'].groupby('Year')['Selling_Price_Birr'].mean()


# In[140]:


fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(16,4))

ax[0].bar(year_mean_manual.index,year_mean_manual)
ax[1].bar(year_mean_automatic.index,year_mean_automatic)

ax[0].set_title('Average Selling Price of Manual Cars every Year')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('Selling Price')

ax[1].set_title('Average Selling Price of Automatic Cars every Year')
ax[1].set_xlabel('Year')
ax[1].set_ylabel('Selling Price')

plt.show()


# In[141]:


# we can also safely remove the datapoints before 2005 since it produces inconsistency and the data points
# below 2000 have low value count
df8 = df7[df7['Year']>2005]


# In[143]:


year_mean_manual_price = df8[df8['Transmission']=='Manual'].groupby('Year')['Selling_Price_Birr'].mean()
year_mean_automatic_price = df8[df8['Transmission']=='Automatic'].groupby('Year')['Selling_Price_Birr'].mean()
year_mean_manual_km = df8[df8['Transmission']=='Manual'].groupby('Year')['Kms_Driven'].mean()
year_mean_automatic_km = df8[df8['Transmission']=='Automatic'].groupby('Year')['Kms_Driven'].mean()


# In[144]:


fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(16,8))

ax[0][0].bar(year_mean_manual_price.index,year_mean_manual_price)
ax[0][1].bar(year_mean_automatic_price.index,year_mean_automatic_price)

ax[1][0].bar(year_mean_manual_km.index,year_mean_manual_km)
ax[1][1].bar(year_mean_automatic_km.index,year_mean_automatic_km)

ax[0][0].set_title('Average Selling Price of Manual Cars every Year')
ax[0][0].set_ylabel('Selling Price')

ax[0][1].set_title('Average Selling Price of Automatic Cars every Year')
ax[0][1].set_ylabel('Selling Price')

ax[1][0].set_title('Average Kilometer driven of Manual Cars every Year')
ax[1][0].set_ylabel('Kilometer driven')

ax[1][1].set_title('Average Kilometer driven of Automatic Cars every Year')
ax[1][1].set_ylabel('Kilometer driven')

plt.show()


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
# Now we will be removing some suspicious data points. Since the average selling price of cars increases each year whereas average kilometer driven by the car should be atleast lower than the average kilometer driven last year, we will be removing manual and automatic cars which kilometer driven is greater than the average kilometer driven last year but having lower selling price compared to the average price last year.
# </div>	

# In[146]:


def remove_outlier_last_year(df):
    temp = pd.DataFrame()
    years = sorted(df.Year.unique())
    for i in range(1,len(years)):
        df_year = df[(df['Year']==years[i])&(df['Transmission']=='Manual')]
        last_mean_km_driven = df[(df['Year']==years[i-1])&(df['Transmission']=='Manual')]['Kms_Driven'].mean()
        last_mean_selling_price = df[(df['Year']==years[i-1])&(df['Transmission']=='Manual')]['Selling_Price_Birr'].mean() 
        df_outlier = df_year[(df_year['Kms_Driven']>last_mean_km_driven)&(df_year['Selling_Price_Birr']<last_mean_selling_price)]
        temp = pd.concat([temp,df_outlier])
        
        df_year = df[(df['Year']==years[i])&(df['Transmission']=='Automatic')]
        last_mean_km_driven = df[(df['Year']==years[i-1])&(df['Transmission']=='Automatic')]['Kms_Driven'].mean()
        last_mean_selling_price = df[(df['Year']==years[i-1])&(df['Transmission']=='Automatic')]['Selling_Price_Birr'].mean() 
        df_outlier = df_year[(df_year['Kms_Driven']>last_mean_km_driven)&(df_year['Selling_Price_Birr']<last_mean_selling_price)]
        temp = pd.concat([temp,df_outlier]) 
    return df.drop(temp.index)
    
df9 = remove_outlier_last_year(df8)


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
# Now we might also consider removing some inconsistencies in selling price based on number of previous owner. If the second owner offers lower selling price than the average selling price of third owner, we will remove this data points. We will also do this with first and second owners based on year.
# </div>	

# In[152]:


def remove_outlier_owner(df):
    temp = pd.DataFrame()
    for a in sorted(df.Year.unique()):
        for b in ['Manual','Automatic']:
            df_year = df[(df['Year']==a)&(df['Transmission']==b)]
            second_mean = df_year[df_year['Owner']=='Second Owner']['Selling_Price_Birr'].mean()
            third_mean = df_year[df_year['Owner']=='Third Owner & Above']['Selling_Price_Birr'].mean()
            df_outlier = df_year[((df_year['Owner']=='Second Owner')&(df_year['Selling_Price_Birr']<third_mean)) | ((df_year['Owner']=='First Owner')&(df_year['Selling_Price_Birr']<second_mean))]
            temp = pd.concat([temp,df_outlier])
    return df.drop(temp.index)
    
df10 = remove_outlier_owner(df9)


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
# We will also remove data points with low kilometers driven but also having low selling price. We will be removing data points with kilometers driven one standard deviation below the mean and at the same time the selling price below one standard deviation below the mean price. We will be removing data points high kilometers driven and high selling price as well.
# </div>	

# In[156]:


def remove_outlier_last_year(df):
    temp = pd.DataFrame()
    for i in sorted(df.Year.unique()):
        df_year = df[df['Year']==i]
        km = df_year['Kms_Driven']
        price = df_year['Selling_Price_Birr']
        outlier = df_year[(df_year['Kms_Driven']<km.mean()-km.std())&(df_year['Selling_Price_Birr']<price.mean()-price.std()) | (df_year['Kms_Driven']>km.mean()+km.std())&(df_year['Selling_Price_Birr']>price.mean()+price.std())]
        temp = pd.concat([temp,outlier])

    return df.drop(temp.index)
    
df11 = remove_outlier_last_year(df10)


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
#     Now we successfully removed outliers.
# </div>	

# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
#     Now lets see our data in 3d.
# </div>	

# In[158]:


manual = df11[df11['Transmission']=='Manual']
automatic = df11[df11['Transmission']=='Automatic']


# In[160]:


from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax=fig.add_axes([0.2,0,1.5,1.5], projection='3d')

ax.scatter(manual['Selling_Price_Birr'],manual['Year'],manual['Kms_Driven'])
ax.scatter(automatic['Selling_Price_Birr'],automatic['Year'],automatic['Kms_Driven'])

ax.set_xlabel('selling price')
ax.set_ylabel('year')
ax.set_zlabel('km driven')

ax.legend(['Manual','Automatic'])

ax=fig.add_axes([0,-1,1,1], projection='3d')

ax.scatter(manual['Kms_Driven'],manual['Selling_Price_Birr'],manual['Year'])
ax.scatter(automatic['Kms_Driven'],automatic['Selling_Price_Birr'],automatic['Year'])

ax.set_xlabel('km driven')
ax.set_ylabel('selling price')
ax.set_zlabel('year')

ax.legend(['Manual','Automatic'])

ax=fig.add_axes([1,-1,1,1], projection='3d')

ax.scatter(manual['Year'],manual['Selling_Price_Birr'],manual['Kms_Driven'])
ax.scatter(automatic['Year'],automatic['Selling_Price_Birr'],automatic['Kms_Driven'])

ax.set_xlabel('year')
ax.set_ylabel('selling price')
ax.set_zlabel('km driven')

ax.legend(['Manual','Automatic'])

plt.show()


# In[163]:


sns.pairplot(df11,hue='Transmission',diag_kind = "kde",kind = "scatter",palette = "husl",height=3.5)
plt.show()


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
#     As you can see in the 3d scatter plot, kilometers driven decreases whereas selling price increases over the years. We can also see that automatic cars have higher prices than manual cars. Now it's time to build our model.
# </div>	

# In[164]:


df11.shape


# In[170]:


transmission = pd.get_dummies(df11['Transmission'])
fuel = pd.get_dummies(df11['Fuel_Type'])
owner = df11['Owner'].map(lambda x: 1 if x=='First Owner' else 2 if x=='Second Owner' else 3)

X = pd.concat([transmission,fuel,owner,df11.drop(['Transmission','Fuel_Type','Owner','Selling_Price_Birr','Manufactur'],axis=1)],axis=1)
y = df11['Selling_Price_Birr']


# In[171]:


X.head(5)


# In[205]:


y.head(5)


# In[206]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[207]:


lr = LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
y_pred = lr.predict(X_test)


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
# Use K Fold cross validation to measure accuracy of our LinearRegression model.
# </div>	

# In[208]:


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# In[210]:


lr.fit(X_train,y_train)


# In[226]:


def predict_price(transmission,fuel,previous_owner,year,km_driven):
    x = []
    x[:8] = np.zeros(8,dtype='int32')
    x[5] = previous_owner
    x[6] = year
    x[7] = km_driven
    
    transmission_index = np.where(X.columns==transmission)[0][0]
    fuel_index = np.where(X.columns==fuel)[0][0]
    
    if transmission_index>=0:
        x[transmission_index] = 1
    if fuel_index>=2:
        x[fuel_index] = 1
        
    return float(format(lr.predict([x])[0],'.2f'))


# # Using Test Datas to try our model
# 

# In[227]:


predict_price('Manual','Diesel',1,2013,10000)


# In[228]:


predict_price('Automatic','Diesel',1,2015,50000)


# In[229]:


predict_price('Manual','Diesel',1,2018,20000)


# In[230]:


predict_price('Manual','Diesel',1,2020,10000)


# <div style="background-color: #303030;padding: 20px;font-size: 16px;color:white;">
# Now we will be exporting our model into a pickle file
# </div>

# In[231]:


with open('used_car_price_model.pickle','wb') as f:
    pickle.dump(model,f)


# In[232]:


# try to load the model
with open('used_car_price_model.pickle','rb') as file:
    mp = pickle.load(file)


# In[233]:


# Manual, Diesel, Second Owner, year 2017, 30000 kilometers driven
lr.predict([[0,1,1,0,0,2,2017,30000]])


# In[234]:


columns = {
    "columns": [col for col in X.columns]
}

with open("columns.json","w") as f:
    f.write(json.dumps(columns))

