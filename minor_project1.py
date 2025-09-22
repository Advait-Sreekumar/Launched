import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel('CarSales.xlsx')




dups=df.duplicated()         #Deleting duplicates
df.drop_duplicates(inplace=True)

df.boxplot(column=['Day'])        #Showing boxplot
plt.show()

def outlier(col):             #Code for removing outlier
    sorted(col)
    q1,q3=col.quantile([0.25,0.75])
    iqr=q3-q1
    lower_range=q1-(1.5*iqr)
    higher_range=q3+(1.5*iqr)
    return lower_range,higher_range

# lr_price,hi_price=outlier(df['Day'])
# df['Day']=np.where(df['Day']<lr_price,lr_price,df['Day'])
# df['Day']=np.where(df['Day']>hi_price,hi_price,df['Day'])


df.boxplot(column=['Day'])        #Showing boxplot
plt.show()


print(df.isnull().sum())

median_price=df['Price'].median()   #Replacing numerical empty value


df=df.dropna(axis=1)   #Deletes rows with null values


mode_color=df['Color'].mode().values[0]                     #Clearing out null data and normalization
df['Color']=df['Color'].replace(np.nan,mode_color)
df=df.dropna(axis=1)
print(df.isnull().sum())


#Univeriate analysis
sns.displot(df['Day'])
plt.show()

#Bivariate analysis
sns.pairplot(df)
plt.show()
