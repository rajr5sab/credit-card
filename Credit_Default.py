# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 22:47:04 2020

@author: P K T Raja Sabaresh
"""
"""
Variables
There are 25 variables:

* ID: ID of each client
* LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
* SEX: Gender (1=male, 2=female)
* EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
* MARRIAGE: Marital status (1=married, 2=single, 3=others)
* AGE: Age in years
* PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
* PAY_2: Repayment status in August, 2005 (scale same as above)
* PAY_3: Repayment status in July, 2005 (scale same as above)
* PAY_4: Repayment status in June, 2005 (scale same as above)
* PAY_5: Repayment status in May, 2005 (scale same as above)
* PAY_6: Repayment status in April, 2005 (scale same as above)
* BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
* BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
* BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
* BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
* BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
* BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
* PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
* PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
* PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
* PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
* PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
* PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
* default.payment.next.month: Default payment (1=yes, 0=no)
"""

## Credit card default dataset
# Importing Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import data
data = pd.read_csv(r"C:\Users\Admin\Downloads\data science\Classification Dataset\Credit default\UCI_Credit_Card.csv")

# Dataset Columns
data.columns
# ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
#        'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
#        'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
#        'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
#        'default.payment.next.month']

# See first 5 observation
data.head()

# Checking for Missing values
data.info()
#  #   Column                      Non-Null Count  Dtype  
# ---  ------                      --------------  -----  
#  0   ID                          30000 non-null  int64  
#  1   LIMIT_BAL                   30000 non-null  float64
#  2   SEX                         30000 non-null  int64  
#  3   EDUCATION                   30000 non-null  int64  
#  4   MARRIAGE                    30000 non-null  int64  
#  5   AGE                         30000 non-null  int64  
#  6   PAY_0                       30000 non-null  int64  
#  7   PAY_2                       30000 non-null  int64  
#  8   PAY_3                       30000 non-null  int64  
#  9   PAY_4                       30000 non-null  int64  
#  10  PAY_5                       30000 non-null  int64  
#  11  PAY_6                       30000 non-null  int64  
#  12  BILL_AMT1                   30000 non-null  float64
#  13  BILL_AMT2                   30000 non-null  float64
#  14  BILL_AMT3                   30000 non-null  float64
#  15  BILL_AMT4                   30000 non-null  float64
#  16  BILL_AMT5                   30000 non-null  float64
#  17  BILL_AMT6                   30000 non-null  float64
#  18  PAY_AMT1                    30000 non-null  float64
#  19  PAY_AMT2                    30000 non-null  float64
#  20  PAY_AMT3                    30000 non-null  float64
#  21  PAY_AMT4                    30000 non-null  float64
#  22  PAY_AMT5                    30000 non-null  float64
#  23  PAY_AMT6                    30000 non-null  float64
#  24  default.payment.next.month  30000 non-null  int64  

# Since there is no missing values, straight away going to visualization
# Categorical Variables Description
data[['SEX','EDUCATION','MARRIAGE']].describe()
#                 SEX     EDUCATION      MARRIAGE
# count  30000.000000  30000.000000  30000.000000
# mean       1.603733      1.853133      1.551867
# std        0.489129      0.790349      0.521970
# min        1.000000      0.000000      0.000000
# 25%        1.000000      1.000000      1.000000
# 50%        2.000000      2.000000      2.000000
# 75%        2.000000      2.000000      2.000000
# max        2.000000      6.000000      3.000000
# Here is no missing values but some unspecifiesd labels available
# EDUCATION - there is no description available for 5 & 6, 0 is unlabelled
# MARRIAGE - has a label 0 that is undocumented

# Rename PAY_0 as PAY_1 and target(default.payment.next.month) as target
data = data.rename(columns={'PAY_0':'PAY_1', 
                            'default.payment.next.month':'target'})
# Drop the unwanted ID variable
data = data.drop("ID", axis=1)

data.columns
# ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
#        'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
#        'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
#        'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'target']

# The Default probability is
data.target.sum()/len(data.target)
# 0.2212

# Frequency of the target variable
print(data.groupby('target').size())
# target
# 0    23364
# 1     6636
print(data.index)
# Calculating Utilization of credit card
for i in data.index:
    data['Utilization']

# Univariate EDA Numeric Variable
# Age
plt.style.use('ggplot')
data.AGE.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Age Distribution of this Credit Card base')
plt.xlabel('Age band')
plt.ylabel('# Applicants')
plt.grid(axis='y', alpha=0.75)
plt.show()

# LIMIT_BAL
data.LIMIT_BAL.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.xlabel('Balance Limit')
plt.ylabel('# Applicants')
plt.title('Balance Limit')
plt.grid(axis='y', alpha=0.75)
plt.show()

# BILL_AMT1
data.BILL_AMT1.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Bill Amount in Sept')
plt.xlabel('Bill Amount in sept')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# BILL_AMT2
data.BILL_AMT2.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Bill Amount in Aug')
plt.xlabel('Bill Amount in Aug')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# BILL_AMT3
data.BILL_AMT3.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Bill Amount in July')
plt.xlabel('Bill Amount in July')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# BILL_AMT4
data.BILL_AMT4.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Bill Amount in June')
plt.xlabel('Bill Amount in June')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# BILL_AMT5
data.BILL_AMT5.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Bill Amount in May')
plt.xlabel('Bill Amount in May')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# BILL_AMT6
data.BILL_AMT6.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Bill Amount in April')
plt.xlabel('Bill Amount in April')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# PAY_AMT1
data.PAY_AMT1.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('PAY Amount in Sept')
plt.xlabel('PAY Amount in sept')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# PAY_AMT2
data.PAY_AMT2.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('PAY Amount in Aug')
plt.xlabel('PAY Amount in Aug')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# PAY_AMT3
data.PAY_AMT3.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('PAY Amount in July')
plt.xlabel('PAY Amount in July')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# PAY_AMT4
data.PAY_AMT4.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('PAY Amount in June')
plt.xlabel('PAY Amount in June')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# PAY_AMT5
data.PAY_AMT5.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('PAY Amount in May')
plt.xlabel('PAY Amount in May')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# PAY_AMT6
data.PAY_AMT6.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('PAY Amount in April')
plt.xlabel('PAY Amount in April')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Bivariate EDA Categorical variables
data.SEX.unique()
# array([2, 1], dtype=int64)
data.EDUCATION.unique()
# array([2, 1, 3, 5, 4, 6, 0], dtype=int64)
data.MARRIAGE.unique()
# array([1, 2, 3, 0], dtype=int64)

