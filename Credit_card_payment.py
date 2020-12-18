# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:03:58 2020

@author: P K T Raja Sabaresh
"""

# Utilization and payment rates are added

# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing data
data = pd.read_csv(r"C:\Users\Admin\Downloads\data science\Classification Dataset\Credit default\UCI_Credit_Card_practice.csv")

# Checking for null values
data.isnull().sum()
data.info()


# Renaming the required Columns
data = data.rename(columns={'PAY_0':'PAY_1', 'default.payment.next.month':'target'})

# Droping ID variable  which is not required
data = data.drop('ID',axis=1)
data.columns
# ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
#        'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
#        'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
#        'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'target',
#        'Utilization1', 'Utilization2', 'Utilization3', 'Utilization4',
#        'Utilization5', 'Utilization6', 'Payment_Behavior_1',
#        'Payment_Behavior_2', 'Payment_Behavior_3', 'Payment_Behavior_4',
#        'Payment_Behavior_5']

# The Default probability is
data.target.sum()/len(data.target)
# 0.2212

# Frequency of target variable
data.groupby('target').size()
# target
# 0    23364
# 1     6636

data.groupby('SEX').size()
# SEX
# 1    11888
# 2    18112

# 1	Paid in full
# 2	Did not spend in the previous month
# 3	<25%
# 4	25% to <50%
# 5	50% to < 75%
# 6	>75%

# Univariate EDA - Categorical Variable
# Target
target_dictionary = {0:'Not Default', 1:'Default'}
data['target_up']= data['target'].map(target_dictionary )
count_target = data['target_up'].value_counts()
plt.bar(count_target.index,count_target.values,color = 'green')
plt.xlabel('Target')
plt.ylabel('# Applicants')
plt.title('Target Distribution')
plt.show()

# GENDER
gender_dictionary = {1:'Male', 2:'Female'}
data['gender_up']= data['SEX'].map(gender_dictionary)
count_Gender = data['gender_up'].value_counts()
plt.bar(count_Gender.index,count_Gender.values,color = 'green')
plt.xlabel('Gender')
plt.ylabel('# Applicants')
plt.title('Gender Distribution')
plt.show()

# EDUCATION
education_dictionary = {1:'Graduate', 2:'Univertsity', 3:'High School', 4:'others', 5:'Unknown', 6:'Unknown'}
data['education_up'] = data['EDUCATION'].map(education_dictionary)
count_Education = data['education_up'].value_counts()
plt.bar(count_Education.index,count_Education.values,color = 'green')
plt.xlabel('Education')
plt.ylabel('# Applicants')
plt.title('Education Distribution')
plt.show()

# Marriage
marriage_dictionary = {1:'Married', 2:'Single', 3:'Others'}
data['marriage_up'] = data['MARRIAGE'].map(marriage_dictionary)
count_Marriage = data['marriage_up'].value_counts(ascending=False)
plt.bar(count_Marriage.index,count_Marriage.values,color = 'green')
plt.xlabel('Marriage')
plt.ylabel('# Applicants')
plt.title('Marriage Distribution')
plt.show()

# Mapping Repayment status
pay_dict = {-2:'Pre duly', -1:'Pay duly', 0:'No delay', 1:'One month delay', 2:'2 Months', 3:'3 Months', 4:'4 Months' ,5:'5 Months', 6:'6 Months', 7:'7 Months', 8:'8 Months', 9:'9 Months'}
# PAY_1
data['PAY_1_up'] = data['PAY_1'].map(pay_dict)
count_PAY_1 = data['PAY_1_up'].value_counts(ascending=False)
plt.figure(figsize=(15,7.5), num=12, dpi=180)
plt.bar(count_PAY_1.index,count_PAY_1.values,color = 'green')
plt.xlabel('PAY_1')
plt.ylabel('# Applicants')
plt.title('Repayment status in september Distribution')
plt.show()

# PAY_2
data['PAY_2_up'] = data['PAY_2'].map(pay_dict)
count_PAY_2 = data['PAY_2_up'].value_counts(ascending=False)
plt.figure(figsize=(15,8), num=12, dpi=360)
plt.bar(count_PAY_2.index,count_PAY_2.values,color = 'green')
plt.xlabel('PAY_2')
plt.ylabel('# Applicants')
plt.title('Repayment status in August Distribution')
plt.show()

# PAY_3
data['PAY_3_up'] = data['PAY_3'].map(pay_dict)
count_PAY_3 = data['PAY_3_up'].value_counts(ascending=False)
plt.figure(figsize=(15,8), num=12, dpi=360)
plt.bar(count_PAY_3.index,count_PAY_3.values,color = 'green')
plt.xlabel('PAY_3')
plt.ylabel('# Applicants')
plt.title('Repayment status in July Distribution')
plt.show()

# PAY_4
data['PAY_4_up'] = data['PAY_4'].map(pay_dict)
count_PAY_4 = data['PAY_4_up'].value_counts(ascending=False)
plt.figure(figsize=(15,8), num=12, dpi=360)
plt.bar(count_PAY_4.index,count_PAY_4.values,color = 'green')
plt.xlabel('PAY_4')
plt.ylabel('# Applicants')
plt.title('Repayment status in June Distribution')
plt.show()

# PAY_5
data['PAY_5_up'] = data['PAY_5'].map(pay_dict)
count_PAY_5 = data['PAY_5_up'].value_counts(ascending=False)
plt.figure(figsize=(15,8), num=12, dpi=360)
plt.bar(count_PAY_5.index,count_PAY_5.values,color = 'green')
plt.xlabel('PAY_5')
plt.ylabel('# Applicants')
plt.title('Repayment status in May Distribution')
plt.show()

# PAY_6
data['PAY_6_up'] = data['PAY_6'].map(pay_dict)
count_PAY_6 = data['PAY_6_up'].value_counts(ascending=False)
plt.figure(figsize=(15,8), num=12, dpi=360)
plt.bar(count_PAY_6.index,count_PAY_6.values,color = 'green')
plt.xlabel('PAY_6')
plt.ylabel('# Applicants')
plt.title('Repayment status in April Distribution')
plt.show()

# Mapping Payment_Behavior to categorical values
def map_payment_behaviour(col):
    col = np.where(col == 1, 'Paid in full',\
                            np.where(col == 2, 'Not spent last month',\
                            np.where(col < 0.25, 'less than 25%',\
                            np.where((col >= 0.25)&(col < 0.5), '25% to 50%',\
                            np.where((col >= 0.5)&(col < 0.75), '50% to 75%',\
                            np.where((col >= 0.75)&(col < 1.0), 'more than 75% to full','Not spent last month'))))))
    return col

data['Payment_Behavior_1_up'] = map_payment_behaviour(data.Payment_Behavior_1)
data['Payment_Behavior_2_up'] = map_payment_behaviour(data.Payment_Behavior_2)
data['Payment_Behavior_3_up'] = map_payment_behaviour(data.Payment_Behavior_3)
data['Payment_Behavior_4_up'] = map_payment_behaviour(data.Payment_Behavior_4)
data['Payment_Behavior_5_up'] = map_payment_behaviour(data.Payment_Behavior_5)

# Payment_Behavior_1
count_Payment_Behavior_1 = data['Payment_Behavior_1_up'].value_counts()
plt.figure(figsize=(12,8))
plt.bar(count_Payment_Behavior_1.index,count_Payment_Behavior_1.values,color = 'green')
plt.xlabel('Payment Behavior of August')
plt.ylabel('# Applicants')
plt.title('Payment Behavior of August Distribution')
plt.show()

# Payment_Behavior_2
count_Payment_Behavior_2 = data['Payment_Behavior_2_up'].value_counts()
plt.figure(figsize=(12,8))
plt.bar(count_Payment_Behavior_2.index,count_Payment_Behavior_2.values,color = 'green')
plt.xlabel('count_Payment_Behavior_2')
plt.ylabel('# Applicants')
plt.title('Payment Behavior of July Distribution')
plt.show()

# Payment_Behavior_3
count_Payment_Behavior_3 = data['Payment_Behavior_3_up'].value_counts()
plt.figure(figsize=(12,8))
plt.bar(count_Payment_Behavior_3.index,count_Payment_Behavior_3.values,color = 'green', align='edge')
plt.xlabel('Payment Behavior of june')
plt.ylabel('# Applicants')
plt.title('Payment Behavior of June Distribution')
plt.show()

# Payment_Behavior_4
count_Payment_Behavior_4 = data['Payment_Behavior_4_up'].value_counts()
plt.figure(figsize=(12,8))
plt.bar(count_Payment_Behavior_4.index,count_Payment_Behavior_4.values,color = 'green')
plt.xlabel('Payment Behavior of may')
plt.ylabel('# Applicants')
plt.title('Payment Behavior of May Distribution')
plt.show()

# Payment_Behavior_5
count_Payment_Behavior_5 = data['Payment_Behavior_5_up'].value_counts()
plt.figure(figsize=(12,8))
plt.bar(count_Payment_Behavior_5.index,count_Payment_Behavior_5.values,color = 'green')
plt.xlabel('Payment Behavior of April')
plt.ylabel('# Applicants')
plt.title('Payment Behavior of April Distribution')
plt.show()

# Uivariate EDA - Numerical Variable
# Age
plt.style.use('ggplot')
data.AGE.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Age Distribution')
plt.xlabel('Age band')
plt.ylabel('# Applicants')
plt.grid(axis='y', alpha=0.75)
plt.show()

# LIMIT_BAL
data.LIMIT_BAL.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.xlabel('Balance Limit Distribution')
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
plt.title('Amount of previous payment in September')
plt.xlabel('PAY Amount in sept')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# PAY_AMT2
data.PAY_AMT2.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Amount of previous payment in August')
plt.xlabel('PAY Amount in Aug')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# PAY_AMT3
data.PAY_AMT3.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Amount of previous payment in July')
plt.xlabel('PAY Amount in July')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# PAY_AMT4
data.PAY_AMT4.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Amount of previous payment in June')
plt.xlabel('PAY Amount in June')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# PAY_AMT5
data.PAY_AMT5.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Amount of previous payment in May')
plt.xlabel('PAY Amount in May')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# PAY_AMT6
data.PAY_AMT6.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Amount of previous payment in April')
plt.xlabel('PAY Amount in April')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.25)
plt.show()

# Utilization1
data.Utilization1.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Utilization in Sept')
plt.xlabel('Utilization in sept')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Utilization2
data.Utilization2.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Utilization in Aug')
plt.xlabel('Utilization in Aug')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Utilization3
data.Utilization3.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Utilization in July')
plt.xlabel('Utilization in July')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Utilization4
data.Utilization4.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Utilization in June')
plt.xlabel('Utilization in June')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Utilization5
data.Utilization5.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Utilization in May')
plt.xlabel('Utilization in May')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Utilization6
data.Utilization6.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.title('Utilization in April')
plt.xlabel('Utilization in April')
plt.ylabel('# Applicant')
plt.grid(axis='y', alpha=0.25)
plt.show()

# Bivariate Categorical Inputs - Target Variable
# Gender & Target
target_Gender_Cross_Freq = data.groupby([data['target_up'],data['gender_up']]).size().reset_index(name='Freq')
target_Gender_1 = target_Gender_Cross_Freq.groupby(['target_up','gender_up'])['Freq'].sum()/target_Gender_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_Gender_1 = target_Gender_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('Target')
plt.ylabel('Gender')
plt.title('Gender with target')
handles, labels = ax_target_Gender_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':20}, bbox_to_anchor=(1.5, 0.8),handles=handles[::-1])
plt.show()

# EDUCATION & Target
target_EDUCATION_Cross_Freq = data.groupby([data['target_up'],data['education_up']]).size().reset_index(name='Freq')
target_EDUCATION_1 = target_EDUCATION_Cross_Freq.groupby(['target_up','education_up'])['Freq'].sum()/target_EDUCATION_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_EDUCATION_1 = target_EDUCATION_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('target')
plt.ylabel('Education')
plt.title('Education with target')
handles, labels = ax_target_EDUCATION_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':18}, bbox_to_anchor=(1.58, 0.87),handles=handles[::-1])
plt.show()

# MARRIAGE & Target
target_MARRIAGE_Cross_Freq = data.groupby([data['target_up'],data['marriage_up']]).size().reset_index(name='Freq')
target_MARRIAGE_1 = target_MARRIAGE_Cross_Freq.groupby(['target_up','marriage_up'])['Freq'].sum()/target_MARRIAGE_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_MARRIAGE_1 = target_MARRIAGE_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('target')
plt.ylabel('Marriage')
plt.title('Marriage with target')
handles, labels = ax_target_MARRIAGE_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':20}, bbox_to_anchor=(1.55, 0.8),handles=handles[::-1])
plt.show()

# PAY_1 & Target
target_PAY_1_Cross_Freq = data.groupby([data['target_up'],data['PAY_1_up']]).size().reset_index(name='Freq')
target_PAY_1_1 = target_PAY_1_Cross_Freq.groupby(['target_up','PAY_1_up'])['Freq'].sum()/target_PAY_1_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_PAY_1_1 = target_PAY_1_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('target')
plt.ylabel('PAY_1')
plt.title('Repayment status in September with target')
handles, labels = ax_target_PAY_1_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':12}, bbox_to_anchor=(1.5, 1.02),handles=handles[::-1])
plt.show()

# PAY_2 & Target
target_PAY_2_Cross_Freq = data.groupby([data['target_up'],data['PAY_2_up']]).size().reset_index(name='Freq')
target_PAY_2_1 = target_PAY_2_Cross_Freq.groupby(['target_up','PAY_2_up'])['Freq'].sum()/target_PAY_2_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_PAY_2_1 = target_PAY_2_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('target')
plt.ylabel('PAY_2')
plt.title('Repayment status in August with target')
handles, labels = ax_target_PAY_2_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':12}, bbox_to_anchor=(1.5, 1.02),handles=handles[::-1])
plt.show()

# PAY_3 & Target
target_PAY_3_Cross_Freq = data.groupby([data['target_up'],data['PAY_3_up']]).size().reset_index(name='Freq')
target_PAY_3_1 = target_PAY_3_Cross_Freq.groupby(['target_up','PAY_3_up'])['Freq'].sum()/target_PAY_3_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_PAY_3_1 = target_PAY_3_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('target')
plt.ylabel('PAY_3')
plt.title('Repayment status in July with target')
handles, labels = ax_target_PAY_3_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':12}, bbox_to_anchor=(1.5, 1.02),handles=handles[::-1])
plt.show()

# PAY_4 & Target
target_PAY_4_Cross_Freq = data.groupby([data['target_up'],data['PAY_4_up']]).size().reset_index(name='Freq')
target_PAY_4_1 = target_PAY_4_Cross_Freq.groupby(['target_up','PAY_4_up'])['Freq'].sum()/target_PAY_4_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_PAY_4_1 = target_PAY_4_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('target')
plt.ylabel('PAY_4')
plt.title('Repayment status in June with target')
handles, labels = ax_target_PAY_4_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':12}, bbox_to_anchor=(1.5, 1.02),handles=handles[::-1])
plt.show()

# PAY_5 & Target
target_PAY_5_Cross_Freq = data.groupby([data['target_up'],data['PAY_5_up']]).size().reset_index(name='Freq')
target_PAY_5_1 = target_PAY_5_Cross_Freq.groupby(['target_up','PAY_5_up'])['Freq'].sum()/target_PAY_5_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_PAY_5_1 = target_PAY_5_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('target')
plt.ylabel('PAY_5')
plt.title('Repayment status in May with target')
handles, labels = ax_target_PAY_5_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':12}, bbox_to_anchor=(1.5, 1.02),handles=handles[::-1])
plt.show()

# PAY_6 & Target
target_PAY_6_Cross_Freq = data.groupby([data['target_up'],data['PAY_6_up']]).size().reset_index(name='Freq')
target_PAY_6_1 = target_PAY_6_Cross_Freq.groupby(['target_up','PAY_6_up'])['Freq'].sum()/target_PAY_6_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_PAY_6_1 = target_PAY_6_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('target')
plt.ylabel('PAY_6')
plt.title('Repayment status in April with target')
handles, labels = ax_target_PAY_6_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':12}, bbox_to_anchor=(1.33, 1.02),handles=handles[::-1])
plt.show()

# Payment_Behavior_1 & Target
target_Payment_Behavior_1_Cross_Freq = data.groupby([data['target_up'],data['Payment_Behavior_1_up']]).size().reset_index(name='Freq')
target_Payment_Behavior_1_1 = target_Payment_Behavior_1_Cross_Freq.groupby(['target_up','Payment_Behavior_1_up'])['Freq'].sum()/target_Payment_Behavior_1_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_Payment_Behavior_1_1 = target_Payment_Behavior_1_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('target')
plt.ylabel('Payment Behavior 1')
plt.title('Payment Behavior of August with target')
handles, labels = ax_target_Payment_Behavior_1_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':15}, bbox_to_anchor=(1.7, 0.85),handles=handles[::-1])
plt.show()

# Payment_Behavior_2 & Target
target_Payment_Behavior_2_Cross_Freq = data.groupby([data['target_up'],data['Payment_Behavior_2_up']]).size().reset_index(name='Freq')
target_Payment_Behavior_2_1 = target_Payment_Behavior_2_Cross_Freq.groupby(['target_up','Payment_Behavior_2_up'])['Freq'].sum()/target_Payment_Behavior_2_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_Payment_Behavior_2_1 = target_Payment_Behavior_2_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('target')
plt.ylabel('Payment Behavior 2')
plt.title('Payment Behavior July with target')
handles, labels = ax_target_Payment_Behavior_2_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':15}, bbox_to_anchor=(1.7, 0.85),handles=handles[::-1])
plt.show()

# Payment_Behavior_3 & Target
target_Payment_Behavior_3_Cross_Freq = data.groupby([data['target_up'],data['Payment_Behavior_3_up']]).size().reset_index(name='Freq')
target_Payment_Behavior_3_1 = target_Payment_Behavior_3_Cross_Freq.groupby(['target_up','Payment_Behavior_3_up'])['Freq'].sum()/target_Payment_Behavior_3_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_Payment_Behavior_3_1 = target_Payment_Behavior_3_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('target')
plt.ylabel('Payment Behavior 3')
plt.title('Payment Behavior June with target')
handles, labels = ax_target_Payment_Behavior_3_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':15}, bbox_to_anchor=(1.7, 0.9),handles=handles[::-1])
plt.show()

# Payment_Behavior_4 & Target
target_Payment_Behavior_4_Cross_Freq = data.groupby([data['target_up'],data['Payment_Behavior_4_up']]).size().reset_index(name='Freq')
target_Payment_Behavior_4_1 = target_Payment_Behavior_4_Cross_Freq.groupby(['target_up','Payment_Behavior_4_up'])['Freq'].sum()/target_Payment_Behavior_4_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_Payment_Behavior_4_1 = target_Payment_Behavior_4_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('target')
plt.ylabel('Payment Behavior 4')
plt.title('Payment Behavior May with target')
handles, labels = ax_target_Payment_Behavior_4_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':15}, bbox_to_anchor=(1.7, 0.9),handles=handles[::-1])
plt.show()

# Payment_Behavior_5 & Target
target_Payment_Behavior_5_Cross_Freq = data.groupby([data['target_up'],data['Payment_Behavior_5_up']]).size().reset_index(name='Freq')
target_Payment_Behavior_5_1 = target_Payment_Behavior_5_Cross_Freq.groupby(['target_up','Payment_Behavior_5_up'])['Freq'].sum()/target_Payment_Behavior_5_Cross_Freq.groupby(['target_up'])['Freq'].sum()
ax_target_Payment_Behavior_5_1 = target_Payment_Behavior_5_1.unstack().plot.bar(stacked=True, rot=0, legend=False)
plt.xlabel('target')
plt.ylabel('Payment Behavior 5')
plt.title('Payment Behavior April with target')
handles, labels = ax_target_Payment_Behavior_5_1.get_legend_handles_labels()
plt.legend(loc='upper right', prop={'size':15}, bbox_to_anchor=(1.7, 0.9),handles=handles[::-1])
plt.show()

# Outlier treatment
def outlier_treatment(col):
    sorted(col)
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    Outlier_Neg = (Q1 - 1.5 * IQR)
    Outlier_Pos = (Q3 + 1.5 * IQR)
    col = np.where(col>Outlier_Pos, Outlier_Pos,
                           np.where(col<Outlier_Neg, Outlier_Neg,col))
    return col

data.LIMIT_BAL = outlier_treatment(data.LIMIT_BAL)
data.AGE = outlier_treatment(data.AGE)
data.BILL_AMT1 = outlier_treatment(data.BILL_AMT1)
data.BILL_AMT2 = outlier_treatment(data.BILL_AMT2)
data.BILL_AMT3 = outlier_treatment(data.BILL_AMT3)
data.BILL_AMT4 = outlier_treatment(data.BILL_AMT4)
data.BILL_AMT5 = outlier_treatment(data.BILL_AMT5)
data.BILL_AMT6 = outlier_treatment(data.BILL_AMT6)

# Bivariate EDA Numerical Inputs - Target Variable
# target & LIMIT_BAL
ax_target_LIMIT_BAL = sns.boxplot(x='target_up', y='LIMIT_BAL', data=data, palette="Set3").set_title('Balance Limit with Target')
plt.xlabel('Target')
plt.ylabel('Balance limit')
plt.show()

# target & AGE
ax_target_AGE = sns.boxplot(x='target_up', y='AGE', data=data, palette="Set3").set_title('Age withTarget')
plt.xlabel('Target')
plt.ylabel('Age')
plt.show()

# target & BILL_AMT1
ax_target_BILL_AMT1 = sns.boxplot(x='target_up', y='BILL_AMT1', data=data, palette="Set3").set_title('Bill statement in September with target')
plt.xlabel('Target')
plt.ylabel('Bill Statement')
plt.show()

# target & BILL_AMT2
ax_target_BILL_AMT2 = sns.boxplot(x='target_up', y='BILL_AMT2', data=data, palette="Set3").set_title('Bill statement in August with target')
plt.xlabel('Target')
plt.ylabel('Bill Statement')
plt.show()

# target & BILL_AMT3
ax_target_BILL_AMT3 = sns.boxplot(x='target_up', y='BILL_AMT3', data=data, palette="Set3").set_title('Bill statement in July with target')
plt.xlabel('Target')
plt.ylabel('Bill Statement')
plt.show()

# target & BILL_AMT4
ax_target_BILL_AMT4 = sns.boxplot(x='target_up', y='BILL_AMT4', data=data, palette="Set3").set_title('Bill statement in June with target')
plt.xlabel('Target')
plt.ylabel('Bill Statement')
plt.show()

# target & BILL_AMT5
ax_target_BILL_AMT5 = sns.boxplot(x='target_up', y='BILL_AMT5', data=data, palette="Set3").set_title('Bill statement in May with target')
plt.xlabel('Target')
plt.ylabel('Bill Statement')
plt.show()

# target & BILL_AMT6
ax_target_BILL_AMT6 = sns.boxplot(x='target_up', y='BILL_AMT6', data=data, palette="Set3").set_title('Bill statement in April with target')
plt.xlabel('Target')
plt.ylabel('Bill Statement')
plt.show()

# Removing outlier for Previous paymemt of each month
data.PAY_AMT1 = outlier_treatment(data.PAY_AMT1)
data.PAY_AMT2 = outlier_treatment(data.PAY_AMT2)
data.PAY_AMT3 = outlier_treatment(data.PAY_AMT3)
data.PAY_AMT4 = outlier_treatment(data.PAY_AMT4)
data.PAY_AMT5 = outlier_treatment(data.PAY_AMT5)
data.PAY_AMT6 = outlier_treatment(data.PAY_AMT6)

# target & PAY_AMT1
ax_target_PAY_AMT1 = sns.boxplot(x='target_up', y='PAY_AMT1', data=data, palette="Set3").set_title('previous payment in September with target')
plt.xlabel('Target')
plt.ylabel('Previous payment')
plt.show()

# target & PAY_AMT2
ax_target_PAY_AMT2 = sns.boxplot(x='target_up', y='PAY_AMT2', data=data, palette="Set3").set_title('previous payment in Aug with target')
plt.xlabel('Target')
plt.ylabel('Previous payment')
plt.show()

# target & PAY_AMT3
ax_target_PAY_AMT3 = sns.boxplot(x='target_up', y='PAY_AMT3', data=data, palette="Set3").set_title('previous payment in July with target')
plt.xlabel('Target')
plt.ylabel('Previous payment')
plt.show()

# target & PAY_AMT4
ax_target_PAY_AMT4 = sns.boxplot(x='target_up', y='PAY_AMT4', data=data, palette="Set3").set_title('previous payment in June with target')
plt.xlabel('Target')
plt.ylabel('Previous payment')
plt.show()

# target & PAY_AMT5
ax_target_PAY_AMT5 = sns.boxplot(x='target_up', y='PAY_AMT5', data=data, palette="Set3").set_title('previous payment in May with target')
plt.xlabel('Target')
plt.ylabel('Previous payment')
plt.show()

# target & PAY_AMT6
ax_target_PAY_AMT6 = sns.boxplot(x='target_up', y='PAY_AMT6', data=data, palette="Set3").set_title('previous payment in April with target')
plt.xlabel('Target')
plt.ylabel('Previous payment')
plt.show()

# Removing outliers of utilization
data.Utilization1 = outlier_treatment(data.Utilization1)
data.Utilization2 = outlier_treatment(data.Utilization2)
data.Utilization3 = outlier_treatment(data.Utilization3)
data.Utilization4 = outlier_treatment(data.Utilization4)
data.Utilization5 = outlier_treatment(data.Utilization5)
data.Utilization6 = outlier_treatment(data.Utilization6)

# target & Utilization1
ax_target_Utilization1 = sns.boxplot(x='target_up', y='Utilization1', data=data, palette="Set3").set_title('Utilization by Sept with target')
plt.xlabel('Target')
plt.ylabel('Utilization')
plt.show()

# target & Utilization2
ax_target_Utilization2 = sns.boxplot(x='target_up', y='Utilization2', data=data, palette="Set3").set_title('Utilization by Aug with target')
plt.xlabel('Target')
plt.ylabel('Utilization')
plt.show()

# target & Utilization3
ax_target_Utilization3 = sns.boxplot(x='target_up', y='Utilization3', data=data, palette="Set3").set_title('Utilization by July with target')
plt.xlabel('Target')
plt.ylabel('Utilization')
plt.show()

# target & Utilization4
ax_target_Utilization4 = sns.boxplot(x='target_up', y='Utilization4', data=data, palette="Set3").set_title('Utilization by June with target')
plt.xlabel('Target')
plt.ylabel('Utilization')
plt.show()

# target & Utilization5
ax_target_Utilization5 = sns.boxplot(x='target_up', y='Utilization5', data=data, palette="Set3").set_title('Utilization by May with target')
plt.xlabel('Target')
plt.ylabel('Utilization')
plt.show()

# target & Utilization6
ax_target_Utilization6 = sns.boxplot(x='target_up', y='Utilization6', data=data, palette="Set3").set_title('Utilization April with target')
plt.xlabel('Target')
plt.ylabel('Utilization')
plt.show()

# Split the data into X(Input Variables) & y(Target Variables)
X = data.drop(columns='target',axis=1)
y = data.target

# Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
