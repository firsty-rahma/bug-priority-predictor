import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_csv('../data/bugs.csv')
print(f"Columns: {data.shape[1]}")
print(f"Shape: {data.shape}")
data.head()

print("Data Info:")
data.info()
print("\n")
print("Missing values:")
data.isnull().sum()
# Some columns have empty values: short_description and long_description, will check it later

# Check severity categories
print("Severity Categories:")
print("="*50)
print(data['severity_category'].value_counts())

print("\n" + "="*50)
print("Percentages:")
print((data['severity_category'].value_counts(normalize=True) * 100).round(2))

print("\n" + "="*50)
print("Severity Codes:")
print(data['severity_code'].value_counts().sort_index())


print("Category-Code Mapping:")
print("="*60)
mapping = data.groupby(['severity_category', 'severity_code']).size().unstack(fill_value=0)
print(mapping)

print("\n" + "="*60)
print("For each category, what codes does it have?")
for cat in data['severity_category'].unique():
    codes = data[data['severity_category'] == cat]['severity_code'].unique()
    count = len(data[data['severity_category'] == cat])
    print(f"{cat:12s} → codes: {sorted(codes)} (n={count})")

# Drop the inconsistent severity_code column
# Decision: Dropping severity_code due to data inconsistency
# - Both 'minor' and 'normal' categories map to code 2
# - Code 3 is missing entirely
# - severity_category provides clear, distinct labels for classification
# - Using severity_category as the target variable
data = data.drop('severity_code', axis=1)
print(f"Shape after dropping severity_code: {data.shape}")

# Save to a new file
data.to_csv('../data/bugs_cleaned.csv', index=False)
print(f"Cleaned data saved: {data.shape}")

mapping_product_name = data.groupby(['severity_category', 'product_name']).size().reset_index(name='count')
print(mapping_product_name)

sn.histplot(data, x="severity_category", hue="product_name", multiple="stack")

# 2. Component name
mapping_component_name = data.groupby(['severity_category', 'component_name']).size().reset_index(name='count')
print(mapping_component_name)

sn.histplot(data, x="severity_category", hue="component_name", multiple="stack")

mapping_resolution_category = data.groupby(['severity_category', 'resolution_category']).size().reset_index(name='count')
print(mapping_resolution_category)

# 4. Status Category
mapping_status_category = data.groupby(['severity_category', 'status_category']).size().reset_index(name='count')
print(mapping_status_category)
sn.catplot(data, x="severity_category", hue="status_category", kind="count")


# In[11]:


# 5. Quantity of comments
mapping_quantity_of_comments = data.groupby('severity_category')['quantity_of_comments'].sum().reset_index(name='total_quantity_of_comments')
print(mapping_quantity_of_comments)

sn.stripplot(data, x="severity_category", y="quantity_of_comments")


# In[12]:


# 5. Quantity of comments
mapping_quantity_of_comments = data.groupby('severity_category')['bug_fix_time'].mean().reset_index(name='average_bug_fix_time')
print(mapping_quantity_of_comments)

sn.stripplot(data, x="severity_category", y="bug_fix_time")


# In[13]:


# Based on my data exploration, the data had largest imbalance on "normal" category 
# Most of bug report activities are in "normal" category

