#!/usr/bin/env python
# coding: utf-8

# ### Importing Dependecies

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,roc_curve, auc


# ### Data Collection and Processing

# In[6]:


df=pd.read_csv("C:/Users/ASUS/OneDrive/Desktop/Data Science/data.csv")


# In[21]:


# Printing first 5 rows of the dataset
df.head()


# In[22]:


# Printing last 5 rows of the dataset
df.tail()


# In[23]:


# number of rows and columns in the dataset
df.shape


# In[24]:


# getting some info about the data
df.info()


# In[25]:


# checking for missing values
print(df.isnull().sum())


# In[26]:


# statistical measures about the data
df.describe()


# In[8]:


X = df.drop('fail', axis=1)
y = df['fail']


# In[9]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# In[10]:


plt.figure(figsize=(15, 10))
features = ['footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature']
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()


# In[11]:


bar=df['fail'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of the Target Variable (fail)')
plt.xlabel('Fail')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No Failure (0)', 'Failure (1)'], rotation=0)
for bar in bar.patches:
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 10, 
             f'{int(bar.get_height())}', ha='center', va='bottom')
plt.show()


# In[12]:


plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[13]:


for col in df.columns:
    if col != 'fail':
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='fail', y=col, data=df)
        plt.title(f'{col} vs. fail')
        plt.show()


# In[14]:


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'FPR': roc_curve(y_test, y_prob)[0],  # False Positive Rate
        'TPR': roc_curve(y_test, y_prob)[1],  # True Positive Rate
        'Thresholds': roc_curve(y_test, y_prob)[2]  # Thresholds
    }
    return results


# In[15]:


models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}


# In[16]:


results = {}
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    results[model_name] = evaluate_model(model, X_train, X_test, y_train, y_test)

# Find the best model based on ROC-AUC score
best_model_name = max(results, key=lambda k: results[k]['ROC-AUC'])
best_model_results = results[best_model_name]

print("\nBest Model:")
print(f"{best_model_name} with ROC-AUC: {best_model_results['ROC-AUC']:.4f}")
print(f"Accuracy: {best_model_results['Accuracy']:.4f}")
print(f"Precision: {best_model_results['Precision']:.4f}")
print(f"Recall: {best_model_results['Recall']:.4f}")
print(f"F1 Score: {best_model_results['F1 Score']:.4f}")


# In[17]:


print("\nAll Model Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        if isinstance(value, (list, np.ndarray)):
            print(f"{metric}: [Array with length {len(value)}]")
        else:
            print(f"{metric}: {value:.4f}")


# In[18]:


plt.figure(figsize=(10, 6))
plt.plot(best_model_results['FPR'], best_model_results['TPR'], color='blue', lw=2, label=f'ROC curve (area = {best_model_results["ROC-AUC"]:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[19]:


best_model = GradientBoostingClassifier()
best_model.fit(X_train, y_train)
feature_importance = best_model.feature_importances_
indices = np.argsort(feature_importance)[::-1]


# In[20]:


print("Feature ranking:")

for f in range(X.shape[1]):
    print(f"{f + 1}. feature {indices[f]} ({feature_importance[indices[f]]:.4f})")

# Plot the feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature importances for GradientBoosting")
plt.bar(range(X.shape[1]), feature_importance[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[ ]:




