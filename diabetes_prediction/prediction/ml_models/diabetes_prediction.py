#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# # 2. Loading the Dataset

# In[2]:


# 1. Load your dataset
df = pd.read_csv('C:\\Users\\Minusha Attygala\\OneDrive\\Documents\\Final Individual Project\\diabetes.csv')
df.head()

# In[3]:


df.shape

# # 3. Data Preprocessing

# In[4]:


# Checking for Missing Values
df.isnull().sum()

# In[5]:


# Basic statistical summary
df.describe()

# In[6]:


# Displaying the types of each column
df.dtypes

# In[7]:


print(df.columns)

# ### - family_history_with_overweight	- Has a family member suffered or suffers from overweight?		 
# ### - FAVC - Do you eat high caloric food frequently?	
# ### - FCVC - Do you usually eat vegetables in your meals?		 
# ### - NCP - How many main meals do you have daily?		 
# ### - CAEC - Do you eat any food between meals?	 
# ### - SMOKE - Do you smoke?	 
# ### - CH2O - How much water do you drink daily?	 
# ### - SCC - Do you monitor the calories you eat daily?	 
# ### - FAF - How often do you have physical activity?		 
# ### - TUE - How much time do you use technological devices such as cell phone, videogames, television, computer and others?	 
# ### - CALC - How often do you drink alcohol?	 
# ### - MTRANS	Feature - Which transportation do you usually use?	
# ### - NObeyesdad - Obesity level		

# In[8]:


# Create a box plot for all numerical columns
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.title('Box Plot of all numerical Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()

# ![image.png](attachment:image.png)

# In[9]:


# Calculate BMI
df['bmi'] = df['Weight'] / (df['Height'] ** 2)

# In[10]:


# Create BMI Category
def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi <= 24.9:
        return 'Normal weight'
    elif 25.0 <= bmi <= 29.9:
        return 'Overweight'
    elif 30.0 <= bmi <= 34.9:
        return 'Obese Class 1'
    elif 35.0 <= bmi <= 39.9:
        return 'Obese Class 2'
    else:
        return 'Morbidly Obese'

df['weight_status'] = df['bmi'].apply(bmi_category)

# In[11]:


# Drop original Height and Weight if not needed anymore
df.drop(['Height', 'Weight', 'NObeyesdad'], axis=1, inplace=True)

# In[12]:


df[['bmi', 'weight_status']].head()

# In[13]:


df.head()

# In[14]:


df['weight_status'].value_counts()

# | Feature | Why it matters |
# |---------|----------------|
# | BMI (calculated from Height & Weight) | Strongest predictor. Higher BMI → higher insulin resistance → higher diabetes risk. |
# | family_history_with_overweight | Genetics & family history play a major role in predisposition. |
# | FAVC (High-calorie food consumption) | Leads to weight gain → increases obesity risk → raises chances of diabetes. |
# | FCVC (Vegetable consumption) | Higher veg intake = better glucose regulation, lower risk. |
# | NCP (Number of meals) | Irregular or excessive eating patterns can spike insulin demand. |
# | CAEC (Eating between meals) | Constant snacking, especially unhealthy, impacts glucose levels. |
# | CH2O (Water intake) | Good hydration supports kidney function and may help glucose management. |
# | SCC (Monitoring calories) | Shows awareness of diet → generally leads to better weight and glucose control. |
# | FAF (Physical activity) | Crucial! Regular activity improves insulin sensitivity. |
# | CALC (Alcohol consumption) | Excessive alcohol can lead to pancreatitis or insulin resistance. |
# | SMOKE | Smoking increases inflammation and insulin resistance. |
# | TUE (Tech time / sedentary behavior) | More screen time = less movement = higher diabetes risk. |
# | MTRANS (Mode of transportation) | Passive modes (car, bike) may reflect a sedentary lifestyle. |

# In[15]:


# Map Weight Status → Diabetes Risk
def map_diabetes_risk(weight_status):
    mapping = {
        "Underweight": "Low risk",
        "Normal weight": "Lowest risk",
        "Overweight": "Moderate to high risk",
        "Obese Class 1": "High risk",
        "Obese Class 2": "Very high risk",
        "Morbidly Obese": "Extremely high risk"
    }
    return mapping[weight_status]

# In[16]:


df['diabetes_risk'] = df['weight_status'].apply(map_diabetes_risk)

# In[17]:


df.head()

# ### Principal Component Analysis (PCA)

# In[18]:


from sklearn.decomposition import PCA # library for PCA
from sklearn.preprocessing import StandardScaler #library to standerize the dats

# In[19]:


features = df.select_dtypes(include=[np.number]).columns.tolist() # selecting numerical features for pca
X = df[features]

# In[20]:


scaler = StandardScaler() # standardizing / feature scalling the data
X_scaled = scaler.fit_transform(X)

# In[21]:


pca = PCA(n_components=2)  # number of components = 2
principal_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2']) #DataFrame with the principal components

# In[22]:


explained_variance = pca.explained_variance_ratio_ #print Explained variance by each principle component
print(f'Explained variance by each principle component: {explained_variance}')

# In[23]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title('PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()

# In[24]:


loadings = pca.components_.T * np.sqrt(pca.explained_variance_) #the relationships between the original features and the principal components
loading_df = pd.DataFrame(loadings, index=features, columns=['PC1', 'PC2'])
print(loading_df)

# In[25]:


# biplot
plt.figure(figsize=(10, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)

# Add arrows for the loadings
for i, feature in enumerate(features):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1],
              color='r', alpha=0.5, head_width=0.05)
    plt.text(loadings[i, 0], loadings[i, 1], feature, color='black', ha='center', va='center')

plt.title('PCA Biplot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.xlim(-1, 1) # change these two the way you want
plt.ylim(-1, 1)
plt.axhline(0, color='grey', lw=0.5, ls='--')
plt.axvline(0, color='grey', lw=0.5, ls='--')
plt.show()

# In[26]:


explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.xticks(range(1, len(explained_variance) + 1))  # Set x-ticks to be the component numbers
plt.grid()
plt.show()

# # Model Building

# ### Random Forest

# In[27]:


from sklearn.ensemble import RandomForestClassifier

# In[28]:


#selecting feature/ variables from the dataset for the models

#encoding categorical variables
df['Gender'] = df['Gender'].astype('category').cat.codes
df['family_history_with_overweight'] = df['family_history_with_overweight'].astype('category').cat.codes
df['FAVC'] = df['FAVC'].astype('category').cat.codes
df['CAEC'] = df['CAEC'].astype('category').cat.codes
df['SMOKE'] = df['SMOKE'].astype('category').cat.codes
df['SCC'] = df['SCC'].astype('category').cat.codes
df['CALC'] = df['CALC'].astype('category').cat.codes
df['MTRANS'] = df['MTRANS'].astype('category').cat.codes
df['weight_status'] = df['weight_status'].astype('category').cat.codes

#encoding Target variable
df['diabetes_risk'] = df['diabetes_risk'].astype('category').cat.codes

#features and target variable
features = ['Gender', 'Age', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'bmi', 'weight_status']
target = ['diabetes_risk']
x = df[features]
y = df[target]

# In[29]:


#split the dataser
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# In[30]:


#Training the model - Random Forest
RF_model = RandomForestClassifier(n_estimators=100, random_state=42)
RF_model.fit(x_train, y_train)

# In[31]:


y_pred_rf = RF_model.predict(x_test) # Predictions

# In[32]:


accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_rf

# In[33]:


print(classification_report(y_test, y_pred_rf))

# In[34]:


cm_RF = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_RF, annot=True, fmt='d', cmap='Blues', xticklabels=RF_model.classes_, yticklabels=RF_model.classes_)
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ### XGBoost

# In[35]:


import xgboost as xgb #XGboost library

# In[36]:


XG_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
XG_model.fit(x_train, y_train)

# In[37]:


# Make predictions
y_pred_xg = XG_model.predict(x_test)

# In[38]:


accuracy_xg = accuracy_score(y_test, y_pred_xg)
accuracy_xg

# In[39]:


print(classification_report(y_test, y_pred_xg))

# In[40]:


cm_XG = confusion_matrix(y_test, y_pred_xg)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_XG, annot=True, fmt='d', cmap='Blues', xticklabels=XG_model.classes_, yticklabels=XG_model.classes_)
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ### Naive Bayes

# In[41]:


from sklearn.naive_bayes import GaussianNB

# In[42]:


NB_model = GaussianNB()
NB_model.fit(x_train, y_train)
y_pred_naiveB = NB_model.predict(x_test)

# In[43]:


accuracy_nb = accuracy_score(y_test, y_pred_naiveB)
accuracy_nb

# In[44]:


print(classification_report(y_test, y_pred_naiveB))

# In[45]:


cm_nb = confusion_matrix(y_test, y_pred_naiveB)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=NB_model.classes_, yticklabels=NB_model.classes_)
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ### KNN

# In[46]:


from sklearn.neighbors import KNeighborsClassifier

# In[47]:


# training the model -knn
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)

# In[48]:


y_pred_knn = knn_model.predict(x_test)

# In[49]:


accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_knn

# In[50]:


print(classification_report(y_test, y_pred_knn))

# In[51]:


cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.title('Confusion Matrix - KNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ### K-Means

# In[52]:


from sklearn.cluster import KMeans

# In[53]:


#KMeans
# use feature scalling of PCA -X_scaled for KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)
df['cluster'] = kmeans.labels_

# In[54]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['cluster'], palette='viridis')
plt.title('KMeans Clustering Results')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.legend(title='Cluster')
plt.show()

# In[57]:


import joblib
joblib.dump(RF_model, "diabetes_model.pkl")
