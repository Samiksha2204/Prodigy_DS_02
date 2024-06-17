import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Titanic-Dataset.csv')
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())


df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(columns=['Cabin'], inplace=True)

# Check for duplicates
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)


df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

# Summary statistics
print(df.describe(include='all'))

# Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Countplot of Survived
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Countplot of Pclass
plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', data=df)
plt.title('Passenger Class Count')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()

# Countplot of Sex
plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', data=df)
plt.title('Gender Count')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Survival rate by Sex
plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.show()

# Survival rate by Pclass
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.show()

# Survival rate by Embarked
plt.figure(figsize=(10, 6))
sns.barplot(x='Embarked', y='Survived', data=df)
plt.title('Survival Rate by Embarkation Point')
plt.xlabel('Embarked')
plt.ylabel('Survival Rate')
plt.show()

# Pairplot to see interactions between variables
plt.figure(figsize=(10, 8))
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass', 'Sex', 'HasCabin']], hue='Survived')
plt.show()

# Select only numeric columns for the correlation matrix
numeric_df = df.select_dtypes(include=[np.number])

# Heatmap to show correlation between numeric features
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
