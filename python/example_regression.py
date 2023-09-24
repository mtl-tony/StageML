from StageML import StageMLRegressor

import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the titanic dataset
df = sns.load_dataset('titanic')

# Drop rows with missing values
df = df.dropna()

# Define the feature columns and target column
feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target_col = 'survived'

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.2, random_state=42)
