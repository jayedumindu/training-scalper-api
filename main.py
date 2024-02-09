import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

# Load your CSV data
df = pd.read_csv('trading_data.csv')

print("DataFrame Columns:", df.columns)

# Explore the dataset
print(df)

# Label encode the 'Actual Result' column
label_encoder = LabelEncoder()
df['Actual Result'] = label_encoder.fit_transform(df['Actual Result'])

print(df)

# Separate features (X) and target variable (y)
X = df[['MACD Crossover Signal', 'Last Candle Signal', 'Shooting Star Signal', 'Moving Average Trend Signal', 'MACD Turning Point Signal', 'RSI Signal', 'Buy/Sell Pressure Signal']]
y = df['Actual Result']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

# Train the model with the best parameters
optimized_model = RandomForestClassifier(**grid_search.best_params_)
optimized_model.fit(X_train, y_train)

# Save the trained model to a file
dump(optimized_model, 'optimized_model.joblib')

# Make predictions on the test set
y_pred = optimized_model.predict(X_test)

# Decode the predicted labels back to original values
y_pred_original = label_encoder.inverse_transform(y_pred)

# Decode the actual labels back to original values for evaluation
y_test_original = label_encoder.inverse_transform(y_test)

# Evaluate the optimized model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
