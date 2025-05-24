import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("loan-recovery.csv")

# Set the style
sns.set(style="whitegrid")

# Create the plots
plt.figure(figsize=(14, 6))

# Plot 1: Distribution of Loan Amount
plt.subplot(1, 2, 1)
sns.histplot(df['Loan_Amount'], bins=30, kde=True, color='teal')
plt.title('Distribution of Loan Amount')
plt.xlabel('Loan Amount')

# Plot 2: Relationship between Monthly Income and Loan Amount
plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='Monthly_Income', y='Loan_Amount', hue='Recovery_Status', alpha=0.7, palette='Set1')
plt.title('Loan Amount vs Monthly Income')
plt.xlabel('Monthly Income')
plt.ylabel('Loan Amount')

plt.tight_layout()
plt.show()


### Set plot style
##sns.set(style="whitegrid")
##plt.figure(figsize=(12, 5))
##
### Plot 1: Count plot of Payment History by Recovery Status
##plt.subplot(1, 2, 1)
##sns.countplot(data=df, x='Payment_History', hue='Recovery_Status', palette='Set2')
##plt.title('Recovery Status by Payment History')
##plt.xlabel('Payment History')
##plt.ylabel('Count')
##plt.legend(title='Recovery Status')
##
### Plot 2: Average Outstanding Loan Amount by Payment History
##plt.subplot(1, 2, 2)
##sns.barplot(data=df, x='Payment_History', y='Outstanding_Loan_Amount', hue='Recovery_Status', palette='Set3')
##plt.title('Avg. Outstanding Loan Amount by Payment History and Recovery Status')
##plt.xlabel('Payment History')
##plt.ylabel('Avg. Outstanding Amount')
##plt.legend(title='Recovery Status')
##
##plt.tight_layout()
##plt.show()
##
##import pandas as pd
##import seaborn as sns
##import matplotlib.pyplot as plt
##
### Load your data
##df = pd.read_csv("loan-recovery.csv")  # Adjust the path as needed
##
##sns.set(style="whitegrid")
##plt.figure(figsize=(16, 10))
##
### Boxplot 1: Monthly Income by Recovery Status
##plt.subplot(2, 2, 1)
##sns.boxplot(data=df, x='Recovery_Status', y='Monthly_Income', palette='pastel')
##plt.title('Monthly Income by Recovery Status')
##plt.xlabel('Recovery Status')
##plt.ylabel('Monthly Income')
##
### Boxplot 2: Loan Amount by Recovery Status
##plt.subplot(2, 2, 2)
##sns.boxplot(data=df, x='Recovery_Status', y='Loan_Amount', palette='muted')
##plt.title('Loan Amount by Recovery Status')
##plt.xlabel('Recovery Status')
##plt.ylabel('Loan Amount')
##
### Scatter Plot: Monthly Income vs Loan Amount colored by Recovery Status
##plt.subplot(2, 1, 2)
##sns.scatterplot(data=df, x='Monthly_Income', y='Loan_Amount', hue='Recovery_Status', alpha=0.7, palette='Set1')
##plt.title('Monthly Income vs Loan Amount by Recovery Status')
##plt.xlabel('Monthly Income')
##plt.ylabel('Loan Amount')
##
##plt.tight_layout()
##plt.show()
##
##
##sns.set(style="whitegrid")
##plt.figure(figsize=(16, 10))
##
### Boxplot 1: Monthly Income by Recovery Status
##plt.subplot(2, 2, 1)
##sns.boxplot(data=df, x='Recovery_Status', y='Monthly_Income', palette='pastel')
##plt.title('Monthly Income by Recovery Status')
##plt.xlabel('Recovery Status')
##plt.ylabel('Monthly Income')
##
### Boxplot 2: Loan Amount by Recovery Status
##plt.subplot(2, 2, 2)
##sns.boxplot(data=df, x='Recovery_Status', y='Loan_Amount', palette='muted')
##plt.title('Loan Amount by Recovery Status')
##plt.xlabel('Recovery Status')
##plt.ylabel('Loan Amount')
##
### Scatter Plot: Monthly Income vs Loan Amount colored by Recovery Status
##plt.subplot(2, 1, 2)
##sns.scatterplot(data=df, x='Monthly_Income', y='Loan_Amount', hue='Recovery_Status', alpha=0.7, palette='Set1')
##plt.title('Monthly Income vs Loan Amount by Recovery Status')
##plt.xlabel('Monthly Income')
##plt.ylabel('Loan Amount')
##
##plt.tight_layout()
##plt.show()
##
##
### Step 1: Encode categorical variables
##label_encoders = {}
##categorical_cols = df.select_dtypes(include='object').columns
##
##for col in categorical_cols:
##    le = LabelEncoder()
##    df[col] = le.fit_transform(df[col])
##    label_encoders[col] = le
##
### Step 2: Define features and target
##X = df.drop('Recovery_Status', axis=1)
##y = df['Recovery_Status']
##
### Step 3: Train-test split
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
##
### Step 4: Train Random Forest
##model = RandomForestClassifier(n_estimators=100, random_state=42)
##model.fit(X_train, y_train)
##
### Step 5: Predict and Evaluate
##y_pred = model.predict(X_test)
##
### Output results
##print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
##print("\nClassification Report:\n", classification_report(y_test, y_pred))
##print("Accuracy Score:", accuracy_score(y_test, y_pred))
##
##
##
### Drop rows with missing values for simplicity
##df = df.dropna()
##
### Encode categorical features
##df = pd.get_dummies(df, drop_first=True)
##
### Define features and target
##X = df.drop('Outstanding_Loan_Amount', axis=1)
##y = df['Outstanding_Loan_Amount']
##
### Train-test split
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
##
### Random Forest Regressor
##rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
##rf_model.fit(X_train, y_train)
##rf_preds = rf_model.predict(X_test)
##
### Linear Regression
##lr_model = LinearRegression()
##lr_model.fit(X_train, y_train)
##lr_preds = lr_model.predict(X_test)
##
### Evaluation
##print("Random Forest Regressor:")
##print("MSE:", mean_squared_error(y_test, rf_preds))
##print("R2 Score:", r2_score(y_test, rf_preds))
##
##print("\nLinear Regression:")
##print("MSE:", mean_squared_error(y_test, lr_preds))
##print("R2 Score:", r2_score(y_test, lr_preds))
##
##
##
##
### Sort by row index and create a pseudo-time series
##df = df.sort_index()
##ts = pd.Series(df['Outstanding_Loan_Amount'].values, index=pd.RangeIndex(start=0, stop=len(df)))
##
### Fit ARIMA model
##model = ARIMA(ts, order=(1, 1, 1))
##model_fit = model.fit()
##
### Forecast the next 10 time points
##forecast = model_fit.forecast(steps=10)
##print(forecast)
##
### Plot the original data and forecast
##plt.figure(figsize=(10, 5))
##plt.plot(ts, label="Original Data")
##plt.plot(range(len(ts), len(ts)+10), forecast, label="Forecast", color='red')
##plt.title("ARIMA Forecast of Outstanding Loan Amount")
##plt.xlabel("Index (pseudo-time)")
##plt.ylabel("Outstanding Loan Amount")
##plt.legend()
##plt.tight_layout()
##plt.show()
##
