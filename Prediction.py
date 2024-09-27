# Let's load and inspect the dataset that the user uploaded to understand its structure and proceed with data cleaning and analysis.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
file_path = 'Flight_delay.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()


# Checking for missing values in the dataset
missing_values = data.isnull().sum()

# Display the missing values for each column
missing_values[missing_values > 0]

# Remove rows with missing values in 'Org_Airport' and 'Dest_Airport'
data_cleaned = data.dropna(subset=['Org_Airport', 'Dest_Airport'])

# Check if missing values are removed
missing_values_cleaned = data_cleaned.isnull().sum()

# Displaying the cleaned data and the missing values count
missing_values_cleaned[missing_values_cleaned > 0]

# Extract delay-related features

# Convert 'CRSArrTime', 'ArrTime', 'DepTime' to minutes from midnight (if these are not NaN or missing)
def convert_to_minutes(time):
    """Convert time in HHMM format to minutes from midnight."""
    if pd.isnull(time):
        return None
    hours = time // 100
    minutes = time % 100
    return hours * 60 + minutes

# Create delay-related features
data_cleaned['DepTimeMinutes'] = data_cleaned['DepTime'].apply(convert_to_minutes)
data_cleaned['ArrTimeMinutes'] = data_cleaned['ArrTime'].apply(convert_to_minutes)
data_cleaned['CRSArrTimeMinutes'] = data_cleaned['CRSArrTime'].apply(convert_to_minutes)

# Calculate arrival and departure delays
data_cleaned['ArrivalDelay'] = data_cleaned['ArrTimeMinutes'] - data_cleaned['CRSArrTimeMinutes']
data_cleaned['DepartureDelay'] = data_cleaned['DepTimeMinutes'] - data_cleaned['CRSArrTimeMinutes']

# Check the newly created delay-related columns
data_cleaned[['DepTime', 'ArrTime', 'CRSArrTime', 'ArrivalDelay', 'DepartureDelay']].head()


# Visualizing distribution of Arrival Delay
plt.figure(figsize=(10, 6))
plt.hist(data_cleaned['ArrivalDelay'].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Arrival Delays')
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Visualizing distribution of Departure Delay
plt.figure(figsize=(10, 6))
plt.hist(data_cleaned['DepartureDelay'].dropna(), bins=50, color='salmon', edgecolor='black')
plt.title('Distribution of Departure Delays')
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Setting up the target variable
# We will define the target as a binary variable: 1 if ArrivalDelay > 15 minutes, 0 otherwise.

delay_threshold = 15  # Define the threshold for what is considered "delayed"
data_cleaned['Delayed'] = (data_cleaned['ArrivalDelay'] > delay_threshold).astype(int)

# Checking the distribution of the target variable
data_cleaned['Delayed'].value_counts()

# Step 1: Select relevant features and encode categorical variables (one-hot encoding for simplicity)
# For simplicity, let's start with a few key features and exclude highly cardinal features like 'FlightNum' and 'TailNum'
features = ['DepTimeMinutes', 'CRSArrTimeMinutes', 'TaxiIn', 'TaxiOut', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'LateAircraftDelay']
X = data_cleaned[features]
y = data_cleaned['Delayed']

# Step 2: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the logistic regression model with class weighting
logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
logreg.fit(X_train, y_train)

# Step 4: Make predictions and evaluate the model
y_pred = logreg.predict(X_test)

# Step 5: Print the classification report and confusion matrix
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

classification_rep, confusion_mat


# Further reducing the sample size to 1,000 rows to avoid memory issues
data_sample_smaller = data_cleaned.sample(n=1000, random_state=42)

# Selecting the relevant features
features = ['DepTimeMinutes', 'CRSArrTimeMinutes', 'TaxiIn', 'TaxiOut', 
                   'CarrierDelay', 'WeatherDelay', 'NASDelay', 'LateAircraftDelay']
X_sample_smaller = data_sample_smaller[features]
y_sample_smaller = data_sample_smaller['Delayed']

# Splitting data into training and testing sets
X_train_sample_smaller, X_test_sample_smaller, y_train_sample_smaller, y_test_sample_smaller = train_test_split(X_sample_smaller, y_sample_smaller, test_size=0.2, random_state=42)

# Training logistic regression with class_weight='balanced'
logreg_sample_smaller = LogisticRegression(class_weight='balanced', max_iter=1000)
logreg_sample_smaller.fit(X_train_sample_smaller, y_train_sample_smaller)

# Predicting on the test set
y_pred_sample_smaller = logreg_sample_smaller.predict(X_test_sample_smaller)

# Evaluating the model using classification report and confusion matrix
classification_rep_sample_smaller = classification_report(y_test_sample_smaller, y_pred_sample_smaller, output_dict=True)
confusion_mat_sample_smaller = confusion_matrix(y_test_sample_smaller, y_pred_sample_smaller)

# Convert classification report to a DataFrame for better readability
classification_df = pd.DataFrame(classification_rep_sample_smaller).transpose()

# Display the confusion matrix
classification_df, confusion_mat_sample_smaller

# Visualizing correlations between delay-related features
# Selecting relevant delay-related features
delay_features = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'LateAircraftDelay', 'TaxiIn', 'TaxiOut', 'ArrivalDelay']

# Compute correlation matrix
corr_matrix = data_cleaned[delay_features].corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Delay-Related Features')
plt.show()

#---------------------------------------------------------------------------------------
# Logistic regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train,y_train)

#Predict on the test data
y_pred_log_reg = log_reg.predict(X_test)


#------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sample_smaller, y_train_sample_smaller) 

# Predict on the test data
y_pred_knn = knn.predict(X_test_sample_smaller)

#Evaluate Logistic Regression
print("Logistic Regression Evaluation:")
print(f"Accuracy: {accuracy_score(y_test_sample_smaller, y_pred_sample_smaller): 2f}")
print(f"Precision: {precision_score(y_test_sample_smaller, y_pred_sample_smaller): 2f}")
print(f"Recall: {recall_score(y_test_sample_smaller, y_pred_sample_smaller): 2f}")
print(f"F1-Score: {f1_score(y_test_sample_smaller, y_pred_sample_smaller): 2f}")

# Evaluate KNN Regression
print("KNN Evaluation:")
print(f"Accuracy: {accuracy_score(y_test_sample_smaller,y_pred_knn):.2f}")
print(f"Precision: {precision_score(y_test_sample_smaller,y_pred_knn):.2f}")
print(f"Recall: {recall_score(y_test_sample_smaller,y_pred_knn):.2f}")
print(f"F1-Score: {f1_score(y_test_sample_smaller,y_pred_knn):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_sample_smaller,y_pred_knn))
print("\nClassification Report:")
print(classification_report(y_test_sample_smaller,y_pred_knn))

#Plotting Logistic  Regression
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

def plot_decision_boundary(model, X, y, ax, title):
    # Create mesh grid for the first two features
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict classes over the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

# Using only the first two features for visualization
X_train_2D = X_train_sample_smaller.iloc[:, :2].values
X_test_2D = X_test_sample_smaller.iloc[:, :2].values
y_train_2D = y_train_sample_smaller  # Use the correct target variable

# Train model again using only the first two features for visualization
log_reg_2D = LogisticRegression(random_state=42)
log_reg_2D.fit(X_train_2D, y_train_2D)  

knn_2D = KNeighborsClassifier(n_neighbors=5)
knn_2D.fit(X_train_2D, y_train_sample_smaller)

# Plot Logistic Regression decision boundary
plot_decision_boundary(log_reg_2D, X_test_2D, y_test_sample_smaller, ax[0], "Logistic Regression")

# Plot KNN decision boundary
plot_decision_boundary(knn_2D, X_test_2D, y_test, ax[1], "KNN")

plt.tight_layout()
plt.show()