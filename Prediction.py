import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
<<<<<<< HEAD
import numpy as np
=======
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
>>>>>>> 91b437950ddb7d7c356807cf82d3506d43c291f2

file_path = 'Flight_delay.csv'
data = pd.read_csv(file_path)

# Drop rows with missing values in the columns of interest
data_cleaned = data.dropna(subset=['Org_Airport', 'Dest_Airport'])

# Making time standard so it can transalte over better
def convert_to_minutes(time):
    try:
        hours = time // 100
        minutes = time % 100
        return hours * 60 + minutes
    except:
        return np.nan
convert_to_minutes_vectorized = np.vectorize(convert_to_minutes)

# Features for the code based on the dataset
data_cleaned['DepTimeMinutes'] = data_cleaned['DepTime'].apply(convert_to_minutes)
data_cleaned['ArrTimeMinutes'] = data_cleaned['ArrTime'].apply(convert_to_minutes)
data_cleaned['CRSArrTimeMinutes'] = data_cleaned['CRSArrTime'].apply(convert_to_minutes)
data_cleaned['ArrivalDelay'] = data_cleaned['ArrTimeMinutes'] - data_cleaned['CRSArrTimeMinutes']
data_cleaned['DepartureDelay'] = data_cleaned['DepTimeMinutes'] - data_cleaned['CRSArrTimeMinutes']

# Define target variable 'Delayed' based on a threshold for arrival delay and select features
delay_threshold = 15
data_cleaned['Delayed'] = (data_cleaned['ArrivalDelay'] > delay_threshold).astype(int)
features = ['DepTimeMinutes', 'CRSArrTimeMinutes', 'TaxiIn', 'TaxiOut', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'LateAircraftDelay']
X = data_cleaned[features]
y = data_cleaned['Delayed']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
logreg.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = logreg.predict(X_test)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
data_sample_smaller = data_cleaned.sample(n=1500, random_state=42)
X_sample_smaller = data_sample_smaller[features]
y_sample_smaller = data_sample_smaller['Delayed']
X_train_sample_smaller, X_test_sample_smaller, y_train_sample_smaller, y_test_sample_smaller = train_test_split(X_sample_smaller, y_sample_smaller, test_size=0.2, random_state=42)

logreg_sample_smaller = LogisticRegression(class_weight='balanced', max_iter=4000)
logreg_sample_smaller.fit(X_train_sample_smaller, y_train_sample_smaller)
y_pred_sample_smaller = logreg_sample_smaller.predict(X_test_sample_smaller)

classification_rep_sample_smaller = classification_report(y_test_sample_smaller, y_pred_sample_smaller, output_dict=True)
confusion_mat_sample_smaller = confusion_matrix(y_test_sample_smaller, y_pred_sample_smaller)

# Convert classification report to DataFrame for better readability
classification_df = pd.DataFrame(classification_rep_sample_smaller).transpose()

# Display the confusion matrix
classification_df, confusion_mat_sample_smaller

delay_features = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'LateAircraftDelay', 'TaxiIn', 'TaxiOut', 'ArrivalDelay']
corr_matrix = data_cleaned[delay_features].corr()

# Repeat the training with a bigger sample size for memory constraints
data_sample_smaller = data_cleaned.sample(n=2000, random_state=42)
X_sample_smaller = data_sample_smaller[features]
y_sample_smaller = data_sample_smaller['Delayed']
X_train_sample_smaller, X_test_sample_smaller, y_train_sample_smaller, y_test_sample_smaller = train_test_split(X_sample_smaller, y_sample_smaller, test_size=0.2, random_state=42)

logreg_sample_smaller = LogisticRegression(class_weight='balanced', max_iter=1000)
logreg_sample_smaller.fit(X_train_sample_smaller, y_train_sample_smaller)
y_pred_sample_smaller = logreg_sample_smaller.predict(X_test_sample_smaller)
classification_rep_sample_smaller = classification_report(y_test_sample_smaller, y_pred_sample_smaller)
confusion_mat_sample_smaller = confusion_matrix(y_test_sample_smaller, y_pred_sample_smaller)



# Charts for the distribution of both delays
plt.figure(figsize=(10, 6))
plt.hist(data_cleaned['ArrivalDelay'].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Arrival Delays')
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(data_cleaned['DepartureDelay'].dropna(), bins=50, color='salmon', edgecolor='black')
plt.title('Distribution of Departure Delays')
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Delay-Related Features')
plt.show()

print("Classification Report:\n", classification_rep_sample_smaller)
print("\nConfusion Matrix:\n", confusion_mat_sample_smaller)
