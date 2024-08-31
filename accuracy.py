import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers
from datetime import datetime

# Load the dataset
file_path = 'weather_data.csv'
weather_data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
weather_data['Date'] = pd.to_datetime(weather_data['Date'])

# Handle missing values by filling them with the mean of their respective columns
weather_data['Temp Max'].fillna(weather_data['Temp Max'].mean(), inplace=True)
weather_data['Temp Min'].fillna(weather_data['Temp Min'].mean(), inplace=True)

# Extract year, month, and day as new features
weather_data['Year'] = weather_data['Date'].dt.year
weather_data['Month'] = weather_data['Date'].dt.month
weather_data['Day'] = weather_data['Date'].dt.day

# Drop unnecessary columns
weather_data.drop(columns=['_id', 'Date'], inplace=True)

# Define features and target variables
X = weather_data[['Rain', 'Year', 'Month', 'Day']]
y_max = weather_data['Temp Max']
y_min = weather_data['Temp Min']

# Split the data into training and testing sets
X_train, X_test, y_max_train, y_max_test, y_min_train, y_min_test = train_test_split(X, y_max, y_min, test_size=0.2, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a more complex model to predict Temp Max
model_max = tf.keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Compile the model with a lower learning rate
model_max.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Build a more complex model to predict Temp Min
model_min = tf.keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Compile the model with a lower learning rate
model_min.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the models with a smaller batch size and more epochs
history_max = model_max.fit(X_train_scaled, y_max_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)
history_min = model_min.fit(X_train_scaled, y_min_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate the models on the test set
y_max_pred = model_max.predict(X_test_scaled)
y_min_pred = model_min.predict(X_test_scaled)

mae_max = mean_absolute_error(y_max_test, y_max_pred)
mae_min = mean_absolute_error(y_min_test, y_min_pred)

r2_max = r2_score(y_max_test, y_max_pred)
r2_min = r2_score(y_min_test, y_min_pred)

print(f"Temp Max Model - MAE: {mae_max}, R2: {r2_max}")
print(f"Temp Min Model - MAE: {mae_min}, R2: {r2_min}")

# Check if models meet accuracy criteria
if r2_max < 0.7 or r2_min < 0.7:
    print("Accuracy still below 70%. Further tuning may be required.")
else:
    print("Models meet the accuracy requirement. Proceeding to save and convert them.")

# Save the models in .h5 format if accuracy is sufficient
if r2_max >= 0.7 and r2_min >= 0.7:
    model_max.save('model_max.h5')
    model_min.save('model_min.h5')

    # Convert the models to TensorFlow Lite formatw
    converter_max = tf.lite.TFLiteConverter.from_keras_model(model_max)
    tflite_model_max = converter_max.convert()

    converter_min = tf.lite.TFLiteConverter.from_keras_model(model_min)
    tflite_model_min = converter_min.convert()

    # Save the .tflite models
    with open('model_max.tflite', 'wb') as f:
        f.write(tflite_model_max)

    with open('model_min.tflite', 'wb') as f:
        f.write(tflite_model_min)

    print("Models have been successfully converted to TensorFlow Lite format.")

# User input for testing the model predictions
user_start_date = input("Enter the start date (YYYY-MM-DD): ")
user_rain = float(input("Enter the amount of rain (in mm): "))

# Function to predict the next five days of temperatures
def predict_next_five_days(rain, start_date_str):
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    year = start_date.year
    month = start_date.month
    start_day = start_date.day
    
    predictions = []
    for day in range(start_day, start_day + 5):
        input_data = scaler.transform([[rain, year, month, day]])
        temp_max_pred = model_max.predict(input_data)
        temp_min_pred = model_min.predict(input_data)
        predictions.append({
            'Day': f'{year}-{month:02d}-{day:02d}',
            'Temp Max': temp_max_pred[0][0],
            'Temp Min': temp_min_pred[0][0],
            'Alert': 'Extreme Heat' if temp_max_pred[0][0] > 35 else 'Normal'
        })
    return predictions

# Make predictions based on the user input
predictions = predict_next_five_days(rain=user_rain, start_date_str=user_start_date)
for pred in predictions:
    print(pred)