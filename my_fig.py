import csv
import pandas as pd
import plotly.express as px

# Open the CSV file and initialize dictionaries to store counts
correct_predictions = {}
incorrect_predictions = {}

with open('results.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        actual_class = row[1]
        predicted_class_1 = row[2]
        # Initialize counts for the class if not already present
        correct_predictions.setdefault(actual_class, 0)
        incorrect_predictions.setdefault(actual_class, 0)
        # Increment counts based on prediction correctness
        if actual_class == predicted_class_1:
            correct_predictions[actual_class] += 1
        else:
            incorrect_predictions[actual_class] += 1

# Convert dictionaries to DataFrame
df_correct = pd.DataFrame(correct_predictions.items(), columns=['Actual_Class', 'Correct_Predictions'])
df_incorrect = pd.DataFrame(incorrect_predictions.items(), columns=['Actual_Class', 'Incorrect_Predictions'])

# Merge DataFrames and fill missing values with 0
df = pd.merge(df_correct, df_incorrect, on='Actual_Class', how='outer').fillna(0)

# Reshape the DataFrame for plotting
melted_df = pd.melt(df, id_vars=['Actual_Class'], value_vars=['Correct_Predictions', 'Incorrect_Predictions'],
                    var_name='Prediction', value_name='Count')

# Creating the bar graph using Plotly Express
fig = px.bar(melted_df, x="Actual_Class", y="Count", color="Prediction",
             barmode='group', title="Correct vs Incorrect Predictions by Class")
fig.show()