import os
from PIL import Image
import base64
from io import BytesIO
import dash
from dash import dcc, html
import csv
import pandas as pd
import plotly.express as px

app = dash.Dash(__name__)

class_folders = ['airplane', 'alarm_clock', 'ant', 'ape', 'apple', 'armor', 'axe', 'banana', 'bat', 'bear', 'bee',
                 'beetle', 'bell', 'bench', 'bicycle', 'blimp', 'bread', 'butterfly', 'cabin', 'camel', 'candle',
                 'cannon', 'car_(sedan)', 'castle', 'cat', 'chair', 'chicken', 'church', 'couch', 'cow', 'crab',
                 'crocodilian', 'cup', 'deer', 'dog', 'dolphin', 'door', 'duck', 'elephant', 'eyeglasses', 'fan',
                 'fish', 'flower', 'frog', 'geyser', 'giraffe', 'guitar', 'hamburger', 'hammer', 'harp', 'hat',
                 'hedgehog', 'helicopter', 'hermit_crab', 'horse', 'hot-air_balloon', 'hotdog', 'hourglass',
                 'jack-o-lantern', 'jellyfish', 'kangaroo', 'knife', 'lion', 'lizard', 'lobster', 'motorcycle',
                 'mouse', 'mushroom', 'owl', 'parrot', 'pear', 'penguin', 'piano', 'pickup_truck', 'pig',
                 'pineapple', 'pistol', 'pizza', 'pretzel', 'rabbit', 'raccoon', 'racket', 'ray', 'rhinoceros',
                 'rifle', 'rocket', 'sailboat', 'saw', 'saxophone', 'scissors', 'scorpion', 'sea_turtle', 'seagull',
                 'seal', 'shark', 'sheep', 'shoe', 'skyscraper', 'snail', 'snake', 'songbird', 'spider', 'spoon',
                 'squirrel', 'starfish', 'strawberry', 'swan', 'sword', 'table', 'tank', 'teapot', 'teddy_bear',
                 'tiger', 'tree', 'trumpet', 'turtle', 'umbrella', 'violin', 'volcano', 'wading_bird', 'wheelchair',
                 'windmill', 'window', 'wine_bottle', 'zebra']

def pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def process_csv_results(csv_file):
    correct_predictions = {}
    incorrect_predictions = {}

    with open(csv_file, 'r') as file:
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

    return df

# Initial processing of CSV file
csv_file = 'results.csv'
results_df = process_csv_results(csv_file)

# Reshape the DataFrame for plotting
melted_df = pd.melt(results_df, id_vars=['Actual_Class'], value_vars=['Correct_Predictions', 'Incorrect_Predictions'],
                    var_name='Prediction', value_name='Count')

# Creating the bar graph using Plotly Express
fig = px.bar(melted_df, x="Actual_Class", y="Count", color="Prediction",
             barmode='group', title="Correct vs Incorrect Predictions by Class")

# Set the layout of the app to include the dropdown, graph, selected class, and images container
app.layout = html.Div([
    dcc.Dropdown(
        id='class-dropdown',
        options=[{'label': class_name, 'value': class_name} for class_name in class_folders],
        value='airplane',
        searchable=True,
        placeholder="Select a class..."
    ),
    dcc.Graph(id='results-graph', figure=fig),  # Graph above images
    html.Div(id='selected-class'),
    html.Div(id='images-container')
])


# Callback to update the graph based on selected class
@app.callback(
    [dash.dependencies.Output('selected-class', 'children'),
     dash.dependencies.Output('images-container', 'children'),
     dash.dependencies.Output('results-graph', 'figure')],
    [dash.dependencies.Input('class-dropdown', 'value')]
)

def update_output_and_images(selected_class):
    if selected_class:
        # Update the selected class message
        selected_class_message = f'You have selected "{selected_class}"'

        # Update the image components
        image_files = [f for f in os.listdir(f'assets/{selected_class}') if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        image_components = []
        for image_file in image_files:
            pil_image = Image.open(os.path.join(f'assets/{selected_class}', image_file))
            image_base64 = pil_image_to_base64(pil_image)
            
            # Read the CSV file to get the predicted class for the current image
            df_results = pd.read_csv('results.csv')
            
            # Drop rows with NaN values in the 'Image_Path' column
            df_results = df_results.dropna(subset=['Image_Path'])
            
            # Filter DataFrame to find rows containing the image file path
            filtered_rows = df_results[df_results['Image_Path'].str.contains(image_file)]
            
            # Check if there are any matching rows and if they contain non-null predicted classes
            if not filtered_rows.empty and not filtered_rows['Predicted_Class_1'].isnull().all():
                predicted_class = filtered_rows['Predicted_Class_1'].iloc[0]
                
                actual_class = selected_class
                if actual_class == predicted_class:
                    # Add a green border or background to indicate correct classification
                    image_components.append(html.Div(html.Img(src=f"data:image/png;base64,{image_base64}"),
                                                      style={'border': '2px solid green', 'padding': '10px'}))
                else:
                    # Add a red border or background to indicate incorrect classification
                    image_components.append(html.Div(html.Img(src=f"data:image/png;base64,{image_base64}"),
                                                      style={'border': '2px solid red', 'padding': '10px'}))
            else:
                # If no matching rows or all predicted classes are null, treat as incorrect classification
                image_components.append(html.Div(html.Img(src=f"data:image/png;base64,{image_base64}"),
                                                  style={'border': '2px solid red', 'padding': '10px'}))

        # Update the graph based on selected class
        filtered_df = results_df[results_df['Actual_Class'] == selected_class]
        melted_df = pd.melt(filtered_df, id_vars=['Actual_Class'], value_vars=['Correct_Predictions', 'Incorrect_Predictions'],
                            var_name='Prediction', value_name='Count')
        updated_fig = px.bar(melted_df, x="Actual_Class", y="Count", color="Prediction",
                             barmode='group', title=f"Correct vs Incorrect Predictions for {selected_class}")
        
        return selected_class_message, image_components, updated_fig
    else:
        return 'Please select a class.', None, None

if __name__ == '__main__':
    app.run_server(debug=True)



