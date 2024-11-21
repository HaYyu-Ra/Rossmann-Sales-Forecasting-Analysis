import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import requests

# Load historical sales data
historical_data_path = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\Rossmann_Sales_Forecasting_Project\data\clean_data.csv"
historical_data = pd.read_csv(historical_data_path)

# Initialize Dash app
app = Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Rossmann Pharmaceuticals Sales Dashboard", style={'text-align': 'center'}),

    # Dropdown to select a store
    dcc.Dropdown(
        id='store-selector',
        options=[{'label': f'Store {store}', 'value': store} for store in historical_data['Store'].unique()],
        value=historical_data['Store'].unique()[0],  # Default to first store
        style={'width': '50%'}
    ),
    
    # Graph for historical sales data
    dcc.Graph(id='sales-trend-graph'),
    
    # Button to make a prediction
    html.Button('Get Sales Prediction', id='predict-button', n_clicks=0),
    
    # Prediction result display
    html.Div(id='prediction-output', style={'text-align': 'center', 'font-size': '20px', 'margin-top': '20px'})
])

# Callback to update the sales trend graph based on selected store
@app.callback(
    Output('sales-trend-graph', 'figure'),
    [Input('store-selector', 'value')]
)
def update_graph(selected_store):
    filtered_data = historical_data[historical_data['Store'] == selected_store]
    fig = px.line(filtered_data, x='Date', y='Sales', title=f'Sales Trend for Store {selected_store}')
    fig.update_layout(xaxis_title='Date', yaxis_title='Sales')
    return fig

# Callback to get prediction when the button is clicked
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks'), Input('store-selector', 'value')]
)
def get_sales_prediction(n_clicks, selected_store):
    if n_clicks > 0:
        prediction_data = {
            "Store": selected_store,
            "Day": 15,
            "Month": 11,
            "Year": 2024,
            "Weekday": 4,  # Example weekday
            "Weekend": 0,
            "IsMonthStart": 0,
            "IsMonthEnd": 0,
            "DaysToHoliday": 10,
            "StateHoliday": "0"
        }
        try:
            response = requests.post("http://127.0.0.1:8000/predict_rf/", json=prediction_data)
            if response.status_code == 200:
                prediction = response.json().get('predicted_sales', 'N/A')
                return f"Predicted Sales: {prediction}"
            else:
                return f"Prediction failed: {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

# Run the dashboard
if __name__ == '__main__':
    app.run_server(debug=True)
