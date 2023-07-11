from dash import Dash, html, dcc, Input, Output
import pandas as pd
import pickle
from lightgbm import LGBMClassifier
import plotly.express as px

app = Dash(__name__)

#Load the data and model
df = pd.read_csv('data/X_sample.csv', index_col="SK_ID_CURR", encoding="utf-8")
target_column = df.TARGET

#description_df = pd.read_csv("data/features_description.csv", usecols=['Row', 'Description'], index_col=0, encoding='unicode_escape')

model = pickle.load(open("data/LGBMClassifier.pkl", "rb"))


def predict_selected_customer_score(selected_customer_id):
    customer_df = df[df.index == selected_customer_id]
    X = customer_df.iloc[:, :-1]                                    ##Clarify, this would mean we don't predictions, we should simply take the y value
    score = model.predict_proba(X[X.index == selected_customer_id])[:, 1]
    return score

def predict_similar_customers_score():     #To be revised
    #One idea is to allow the selection of features and compare the score of the selected customer with the score of
    #all other customer with the same value for the selected feature or features
    return None


#Design the Dashboard
#After creating the app.layout below, we come back here to design each element of the layout
#Title
dashboard_title = html.H1(
    'Dashboard Scoring Credit',
    style={'backgroundColor': 'tomato', 'padding': '10px', 'borderRadius': '10px', 'color': 'white', 'textAlign': 'center'}
    )

#Customer selection and information side
top_right_display = html.Div(
    [
        html.H1('Customer Information'),
        html.Label("Customer ID"),
        dcc.Dropdown(
            id="customers_ids_dropdown",
            options=[
                {'label': i, 'value': i} for i in sorted(df.index)],
            value = df.index[0]),   #Default value
        html.H2("Selected Customer Score"),
        html.H2(id="predicted_score"),
    ],
    style={'width': '25%', 'float': 'left', 'padding': '20px'}
    )
"""
#Body
main_content = html.Div(
    [
        html.H2(id='customer-id'),
        html.Div(id='customer-info'),

        html.H3(children='Customer file analysis'),
        html.Div(id='customer-analysis'),

        html.H3(children='Feature importance / description'),
        dcc.Graph(id='feature-importance'),

        html.H3(children='Similar customer files display'),
        html.Div(id='similar-customers'),
    ],
    style={'width': '75%', 'float': 'left', 'padding': '20px'})
"""
#This is the app Layout
app.layout = html.Div(
    [
        dashboard_title,
        html.P("Credit decision support.", style={'fontSize': '20px', 'fontWeight': 'bold', 'textAlign': 'center'}),
        top_right_display,
    ])

#Add interactivity

#Show prediction for the sellected customer
@app.callback(
    Output("predicted_score", "children"),
    Input("customers_ids_dropdown", "value")
    )
def display_customer_score(customer_id):
    prediction = predict_selected_customer_score(customer_id)
    return prediction

"""
@app.callback(
    [Output('customer-id', 'children'),
     Output('customer-info', 'children'),
     Output('customer-analysis', 'children'),
     Output('feature-importance', 'figure'),
     Output('similar-customers', 'children')],
    [Input('client-id', 'value')]
)
def update_customer_info(value):
    if value:
        # Customer ID
        customer_id = f"Customer ID selection: {value}"

        # Customer Info
        infos_client = customer_data(data, value)
        if not infos_client.empty():
            gender = infos_client["CODE_GENDER"].values[0]
            age = int(infos_client["DAYS_BIRTH"] / 365)
            family_status = infos_client["NAME_FAMILY_STATUS"].values[0]
            children = int(infos_client["CNT_CHILDREN"].values[0])
            customer_info = [
                html.P(f"Gender: {gender}"),
                html.P(f"Age: {age}"),
                html.P(f"Family Status: {family_status}"),
                html.P(f"Number of Children: {children}"),
            ]
        else:
            customer_info = []
        return customer_id, customer_info, customer_analysis, fig, similar_customers_output
"""

if __name__ == '__main__':
    app.run_server(debug=True)
