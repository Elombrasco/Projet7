from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import pickle
from lightgbm import LGBMClassifier
import plotly.express as px

app = Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])

#Load the data and model
df = pd.read_csv('data/X_sample.csv', index_col="SK_ID_CURR", encoding="utf-8")
#description_df = pd.read_csv("data/features_description.csv", usecols=['Row', 'Description'], index_col=0, encoding='unicode_escape')
model = pickle.load(open("data/LGBMClassifier.pkl", "rb"))

def predict_selected_customer_score(selected_customer_id):
    customer_df = df[df.index == selected_customer_id]
    X = customer_df.iloc[:, :-1]
    score = model.predict_proba(X[X.index == selected_customer_id])[0 , 1]
    return score*100

def predict_similar_customers_score():     #To be revised
    return None

def get_customer_info():
    gender = df["CODE_GENDER"]
    age = df["DAYS_BIRTH"] #Decimal problem to be resolved
    family_status = df["NAME_FAMILY_STATUS"]
    number_of_children = df["CNT_CHILDREN"]

customers_ids = [{'label': i, 'value': i} for i in sorted(df.index)]

#Design the Dashboard
#After creating the app.layout below, we come back here to design each element of the layout
#Title
dashboard_title = html.H1(
    'Dashboard Scoring Credit',
    style={'backgroundColor': "#5F9EA0", 'padding': '10px', 'borderRadius': '10px', 'color': 'white', 'textAlign': 'center'}
    )

#Customer selection
customer_selection = dbc.Card(
    dbc.CardBody(
    [
        html.H2("Customer Selection"),
        dcc.Dropdown(
            id="customers_ids_dropdown",
            options = customers_ids,
            value = customers_ids[0]["value"]),   #Default value
    ],
    className = "")
    )

#Customer Score
customer_score = dbc.Card(
    dbc.CardBody(
        [
            html.H2("Credit Score"),
            dbc.Progress(id="predicted_score", style = {"height" : "30px"})
        ],
        className = ""
        )
    )

#Customer Information
customer_information = dbc.Card(
    dbc.CardBody( "To be completed")
    )

customer_figures = dbc.Card(
    dbc.CardBody( "To be completed")
    )

all_customers = dbc.Card(
    dbc.CardBody( "To be completed")
    )

some_customers = dbc.Card(
    dbc.CardBody( "To be completed")
    )
"""
#From previprevious work, to be revised
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
app.layout = dbc.Container(     #Same as html.Div but with additional customization
    [
        dashboard_title,
        dbc.Row(
            [
                dbc.Col(customer_selection, width = 6),
                dbc.Col(customer_score, width = 6),
                ]
            ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(customer_information),
                dbc.Col(customer_figures)
                ]
            ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(all_customers),
                dbc.Col(some_customers)
                ]
            )
    ],
    fluid = True
    )

#Add interactivity

#Show prediction for the sellected customer
@app.callback(
    Output("predicted_score", "value"),
    Output("predicted_score", "label"),
    Input("customers_ids_dropdown", "value")
    )
def display_customer_score(customer_id):
    if not customer_id:
        return 0, 0    #Just in case the user removes the id from dropdown
    prediction = round(predict_selected_customer_score(customer_id), 2)
    return prediction, f"{prediction}"

if __name__ == '__main__':
    app.run_server(debug=True)
