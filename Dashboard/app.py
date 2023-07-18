from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import requests

app = Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])
server = app.server

#Load the data and model
df = pd.read_csv("data/application_train_sample.csv", index_col="SK_ID_CURR", nrows = 20000)
#Remove nrows for the final data

feature_importances = pd.read_csv("data/feature_importances.csv", names = ["name", "importance"])

#Select features that will be used for comparison
"""treshold = 210 #Choose a value for minimum feature importance
features = [{'label': i, 'value': i} for i, j in zip(feature_importances["name"], 
    feature_importances["importance"]) if j > treshold]"""

features = [{"label":i, "value":i} for i in df.columns]

customers_ids = [{'label': i, 'value': i} for i in sorted(df.index)]

def similar_customers_df(reference_customer_id): #To be revised
    selected_features = [feature["label"] for feature in features]
    reference_customer_df = df.loc[df.index == reference_customer_id, selected_features]
    mask = df[selected_features].apply(lambda row: any(row == reference_customer_df.values[0]), axis=1)
    similar_customers = df[mask]
    return similar_customers


#Design the Dashboard
#After creating the app.layout below, we come back here to design each element of the layout
#Title
dashboard_title = html.H1(
    "Credit Scoring Dashboard",
    style={'backgroundColor': "#5F9EA0", 'padding': '10px', 'borderRadius': '10px', 'color': 'white', 'textAlign': 'center'}
    )

#Customer selection
customer_selection = dbc.Card(
    [
    dbc.CardHeader("Customer Selection"),
    dbc.CardBody(
    [
        dcc.Dropdown(
            id="customers_ids_dropdown",
            options = customers_ids,
            value = customers_ids[0]["value"]),   #Default value
    ]),
    ])

#Customer Score
customer_score = dbc.Card(
    [
    dbc.CardHeader("Default Probability"),
    dbc.CardBody(
        [
            dbc.Progress(id="predicted_score", style = {"height" : "30px"})
        ])
    ])

#Meaning of the probability
def probability_class(default_probability):
    if default_probability < 0:
        return "Unavailable"
    elif default_probability < 20:
        return "Low"
    elif default_probability < 50:
        return "Medium"
    elif default_probability < 80:
        return "High"
    else:
        return "Very high"

#Customer Information
customer_information = dbc.Card(
    dbc.CardBody(
        [
        html.H3("Customer Information"),
        html.Hr(),
        html.H6([html.Span("Age: "), html.Span(id="age_value")]),
        html.H6([html.Span("Marital Status: "), html.Span(id="marital_status_value")]),
        html.H6([html.Span("Gender: "), html.Span(id="gender_value")]),
        html.H6([html.Span("Occupation: "), html.Span(id="occupation_value")]),
        html.H6([html.Span("Income: "), html.Span(id="income_value")])
        ]
        )
    )

#Score Interpretation
score_interpretation = dbc.Card(
    dbc.CardBody(
        [
        html.H3("Score Description"),
        html.Hr(),
        html.H6([html.Span("Loan Amount: "), html.Span(id="loan_amount")]),
        html.H6([html.Span("Probability of Default: "), html.Span(id="likelyhood")]),
        html.H6([html.Span("Explanations: "), html.Span(id="explanations")])
        ]
        )
    )

features_selection = dbc.Card(
    [
    dbc.CardHeader("Features Selection"),
    dbc.CardBody(
    [
        dcc.Dropdown(
            id="features_dropdown",
            options = features,
            value = features[0]["value"]),   #Default value
    ])
    ])

comparison_all_customers = dbc.Card(
    dbc.CardBody(
        [
        html.H3("Comparison with All Customers"),
        dcc.Graph(id="comparison_all")
        ]
        )
    )

comparison_some_customers = dbc.Card(
    dbc.CardBody(
        [
        html.H3("Comparison with Similar customers"),
        dcc.Graph(id="comparison_some")
        ]
        )
    )

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
                dbc.Col(customer_information, width = 4),
                dbc.Col(score_interpretation, width = 8)
                ]
            ),
        html.Br(),
        features_selection,
        html.Br(),
        dbc.Row(
            [
                dbc.Col(comparison_all_customers),
                dbc.Col(comparison_some_customers)
                ]
            ),
        html.Hr()
    ],
    fluid = True
    )

#Add interactivity

#show selected customer information
@callback(
    Output("age_value", "children"),
    Output("marital_status_value", "children"),
    Output("gender_value", "children"),
    Output("income_value", "children"),
    Output("occupation_value", "children"),
    Output("loan_amount", "children"),
    Input("customers_ids_dropdown", "value")
    )
def customer_info(customer_id):
    age = int(df.loc[df.index == customer_id, "DAYS_BIRTH"]/(-365))
    marital_status = df.loc[df.index == customer_id, "NAME_FAMILY_STATUS"]
    gender = df.loc[df.index == customer_id, "CODE_GENDER"]
    income = df.loc[df.index == customer_id, "AMT_INCOME_TOTAL"].values[0]
    occupation = df.loc[df.index == customer_id, "OCCUPATION_TYPE"]
    loan_amount = df.loc[df.index == customer_id, "AMT_CREDIT"].values[0]

    return age, marital_status, gender, f"$ {income:,.0f}", occupation, f"$ {loan_amount:,.0f}"

#Show prediction for the selected customer
@callback(
    Output("predicted_score", "value"),
    Output("predicted_score", "label"),
    Input("customers_ids_dropdown", "value")
    )
def display_customer_score(customer_id):
    if not customer_id:
        return 0, 0  # Just in case the user removes the id from dropdown

    # Prepare the JSON message
    message = {"customer_id": customer_id}

    try:
        # Make the API request
        response = requests.post("http://127.0.0.1:8000/predict", json=message)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            prediction = response.json()["score"]
            return prediction, f"{prediction}"
        else:
            return -1, -1
    except requests.exceptions.RequestException as e:
        return -2, -2

#Score Interpretation
@callback(
    Output("likelyhood", "children"),
    Input("predicted_score", "value"))
def describe_score(customer_score):
    return probability_class(customer_score)

@callback(
    Output("explanations", "children"),
    Input("predicted_score", "value")
    )
def score_explanation(customer_score):
    explain = {
    "Low" : f"With a score of {customer_score}, much less than the 50 percent treshold, this customer is very likely to repay their loan.",
    "Medium" : f"The customer scored {customer_score}, less than the 50 percent treshold, they are quiet likely to repay their loan.",
    "High" : f"The customer has a score of {customer_score}, higher than the 50 percent treshold, they are not likely to repay their loan.",
    "Very high" : f"This customer, with a score of {customer_score}, much more than the 50 percent treshold, will not repay their loan.",
    "Unavailable" : "No data available at this time."
    }
    return explain[probability_class(customer_score)]

@callback(
    Output("comparison_all", "figure"),
    Input("features_dropdown", "value")
    )
def graph_comparison_with_all_customers(selected_features):
    fig = px.histogram(df, x=selected_features, color="TARGET", nbins =20, barmode="group")

    fig.update_layout({
        "plot_bgcolor":"rgba(0,0,0,0)", 
        "paper_bgcolor":"rgba(0,0,0,0)"})
    fig.update_xaxes(showline=True, linecolor='black')
    fig.update_yaxes(showline=True, linecolor='black')
    fig.update_traces(textposition='outside')
    return fig;

@callback(
    Output("comparison_some", "figure"),
    Input("features_dropdown", "value"),
    Input("customers_ids_dropdown", "value")
    )
def graph_comparison_with_similar_customers(selected_features, reference_customer):
    similar_df = similar_customers_df(reference_customer)
    fig = px.histogram(similar_df, x=selected_features, color="TARGET", nbins =20, barmode="group")

    fig.update_layout({'plot_bgcolor':'rgba(0,0,0,0)', 'paper_bgcolor':'rgba(0,0,0,0)'})
    fig.update_xaxes(showline=True, linecolor='black')
    fig.update_yaxes(showline=True, linecolor='black')
    fig.update_traces(textposition='outside')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)


