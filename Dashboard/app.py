#!/usr/bin/env python
# coding: utf-8

# In[4]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import pickle
import shap
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.neighbors import KNeighborsClassifier
from zipfile import ZipFile

app = dash.Dash(__name__)

def load_data():
    z = ZipFile("C:/Users/Raider/OneDrive/Bureau/openclassrooms/projet7/p7_00_data/default_risk.zip")
    data = pd.read_csv(z.open('default_risk.csv'), index_col='SK_ID_CURR', encoding='utf-8')

    z = ZipFile("C:/Users/Raider/OneDrive/Bureau/openclassrooms/projet7/p7_00_data/X_sample.zip")
    sample = pd.read_csv(z.open('X_sample.csv'), index_col='SK_ID_CURR', encoding='utf-8')

    description = pd.read_csv("C:/Users/Raider/OneDrive/Bureau/openclassrooms/projet7/p7_00_data/features_description.csv",
                              usecols=['Row', 'Description'], index_col=0, encoding='unicode_escape')

    target = data.iloc[:, -1:]

    return data, sample, target, description


def load_model():
    '''loading the trained model'''
    pickle_in = open('C:/Users/Raider/OneDrive/Bureau/openclassrooms/projet7/p7_00_data/p7_00_models/LGBMClassifier.pkl', 'rb')
    clf = pickle.load(pickle_in)
    return clf


def load_knn(sample):
    # Define and train the k-NN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(sample.iloc[:, :-1], sample.iloc[:, -1])
    return knn


data, sample, target, description = load_data()
id_client = sample.index.values
clf = load_model()
knn = load_knn(sample)


app.layout = html.Div(children=[
    html.H1(children='Dashboard Scoring Credit',
            style={'backgroundColor': 'tomato', 'padding': '10px', 'borderRadius': '10px', 'color': 'white',
                   'textAlign': 'center'}),
    html.P("Credit decision support...", style={'fontSize': '20px', 'fontWeight': 'bold', 'textAlign': 'center'}),

    # SIDEBAR
    html.Div([
        html.H2(children='General Info'),

        html.Label("Client ID"),
        dcc.Dropdown(
            id='client-id',
            options=[{'label': i, 'value': i} for i in id_client],
        ),

        html.Div(id='general-info'),
    ], style={'width': '25%', 'float': 'left', 'padding': '20px'}),

    # HOME PAGE - MAIN CONTENT
    html.Div([
        html.H2(id='customer-id'),
        html.Div(id='customer-info'),

        html.H3(children='Customer file analysis'),
        html.Div(id='customer-analysis'),

        html.H3(children='Feature importance / description'),
        dcc.Graph(id='feature-importance'),

        html.H3(children='Similar customer files display'),
        html.Div(id='similar-customers'),
    ], style={'width': '75%', 'float': 'left', 'padding': '20px'})
])


@app.callback(
    [dash.dependencies.Output('customer-id', 'children'),
     dash.dependencies.Output('customer-info', 'children'),
     dash.dependencies.Output('customer-analysis', 'children'),
     dash.dependencies.Output('feature-importance', 'figure'),
     dash.dependencies.Output('similar-customers', 'children')],
    [dash.dependencies.Input('client-id', 'value')]
)
def update_customer_info(value):
    if value:
        # Customer ID
        customer_id = f"Customer ID selection: {value}"

        # Customer Info
        infos_client = identite_client(data, value)
        if not infos_client.empty:
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

        # Customer Analysis
        prediction = load_prediction(sample, value, clf)
        default_probability = round(float(prediction) * 100, 2)
        customer_analysis = [
            html.P(f"Default Probability: {default_probability}%"),
            # Add decision logic here based on threshold
        ]

        # Feature Importance
        X = sample.iloc[:, :-1]
        X = X[X.index == value]
        number = 5
        explainer = shap.TreeExplainer(load_model())
        shap_values = explainer.shap_values(X)

        if shap_values is not None:
            fig = shap.summary_plot(shap_values[0], X, plot_type="bar", max_display=number, show=False)

            if fig is not None:
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=500)
            else:
                fig = go.Figure()

        else:
            fig = go.Figure()

        # Similar Customers
        similar_customers = update_similar_customers_helper(value, sample, knn)
        similar_customers_output = [
            html.Ul([
                html.Li(f"File {i + 1}: {row['column_name']}")
                for i, row in similar_customers.iterrows()
            ])
        ]

        return customer_id, customer_info, customer_analysis, fig, similar_customers_output

    return "", [], [], go.Figure(), []


def identite_client(data, id):
    data_client = data[data.index == int(id)]
    return data_client


def load_prediction(sample, id, clf):
    X = sample.iloc[:, :-1]
    score = clf.predict_proba(X[X.index == int(id)])[:, 1]
    return score

def update_similar_customers_helper(value, sample, knn):
    similar_customers = load_kmeans(sample, value, knn)
    return similar_customers


if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




