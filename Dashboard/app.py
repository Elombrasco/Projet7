from dash import Dash, html, dcc, Input, Output
import pandas as pd
import pickle
from lightgbm import LGBMClassifier
from zipfile import ZipFile
import plotly.express as px

app = Dash(__name__)

def load_data():
    data = pd.read_csv(z.open('X_data.csv'), index_col='SK_ID_CURR', encoding='utf-8')

    sample = pd.read_csv('data/X_sample.csv', index_col='SK_ID_CURR', encoding='utf-8')

    description = pd.read_csv("data/features_description.csv",
                              usecols=['Row', 'Description'], index_col=0, encoding='unicode_escape')

    target = data.iloc[:, -1:]

    return data, sample, target, description


def load_model():
    '''loading the trained model'''
    pickle_in = open('data/LGBMClassifier.pkl', 'rb')
    clf = pickle.load(pickle_in)
    return clf

def identite_client(data, id):
    data_client = data[data.index == int(id)]
    return data_client

def load_prediction(sample, id, clf):
    X = sample.iloc[:, :-1]
    score = clf.predict_proba(X[X.index == int(id)])[:, 1]
    return score

def update_similar_customers_helper(value, sample, knn):     #To be revised
    similar_customers = load_kmeans(sample, value, knn)
    return similar_customers
    
#Load Data
data, sample, target, description = load_data()
id_client = sample.index.values
clf = load_model()


#Design the Dashboard
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

#Add interactivity
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
                fig = px.Figure()

        else:
            fig = px.Figure()

        # Similar Customers
        similar_customers = update_similar_customers_helper(value, sample, knn)
        similar_customers_output = [
            html.Ul([
                html.Li(f"File {i + 1}: {row['column_name']}")
                for i, row in similar_customers.iterrows()
            ])
        ]

        return customer_id, customer_info, customer_analysis, fig, similar_customers_output

    return "", [], [], px.Figure(), []

#All functions that were here were not used and could not be used from here, I moved them to the top

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




