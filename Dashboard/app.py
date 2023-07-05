#Dashboard App Development
#Version 0.0

import dash_bootstrap_components as dbc
from dash import Dash, html, dcc
import pandas as pd

app = Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])

#Specify the content of the four quadrants (two squares on top, two on the middle and two at the bottom
title = dbc.Col("Dashboard")
top_left = dbc.Col(
    dbc.Card([dbc.CardHeader("Characteristiques du Clients"), dbc.CardBody("Body")])
    )
top_right = dbc.Col(
    dbc.Card([dbc.CardHeader("Characteristiques des autres Clients"), dbc.CardBody("Body")]))
mid_left = dbc.Col(
    dbc.Card([dbc.CardHeader("Score du client"), dbc.CardBody("Body")]))
mid_right = dbc.Col(
    dbc.Card([dbc.CardHeader("Score des autres clients selectionés"), dbc.CardBody("Body")]))
bottom_left = dbc.Col(
    dbc.Card([dbc.CardHeader("Interpretation du score du client"), dbc.CardBody("Body")]))
bottom_right = dbc.Col(
    dbc.Card([dbc.CardHeader("Comparaison du client avec les autres clients"), dbc.CardBody("Body")]))

app.layout = dbc.Container([
    dbc.Row(title),
    dbc.Row([top_left, top_right]),
    dbc.Row([mid_left, mid_right]),
    dbc.Row([bottom_left, bottom_right])
    ]
    ,
    fluid = True #You want youd dbc.Container to fill available horizontal space and resize fluidly
    )

if __name__ == "__main__":
    app.run_server(debug=True)



