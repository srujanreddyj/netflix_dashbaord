import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc

import datetime

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import country_converter as coco
import pycountry
import squarify
import matplotlib.pyplot as plt

from IPython.display import display, Markdown
def bold(string):
    Markdown(string)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# https://www.bootstrapcdn.com/bootswatch/
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

app.config.suppress_callback_exceptions = True

netflix_data = pd.read_csv('netflix_titles.csv')

n_data = netflix_data[['date_added']].dropna()
netflix_data['year'] = n_data['date_added'].apply(lambda x: x.split(', ')[-1])
netflix_data['month'] = n_data['date_added'].apply(lambda x: x.lstrip().split(' ')[0])


#netflix_movies['duration_min'] = netflix_movies['duration'].map(lambda x : x.split(' ')[0])
#netflix_tvshows['duration_season'] = netflix_tvshows['duration'].map(lambda x : x.split(' ')[0])



app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H1("Stock Market Dashboard",
                        className='text-center text-primary mb-4'),
                width=12)
    ),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='cluster_graph', figure={})
        ],  # width={'size':5, 'offset':1, 'order':1},
            xs=12, sm=12, md=12, lg=5, xl=5
        ),
        # dbc.Col([
        #     dcc.Graph(id='line-fig2', figure={})
        # ],  # width={'size':5, 'offset':0, 'order':2},
        #     xs=12, sm=12, md=12, lg=5, xl=5
        # ),
    ], no_gutters=True, justify='start'),  # Horizontal:start,center,end,between,around
])

@app.callback(
    Output("cluster_graph", "figure"),
    [Input("df", "data")],
)
def content_produced(df):
    netflix_data = pd.read_csv('netflix_titles.csv')

    n_data = netflix_data[['date_added']].dropna()
    netflix_data['year'] = n_data['date_added'].apply(lambda x: x.split(', ')[-1])
    netflix_data['month'] = n_data['date_added'].apply(lambda x: x.lstrip().split(' ')[0])

    netflix_tvshows = netflix_data[netflix_data['type'] == 'TV Show']
    netflix_movies = netflix_data[netflix_data['type'] == 'Movie']
    


    trace = go.Pie(
        labels=['Movies', 'TV Shows'],
        values=[netflix_movies.type.count(), netflix_tvshows.type.count()],
        title='NETFLIX CONTENT',
        hoverinfo='value',
        textinfo='percent',
        textposition='inside',
        hole=0.6,
        showlegend=True,
        marker=dict(colors=['mediumturquoise', 'lightgreen'],
                    line=dict(color='#000000', width=2),
                )
    )

    #layout = go.Layout(title=bold("**NETFLIX HAVE MORE MOVIES THAN TV SHOWS**"))
    layout = dict(autosize=True, margin=dict(l=15, r=10, t=0, b=65))
    return go.Figure(data=[trace],layout=layout)


if __name__ == '__main__':
    app.run_server(debug=True)
