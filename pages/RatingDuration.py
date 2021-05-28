import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from utils import Header

import pandas as pd
import numpy as np
import pathlib
import datetime

import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()


netflix_data = pd.read_csv(DATA_PATH.joinpath('netflix_titles.csv'))

n_data = netflix_data[['date_added']].dropna()
netflix_data['year'] = n_data['date_added'].apply(lambda x: x.split(', ')[-1])
netflix_data['month'] = n_data['date_added'].apply(lambda x: x.lstrip().split(' ')[0])

netflix_tvshows = netflix_data[netflix_data['type'] == 'TV Show']
netflix_movies = netflix_data[netflix_data['type'] == 'Movie']

netflix_movies['duration_min'] = netflix_movies['duration'].map(lambda x : x.split(' ')[0])
netflix_tvshows['duration_season'] = netflix_tvshows['duration'].map(lambda x : x.split(' ')[0])

def rating_fig():
    fig = make_subplots(rows=1, cols=2, specs=[
                        [{'type': 'domain'}, {'type': 'domain'}]])

    fig.add_trace(go.Pie(labels=netflix_movies.rating.value_counts().index, values=netflix_movies.rating.value_counts().values, name="Movie Rating"),
                1, 1)
    fig.add_trace(go.Pie(labels=netflix_tvshows.rating.value_counts().index, values=netflix_tvshows.rating.value_counts().values, name="TV Show Rating"),
                1, 2)

    layout = go.Layout(hovermode='closest',
                    annotations=[dict(text='MOVIES', x=0.17, y=0.5, font_size=13, showarrow=False),
                                    dict(text='TV SHOWS', x= 0.85, y=0.5, font_size=13, showarrow=False)],
                       margin=dict(l=15, r=15, b=10, t=10, pad=2),
                    #xaxis = dict(title = 'Year'),
                    #yaxis = dict(title = 'Months'),
                    template="plotly")


    fig.update_traces(hole=.4, hoverinfo="label+percent+value+name")
    fig.update_layout(layout)
    return fig

def duration_movies():
    x1 = netflix_movies['duration_min'].fillna(0.0).astype(float)
    fig = ff.create_distplot([x1], ['a'], bin_size=0.7,
                            curve_type='normal', colors=["#6ad49b"])
    fig.update_layout(showlegend=False, height=600, margin=dict(l=15, r=15, b=10, t=10, pad=2), hovermode='closest', )
    return fig

def duration_tv():
    trace1 = go.Bar(
        x=netflix_tvshows['duration_season'].value_counts().index, 
        y=netflix_tvshows['duration_season'].value_counts(), 
        name="TV Shows", marker=dict(color="#a678de"),
        text=netflix_tvshows['duration_season'].value_counts(),
        textposition='auto',)
    data = [trace1]
    layout = go.Layout(xaxis=dict(title='Season'), yaxis=dict(title='Number of TV SHOWS'), legend=dict(x=0.1, y=1.1, orientation="h"), height = 500)
    fig = go.Figure(data, layout=layout)
    fig.update_traces(texttemplate='%{text:.2s}', textposition='auto')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', margin=dict(l=15, r=15, b=10, t=10, pad=2), hovermode='closest',)
    return fig

def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 2
            html.Div(
                [
                    # Row 1
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        dcc.Markdown('''**MATURITY RATINGS & CLASSIFICATION ACROSS MOVIES & TV**'''), className="subtitle padded"
                                    ),
                                    #html.Table(make_dash_table(df_current_prices)),
                                    dcc.Graph(
                                        id="pie_2",
                                        figure=rating_fig()
                                    )
                                ],
                                #className="six columns",
                            ),
                            # html.Div(
                            #     [
                            #         html.H6(
                            #             ["Historical Prices"],
                            #             className="subtitle padded",
                            #         ),
                            #         html.Table(make_dash_table(df_hist_prices)),
                            #     ],
                            #     className="six columns",
                            # ),
                        ],
                        className="row ",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(dcc.Markdown('''**DURATION of the MOVIES on NETFLIX**'''), className="subtitle padded"),
                                    dcc.Graph(
                                        id="graph-4",
                                        figure=duration_movies(),
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(dcc.Markdown('''**No. of Seasons of TV SHOWS on NETFLIX**'''),className="subtitle padded",),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id='graph_5',
                                                figure = duration_tv(),
                                                config={"displayModeBar": False},)
                                        ],
                                        #style={"overflow-x": "auto"},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 4
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
