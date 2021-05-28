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

from IPython.display import display, Markdown
def bold(string):
    display(Markdown(string))

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()

netflix_data = pd.read_csv(DATA_PATH.joinpath('netflix_titles.csv'))

n_data = netflix_data[['date_added']].dropna()
netflix_data['year'] = n_data['date_added'].apply(lambda x: x.split(', ')[-1])
netflix_data['month'] = n_data['date_added'].apply(lambda x: x.lstrip().split(' ')[0])

netflix_tvshows = netflix_data[netflix_data['type'] == 'TV Show']
netflix_movies = netflix_data[netflix_data['type'] == 'Movie']

col = "year"

vc1 = netflix_movies[netflix_movies['year'] != '2021']['year'].value_counts().reset_index()
vc1 = vc1.rename(columns = {col : "count", "index" : col})
vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))
vc1 = vc1.sort_values(col)

vc2 = netflix_tvshows[netflix_tvshows['year'] != '2021']['year'].value_counts().reset_index()
vc2 = vc2.rename(columns = {col : "count", "index" : col})
vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))
vc2 = vc2.sort_values(col)


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][::-1]
hdf = netflix_data.groupby('year')['month'].value_counts().unstack().fillna(0)[month_order].T




def create_layout(app):
    # Page layouts
    return html.Div(
        [
            html.Div([Header(app)]),
            # page 1
            html.Div(
                [
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5("Product Summary"),
                                    html.Br([]),
                                    html.H6(dcc.Markdown('''An attempt to visualize the Netflix data scrapped through flixable which is available online.
                                    *  This is an extensive visualization of the Netflix content for personal purposes. Netflix is the biggest streaming company in the world.
                                    *  It recently became self-sufficient in terms of finances, meaning NETFLIX doesn't need to borrow money for the future.
                                    * As we all know, NETFLIX content contains more movies than TV shows, but it appears it is rapidly increasing in both the content count.
                                    * NETFLIX is adding content rapidly and also making it available across countries very rapidly which can be seen in the third tab.
                                    * Most of the content in movies or TV shows are of dramas and comedies. 
                                    * Correlations tab shows even more interesting insights across relations between different genres.'''),
                                    style={"color": "#ffffff"},
                                    className="row",),
                                ],
                                className="product",
                            )
                        ],
                        className="row",
                    ),
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [
                                    #html.H6( <b> NETFLIX HAVE MORE MOVIES THAN TV SHOWS </b>, className="subtitle padded"),
                                    #html.Div([html.Span('First Part', style={'color': 'red', 'font-style': 'italic', 'font-weight': 'bold'}), ' Second Part']),
                                    html.H6(
                                        dcc.Markdown('''**NETFLIX HAVE MORE MOVIES THAN TV SHOWS**'''), className="subtitle padded"
                                    ),
                                    dcc.Graph(
                                        id="pie_1",
                                        figure={
                                            "data": [go.Pie(
                                                labels=['Movies', 'TV Shows'],
                                                values=[netflix_movies.type.count(), netflix_tvshows.type.count()],
                                                title='NETFLIX CONTENT',
                                                hoverinfo='value',
                                                textinfo='percent',
                                                textposition='inside',
                                                hole=0.6,
                                                showlegend=True,
                                                marker=dict(colors=['mediumturquoise', 'lightgreen'],
                                                            line=dict(
                                                    color='#000000', width=2),
                                                )
                                            )],
                                            "layout": go.Layout(dict(autosize=True, margin=dict(l=15, r=10, t=0, b=65))),
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(dcc.Markdown(''' * NETFLIX has added more content in 2019 than in 2020. For 2021 it is getting started. Maybe due to COVID, number of releases are less compared to 2019.''')), 
                                    html.H6(dcc.Markdown(''' * NETFLIX also added more content in NOVEMBER month than in another months.''')),
                                    html.H6(dcc.Markdown(''' * Surprisingly there is less content for children compared to adults hence more competition from DISNEY.''')),
                                    html.H6(dcc.Markdown(''' * NETFLIX is targeting INDIA more than any other country. More movies and TV shows are being added at a rapid space than any another country.''')),
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row",
                        style={"margin-bottom": "35px"},
                    ),
                    # Row 5
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        dcc.Markdown('''**CONTENT ADDED TO NETFLIX ACROSS YEARS**'''), className="subtitle padded"
                                    ),
                                    dcc.Graph(
                                        id="graph-1",
                                        figure={
                                            "data": [
                                                go.Scatter(
                                                    x=vc1[col], y=vc1["count"], name="Movies", marker=dict(color="green")),
                                                go.Scatter(
                                                    x=vc2[col], y=vc2["count"], name="TV Shows", marker=dict(color="gold")),
                                            ],
                                            "layout": go.Layout(
                                                hovermode='closest',
                                                template="plotly_dark",
                                                autosize=False,
                                                font={
                                                    "family": "Raleway", "size": 10},
                                                #height=200,
                                                # legend={
                                                #     "x": -0.0228945952895,
                                                #     "y": -0.189563896463,
                                                #     "orientation": "h",
                                                #     "yanchor": "top",
                                                # },
                                                margin={
                                                    "r": 0,
                                                    "t": 20,
                                                    "b": 10,
                                                    "l": 10,
                                                },
                                                showlegend=True,
                                                title="",
                                                #width=330,
                                                xaxis={
                                                    "autorange": True,
                                                    #"range": [-0.5, 4.5],
                                                    "showline": True,
                                                    "title": "Year",
                                                    "type": "category",
                                                },
                                                yaxis={
                                                    "autorange": True,
                                                    #"range": [0, 22.9789473684],
                                                    "showgrid": True,
                                                    "showline": True,
                                                    "title": "Count",
                                                    "type": "linear",
                                                    "zeroline": False,
                                                },
                                            ),
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                #className="six columns",
                            ),
                        ],
                        className='row',
                        style={"margin-bottom": "35px"},
                    ),
                    # Row 6
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        dcc.Markdown('''**CONTENT ADDED TO NETFLIX ACROSS MONTHS**'''), className="subtitle padded"
                                    ),
                                    dcc.Graph(
                                        id="graph-2",
                                        figure={
                                            "data": [
                                                go.Heatmap(
                                                    z=hdf,
                                                    x=hdf.columns,
                                                    y=hdf.index,
                                                    hovertemplate='Year: %{x}<br>Month: %{y}<br>No. of Releases: %{z}<extra></extra>',
                                                    hoverongaps=False,
                                                    colorscale='GnBu')
                                            ],
                                            "layout": go.Layout(hovermode='closest', title='Content added over the years',
                                                                xaxis=dict(
                                                                    title='Year'),
                                                                yaxis=dict(
                                                                    title='Months'),
                                                                template="plotly",
                                                                legend=dict(x=0.1, y=1.1, orientation="h")),
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                #className="six columns",
                            ),
                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
