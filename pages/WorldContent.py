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

from collections import Counter
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
netflix_data['country'] = netflix_data['country'].astype('str')
country_data = netflix_data['country']

netflix_movies = netflix_data[netflix_data['type'] == 'Movie']
netflix_tvshows = netflix_data[netflix_data['type'] == 'TV Show']

netflix_movies['duration_min'] = netflix_movies['duration'].map(lambda x : x.split(' ')[0])
netflix_tvshows['duration_season'] = netflix_tvshows['duration'].map(lambda x : x.split(' ')[0])

country_counting = pd.Series(dict(Counter(','.join(country_data).replace(' ,',',').replace(', ',',').split(',')))).sort_values(ascending=False)
#country_counting.drop(['NULL'], axis=0, inplace=True)

netflix_data['country'] = netflix_data['country'].dropna().apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(','))
lst_col = 'country'
data2 = pd.DataFrame({col :  np.repeat(netflix_data[col].values, netflix_data[lst_col].str.len())
                    for col in netflix_data.columns.drop(lst_col)}).assign(
                        **{lst_col:np.concatenate(netflix_data[lst_col].values)})[netflix_data.columns.tolist()]
year_country = data2.groupby('year')['country'].value_counts().reset_index(name='counts')

ych = year_country.groupby(['country'], as_index=False)['counts'].agg(sum)
ych.drop(ych.tail(1).index, inplace=True)  # drop last n rows
ych.drop(ych.head(1).index, inplace=True)  # drop last n rows

def world_map():
    import plotly.express as px
    fig = px.scatter_geo(ych, locations="country", color="counts",
                        locationmode='country names', size='counts',
                        hover_name="country",
                        projection="natural earth")
    fig.update_layout(margin=dict(l=15, r=15, b=10, t=10, pad=2), )
    return fig

def whole_pic_map():
    fig = px.choropleth(year_country, locations="country", color="counts",
                        locationmode='country names',
                        animation_frame='year',
                        range_color=[0, 700],
                        color_continuous_scale=px.colors.sequential.OrRd
                        )

    fig.update_layout( margin=dict(l=15, r=15, b=10, t=10, pad=2), )
    return fig

def tunnel_count():
    data = dict(
        number=[1063, 619, 135, 60, 44, 41, 40, 40, 38, 35],
        country=["United States", "India", "United Kingdom", "Canada", "Spain", 'Turkey', 'Philippines', 'France', 'South Korea', 'Australia'])
    #fig = px.funnel(data, x='number', y='country')
    fig = px.funnel(ych.sort_values(by=['counts'], ascending=False),
                    x="counts", y="country", height=2500)
    fig.update_layout(autosize=True, margin=dict(l=15, r=15, b=10, t=10, pad=2))
    return fig

def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 3
            html.Div(
                [
                    # Row 1
                    html.Div(
                        [
                            html.Div(
                                [html.H6(dcc.Markdown('''**Total Content Streaming across the World**'''), className="subtitle padded")],
                                className="twelve columns",
                            )
                        ],
                        className="rows",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(
                                [
                                    #html.P(["Stock style"], style={"color": "#7a7a7a"}),
                                    html.Div(
                                        dcc.Graph(
                                        id="graph-5",
                                        figure=world_map(),
                                        config={"displayModeBar": False},
                                        )
                                    )
                                ],
                                className="twelve columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # Row 3
                    html.Br([]),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        dcc.Markdown('''**Content Streaming across the World over years**'''),
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                        id="graph-6",
                                        figure=whole_pic_map(),
                                        config={"displayModeBar": False},
                                    )
                                ],
                                className=" twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        dcc.Markdown('''**Content present in each Country**'''),
                                        className="subtitle padded",
                                    ),
                                    html.Div(
                                        dcc.Graph(
                                            id="graph-5",
                                            figure=tunnel_count(),
                                            config={"displayModeBar": False},
                                        ), style={'overflowY': 'scroll', 'height': 500}
                                    )
                                ],
                                className=" twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
