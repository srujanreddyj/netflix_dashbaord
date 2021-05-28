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


def country_trace(df, country, flag = "movie"):
    df["from_us"] = df['country'].fillna("").map(lambda x : 1 if country.lower() in x.lower() else 0)
    small = df[df["from_us"] == 1]
    if flag == "movie":
        small = small[small["duration"] != ""]
    else:
        small = small[small["season_count"] != ""]
    cast = ", ".join(small['cast'].fillna("")).split(", ")
    tags = Counter(cast).most_common(25)
    tags = [_ for _ in tags if "" != _[0]]

    labels, values = [_[0]+"  " for _ in tags], [_[1] for _ in tags]
    trace = go.Bar(y=labels[::-1], x=values[::-1], orientation="h", name="",)
    return trace


titles = ["United States", "", 
          "India", "",
          "United Kingdom", "", 
          "Canada", "", 
          "Spain", "", 
          "Japan"]

def actors():
    traces = []
    titles = ["United States", "", "India", "",
            "United Kingdom", "", "Canada", "", "Spain", "", "Japan"]
    for title in titles:
        if title != "":
            traces.append(country_trace(data2, title))

    fig = make_subplots(rows=3, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.07, vertical_spacing=0.07)
    fig.add_trace(traces[0], 1, 1)
    fig.add_trace(traces[1], 1, 3)
    fig.add_trace(traces[2], 2, 1)
    fig.add_trace(traces[3], 2, 3)
    fig.add_trace(traces[4], 3, 1)
    fig.add_trace(traces[5], 3, 3)

    fig.update_layout(height=1000, showlegend=False, template="plotly_dark",)
    return fig

def directors():
    dir_data = data2[data2["type"] == "Movie"]
    ind_dir = dir_data[dir_data["country"] == "India"]
    us_dir = dir_data[dir_data["country"] == "United States"]

    ind_counter_list = Counter(", ".join(ind_dir['director'].fillna("")).split(", ")).most_common(20)
    us_counter_list = Counter(", ".join(us_dir['director'].fillna("")).split(", ")).most_common(20)

    ind_counter_list = [_ for _ in ind_counter_list if _[0] != ""]
    ilabels = [_[0] for _ in ind_counter_list][::-1]
    ivalues = [_[1] for _ in ind_counter_list][::-1]


    us_counter_list = [_ for _ in us_counter_list if _[0] != ""]
    us_labels = [_[0] for _ in us_counter_list][::-1]
    us_values = [_[1] for _ in us_counter_list][::-1]

    fig = make_subplots(rows=1, cols=3, subplot_titles=['India', "", "USA"],
                        horizontal_spacing=0.07, vertical_spacing=0.07)
    fig.add_trace(go.Bar(y=ilabels, x=ivalues, orientation="h",
                        marker=dict(color="orange")), 1, 1)
    fig.add_trace(go.Bar(y=us_labels, x=us_values, orientation="h",
                        marker=dict(color="green")), 1, 3)

    #data = [trace_ind, trace_us]#layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))
    fig.update_layout(showlegend=False, template="plotly_dark",)

    return fig


def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 6
            html.Div(
                [
                    # Row 1
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        dcc.Markdown('''**TOP ACTORS ACROSS NETFLIX CONTENT ACROSS COUNTRIES**'''), className="subtitle padded"
                                    ),
                                    #html.H6("TOP ACTORS ACROSS NETFLIX CONTENT", className="subtitle padded"),
                                    html.Br([]),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Strong(),
                                                    html.Div(
                                                        dcc.Graph(
                                                            id="graph-12",
                                                            figure=actors(),
                                                            config={"displayModeBar": False},
                                                        )
                                                    ),
                                                ],
                                                #style={"overflow-x": "auto"},
                                            ),
                                        ],
                                        style={"color": "#7a7a7a"},
                                    ),
                                ],
                                className="row",
                            ),
                        ],
                    ),
                    #Row 2
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        dcc.Markdown('''**Directors in India & US Films**'''), className="subtitle padded"
                                    ),
                                    html.Br([]),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Strong(),
                                                    html.Div(
                                                        dcc.Graph(
                                                            id="graph-13",
                                                            figure=directors(),
                                                            config={"displayModeBar": False},
                                                        )
                                                    ),
                                                ],
                                                #style={"overflow-x": "auto"},
                                            ),
                                        ],
                                        style={"color": "#7a7a7a"},
                                    ),
                                ],
                                className="row",
                            ),
                        ],
                    )
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
