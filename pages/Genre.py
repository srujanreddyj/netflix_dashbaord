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

##Movie Categories
mcounter_list = Counter(", ".join(netflix_movies['listed_in']).split(", ")).most_common(50)
mlabels = [_[0] for _ in mcounter_list][::-1]
mvalues = [_[1] for _ in mcounter_list][::-1]
#trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="#a678de"))

#TV Categories
tcounter_list = Counter(", ".join(netflix_tvshows['listed_in']).split(", ")).most_common(50)
tlabels = [_[0] for _ in tcounter_list][::-1]
tvalues = [_[1] for _ in tcounter_list][::-1]
#trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="#a678de"))

netflix_data['Genres'] = netflix_data['listed_in'].str.extract('([A-Z]\w{2,})', expand=True)
temp_df = netflix_data['Genres'].value_counts().reset_index()

sizes=np.array(temp_df['Genres'])
labels=temp_df['index']
#colors = [plt.cm.Paired(i/float(len(labels))) for i in range(len(labels))]

def top_categories():
    fig = make_subplots(rows=2, cols=1, horizontal_spacing=0.07, vertical_spacing=0.07, subplot_titles=("MOVIES", "TV SHOW",))
    fig.add_trace(go.Bar(y=mlabels, x=mvalues, orientation="h", name="Movies",), 1, 1)
    fig.add_trace(go.Bar(y=tlabels, x=tvalues, orientation="h", name="TV Shows",), 2, 1)
    #data = [trace1]
    # #layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))
    fig.update_layout(height=1000, template="plotly_dark", hovermode='closest', showlegend=False)

    #fig = go.Figure(data, layout=layout)
    return fig

def heat_cat():
    dfg = netflix_data[['type', 'Genres']]
    dfg['values'] = 1

    fig = px.treemap(dfg, path=['type', 'Genres'], values='values',
                    color_continuous_scale='PuBuGn',
                    )

    layout = go.Layout(
        #autosize=True,
        #width=750,
        #height=1000,
        margin=dict(
            l=15,
            r=15,
            b=10,
            t=10,
            pad=2
        ),
        hovermode='closest', 
        #title='CATEGORIES',
        template="plotly_dark")
    fig.update_layout(layout)

    return fig

def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 4
            html.Div(
                [
                    # Row 1
                    html.Div(
                        [
                            html.Div(
                                [html.H6(dcc.Markdown('''**MOVIE and TV SHOWS GENRE**'''), className="subtitle padded")],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 2
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Strong(),
                                    html.Div(
                                        dcc.Graph(
                                            id="graph-6",
                                            figure=top_categories(),
                                            config={"displayModeBar": False},
                                        )
                                    ),
                                ],
                                className="twelve columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Br([]),
                                    html.H6(dcc.Markdown('''**MOVIE and TV SHOWS GENRE COMPARSION**'''), className="subtitle padded"),
                                    html.Br([]),
                                    html.Div(
                                        [
                                            html.Strong(),
                                            html.Div(
                                                dcc.Graph(
                                                    id="graph-7",
                                                    figure=heat_cat(),
                                                    config={
                                                        "displayModeBar": False},
                                                )
                                            ),
                                        ],
                                        className="twelve columns",
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
