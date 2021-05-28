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

from sklearn.preprocessing import MultiLabelBinarizer

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

def cat_pie_movies():
    top_movies_genres = [
        'International Movies',
        'Dramas',
        'Comedies',
        'Documentaries',
        'Action & Adventure',
    ]

    net_movies = netflix_data.copy()
    net_movies = net_movies[net_movies['type'] == 'Movie']
    net_movies['year'] = net_movies['year'].astype('int')
    net_movies['Genres'] = net_movies['Genres'].astype('str')

    #net_movies['principal_genre'] = net_movies['Genres'].apply(lambda genres: Genres[0])
    #net_movies['principal_genre'].head()
    year_genre_df = net_movies[(net_movies['Genres'].isin(top_movies_genres)) & (net_movies['year'] >= 2015) & (net_movies['year'] < 2021)].groupby(['Genres', 'year']).agg({'title': 'count'})
    year_genre_df = year_genre_df.reset_index()
    year_genre_df.columns = ['Genres', 'year', 'count']

    fig = px.sunburst(year_genre_df, path=['year', 'Genres'], values='count')
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
        template="plotly")
    fig.update_layout(layout)
    return fig

def cat_pie_tv():
    top_movies_genres = [
        'International Movies',
        'Dramas',
        'Comedies',
        'Documentaries',
        'Action & Adventure',
    ]

    tv_movies = netflix_data.copy()
    tv_movies = tv_movies[tv_movies['type'] == 'TV Show']
    tv_movies['year'] = tv_movies['year'].fillna(0)
    tv_movies['year'] = tv_movies['year'].astype("Int32")
    tv_movies['Genres'] = tv_movies['Genres'].astype('str')

    #tv_movies['principal_genre'] = tv_movies['Genres'].apply(lambda genres: Genres[0])
    #tv_movies['principal_genre'].head()
    year_genre_df = tv_movies[(tv_movies['Genres'].isin(top_movies_genres)) & (tv_movies['year'] >= 2015) & (tv_movies['year'] < 2021)].groupby(['Genres', 'year']).agg({'title': 'count'})
    year_genre_df = year_genre_df.reset_index()
    year_genre_df.columns = ['Genres', 'year', 'count']

    fig = px.sunburst(year_genre_df, path=['year', 'Genres'], values='count')
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
        template="plotly")
    fig.update_layout(layout)
    return fig

def relation_heatmap(df):
    df_corr_heat= df['listed_in'].astype(str).map(lambda s : s.replace('&',' ').replace(',', ' ').split()) 

    test = df_corr_heat
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_)
    corr = res.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    return corr

def corr_map_movies():
    mov_corr = relation_heatmap(netflix_movies)

    fig_hm_corr = go.Figure(
        data=go.Heatmap(
            z=mov_corr,
            x=mov_corr.columns,
            y=mov_corr.index,
            #hovertemplate='Year: %{x}<br>Month: %{y}<br>No. of Releases: %{z}<extra></extra>',
            hoverongaps = False,
            colorscale = 'GnBu'))

    layout = go.Layout(hovermode= 'closest', title = 'Categories Correlation' , 
                    xaxis = dict(title = 'Genres'), 
                    yaxis = dict(title = 'Genres'),
                    template= "plotly_white",
                    legend=dict(x=0.1, y=1.1, orientation="h"),
                    height=800
    )                   
                                        
    fig_hm_corr.update_layout(layout)
    return fig_hm_corr

def corr_map_tv():
    tv_corr = relation_heatmap(netflix_tvshows)

    fig_hm_corr = go.Figure(
        data=go.Heatmap(
            z=tv_corr,
            x=tv_corr.columns,
            y=tv_corr.index,
            #hovertemplate='Year: %{x}<br>Month: %{y}<br>No. of Releases: %{z}<extra></extra>',
            hoverongaps = False,
            colorscale = 'GnBu'))

    layout = go.Layout(hovermode= 'closest', title = 'Categories Correlation' , 
                    xaxis = dict(title = 'Genres'), 
                    yaxis = dict(title = 'Genres'),
                    template= "plotly_white",
                    legend=dict(x=0.1, y=1.1, orientation="h"),
                    height=800
                )                   
                                        
    fig_hm_corr.update_layout(layout)
    return fig_hm_corr


def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 5
            html.Div(
                [
                    # Row 1
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(dcc.Markdown('''**GENRES ACROSS YEARS**'''), className="subtitle padded"),
                                    html.Div(
                                        [
                                            html.Strong(),
                                            html.Div(
                                                dcc.Graph(
                                                    id="graph-8",
                                                    figure=cat_pie_movies(),
                                                    config={
                                                        "displayModeBar": False},
                                                )
                                            ),
                                        ],
                                        className="six columns",
                                    ),
                                    html.Div(
                                        [
                                            html.Strong(),
                                            html.Div(
                                                dcc.Graph(
                                                    id="graph-9",
                                                    figure=cat_pie_tv(),
                                                    config={
                                                        "displayModeBar": False},
                                                )
                                            ),
                                        ],
                                        className="six columns",
                                    ),
                                ],
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
                                    html.Br([]),
                                    html.H6(dcc.Markdown('''**GENRES CORRELATION ACROSS MOVIES**'''), className="subtitle padded"),
                                    html.Div(
                                        [
                                            html.Strong(),
                                            html.Div(
                                                dcc.Graph(
                                                    id="graph-10",
                                                    figure=corr_map_movies(),
                                                    config={
                                                        "displayModeBar": False},
                                                )
                                            ),
                                        ],
                                        #style={"overflow-x": "auto"},
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
                                    html.H6(dcc.Markdown('''**GENRES CORRELATION ACROSS TV SHOWS**'''), className="subtitle padded"),
                                    html.Div(
                                        [
                                            html.Strong(),
                                            html.Div(
                                                dcc.Graph(
                                                    id="graph-11",
                                                    figure=corr_map_tv(),
                                                    config={
                                                        "displayModeBar": False},
                                                )
                                            ),
                                        ],
                                        #style={"overflow-x": "auto"},
                                    ),
                                ],
                                className=" twelve columns",
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
