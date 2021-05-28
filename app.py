# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from pages import (
    overview,
    RatingDuration,
    WorldContent,
    Genre,
    correlations,
    Cast,
)

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

# Describe the layout/ UI of the app
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)

# Update page
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/visual-data-netflix/rating-duration":
        return RatingDuration.create_layout(app)
    elif pathname == "/visual-data-netflix/world-content":
        return WorldContent.create_layout(app)
    elif pathname == "/visual-data-netflix/genre":
        return Genre.create_layout(app)
    elif pathname == "/visual-data-netflix/correlations":
        return correlations.create_layout(app)
    elif pathname == "/visual-data-netflix/cast-and-directors":
        return Cast.create_layout(app)
    elif pathname == "/visual-data-netflix/full-view":
        return (
            overview.create_layout(app),
            RatingDuration.create_layout(app),
            WorldContent.create_layout(app),
            Genre.create_layout(app),
            correlations.create_layout(app),
            Cast.create_layout(app),
        )
    else:
        return overview.create_layout(app)


if __name__ == "__main__":
    app.run_server(debug=True)
