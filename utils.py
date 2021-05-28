import dash_html_components as html
import dash_core_components as dcc


def Header(app):
    return html.Div([get_header(app), html.Br([]), get_menu()])


def get_header(app):
    header = html.Div(
        [
            html.Div(
                [
                    html.Img(
                        src=app.get_asset_url("download.png"),
                        className="logo",
                    ),
                    # html.A(
                    #     html.Button("Learn More", id="learn-more-button"),
                    #     href="https://plot.ly/dash/pricing/",
                    # ),
                ],
                className="row",
            ),
            html.Div(
                [
                    html.Div(
                        [html.H5("NETFLIX DATA VISUALIZATION")],
                        className="seven columns main-title",
                    ),
                    html.Div(
                        [
                            dcc.Link(
                                "Full View",
                                href="/visual-data-netflix/full-view",
                                className="full-view-link",
                            )
                        ],
                        className="five columns",
                    ),
                ],
                className="twelve columns",
                style={"padding-left": "0"},
            ),
        ],
        className="row",
    )
    return header


def get_menu():
    menu = html.Div(
        [
            dcc.Link(
                "Overview",
                href="/visual-data-netflix/overview",
                className="tab first",
            ),
            dcc.Link(
                "Duration & Ratings ",
                href="/visual-data-netflix/rating-duration",
                className="tab",
            ),
            dcc.Link(
                "Content & World",
                href="/visual-data-netflix/world-content",
                className="tab",
            ),
            dcc.Link(
                "Genre", href="/visual-data-netflix/genre", className="tab"
            ),
            dcc.Link(
                "Correlations",
                href="/visual-data-netflix/correlations",
                className="tab",
            ),
            dcc.Link(
                "Cast & Directors",
                href="/visual-data-netflix/cast-and-directors",
                className="tab",
            ),
        ],
        className="row all-tabs",
    )
    return menu


def make_dash_table(df):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table
