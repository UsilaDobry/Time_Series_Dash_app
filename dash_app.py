# Local resources
import tsa
import markdown
# dash components
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Dash, dash_table, State
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import os

app = Dash(name=__name__, external_stylesheets=[dbc.themes.SOLAR])
server = app.server
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

#-----------------------------------------------------КОМПОНЕНТЫ--------------------------------------------------------
#-------------------------------------------------------HEADER----------------------------------------------------------
logo = html.Img(src=app.get_asset_url('./Logo_Yandex/ya_praktikum_2.jpg'), style={'width': "200px", 'height': "100x"},
                className='inline-image')
header_1 = html.H1(children="DashBoard Machine Learning", style={'text-transform': "uppercase"})
header_2 = html.H3(children="Анализ временных рядов", style={'text-transform': "uppercase"})

#--------------------------------------------------------CONTENT--------------------------------------------------------
#--------------------------------------------------ПОСТАНОВКА ЗАДАЧИ----------------------------------------------------
tab1_content = dbc.Card(
    dbc.CardBody(
        [
            dcc.Markdown(children=markdown.task_text)
        ]
    ),
    className="mt-3",
)
# ---------------------------------------------------------ETL----------------------------------------------------------
loading = html.Div(
    [
        html.Br(),
        html.H4("Данные временного ряда"),
        html.Br(),
        dash_table.DataTable(data=tsa.data.to_dict('records'),
                                 columns=[{'id':c, 'name':c} for c in tsa.data.columns],
                                 page_size=10,
                                 style_header={
                                     'backgroundColor': 'rgd(30,30,30)',
                                     'color': 'black'
                                 },
                                 style_data={
                                     'backgroundColor': 'rgb(50,50,50)',
                                     'color': 'white'
                                 },
                                 sort_action='native',
                                 tooltip_header={
                                     'datetime': 'Дата заказа такси',
                                     'num_orders': 'Количество заказов такси'
                                 }),
    ]
)
resampling = html.Div(
    [
        html.Br(),
        html.H4("Понижение частоты временного ряда"),
        html.Br(),
        dcc.Graph(id='frequency-graph', figure=tsa.fig_freq_1h),
    ]
)
decomposition = html.Div(
    [
        html.Br(),
        html.H4("Выделение трендовой и сезонной составляющих"),
        html.Br(),
        dcc.Graph(id='trend-graph', figure=tsa.fig_trend),
        dcc.Graph(id='seasonal-graph', figure=tsa.fig_seasonal),
    ]
)
result_adf = html.Div(id='adf_div')
result_kpss = html.Div(id='kpss_div')
check = html.Div(
    [
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H4("Проверка на монотонность"),
                                            dbc.Button('Проверим', id='mono_btn'),
                                            html.Hr(),
                                            html.Div('Результат проверки', id='mono_div',
                                                style={'text-transform': "uppercase", 'margin-left':'0px'},
                                                className='text-primary'),
                                        ]
                                    ),
                                ]
                            ),
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.H4('Результат тестирования'),
                                            dbc.Tabs(
                                                [
                                                    dbc.Tab(result_adf, tab_id='res_adf', label='ADF test'),
                                                    dbc.Tab(result_kpss, tab_id='res_kpss', label='KPSS test'),

                                                ],
                                                id='tabs_result_testing',
                                                active_tab='res_adf'
                                            ),
                                        ]
                                    )
                                ]
                            ),
                        ],
                        style={'max-width': '35%'}
                    ),

                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    html.H4("Проверка на стационарность"),
                                                    html.Br(),
                                                    html.Div('Результат проверки по итогам двух тестов', id='stationarity_div',
                                                            style={'text-transform': "uppercase", 'margin-left':'0px'},
                                                            className='text-primary'),
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Br(),
                                                            html.H5("Тест Dickey-Fuller"),
                                                            html.Br(),
                                                            dcc.Markdown(children=markdown.adf_test_text),
                                                            html.Br(),
                                                            dbc.Button('Тестируем', id='adf_btn'),
                                                        ]
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Br(),
                                                            html.H5("Тест KPSS"),
                                                            html.Br(),
                                                            dcc.Markdown(children=markdown.kpss_test_text),
                                                            html.Br(),
                                                            dbc.Button('Тестируем', id='kpss_btn'),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                        ]
                                    )
                                ]
                            ),
                        ]
                    ),
                ]
            )
        ),
    ]
)
construction = html.Div(
    [
        html.Br(),
        html.H4("Конструирование признаков"),
        html.Br(),
        dash_table.DataTable(data=tsa.df.to_dict('records'),
                        columns=[{'id':c, 'name':c} for c in tsa.df.columns],
                        page_size=10,
                        style_header={
                            'backgroundColor': 'rgd(30,30,30)',
                            'color': 'black'
                        },
                        style_data={
                            'backgroundColor': 'rgb(50,50,50)',
                            'color': 'white'
                        },
                        sort_action='native',
                        tooltip_header={
                            'trend': 'Тренд количества заказов такси',
                            'seasonal': 'Сезонность количества заказов такси',
                            'month': 'Месяц',
                            'hour': 'Час',
                            'dayofweek': 'День недели',
                            'rolling_mean_D': 'Дневная скользящая средняя',
                            'rolling_mean_W': 'Недельная скользящая средняя'
                        }
        ),
    ]
)
conclusion_preprocessing = html.Div(
    [
        html.Br(),
        dcc.Markdown(children=markdown.conclusion_preprocessing_text)
    ]
)

tab2_content = dbc.Card(
    dbc.CardBody(
        [
            html.H3("Получение, преобразование и загрузка данных (ETL)"),
            dbc.Tabs(
                [
                dbc.Tab(loading, tab_id='loading', label='Загрузка данных'),
                dbc.Tab(resampling, tab_id='resampling', label='Ресемплирование данных'),
                dbc.Tab(decomposition, tab_id='decomposition', label='Декомпозиция данных'),
                dbc.Tab(check, tab_id='check', label='Проверка данных'),
                dbc.Tab(construction, tab_id='construction', label='Конструирование данных'),
                dbc.Tab(conclusion_preprocessing, tab_id='conclusion_preproc', label='Вывод'),
                ],
                id='tabs_preprocessing',
                active_tab='loading'
            ),
        ]
    ),
    className="mt-3",
)
# --------------------------------------------------------EDA-----------------------------------------------------------
total_trand = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id='total-trand-graph', figure=tsa.fig_total_trand),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.H4("Интерпретация графика"),
                                html.Br(),
                                dcc.Markdown(),
                            ],
                            style={'max-width': '35%'}
                        ),
                    ]
                )
            ]
        ),
    ]
)
time_distribution = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id='distribution-1h-graph', figure=tsa.fig_hour),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.H4("Интерпретация графика"),
                                html.Br(),
                                dcc.Markdown(),
                            ],
                            style={'max-width': '35%'}
                        ),
                    ]
                )
            ]
        ),
    ]
)
day_distribution = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id='distribution-weekday-graph', figure=tsa.fig_week),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.H4("Интерпретация графика"),
                                html.Br(),
                                dcc.Markdown(),
                            ],
                            style={'max-width': '35%'}
                        ),
                    ]
                )
            ]
        ),
    ]
)
month_distribution = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id='distribution-month-graph', figure=tsa.fig_month),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.H4("Интерпретация графика"),
                                html.Br(),
                                dcc.Markdown(),
                            ],
                            style={'max-width': '35%'}
                        ),
                    ]
                ),
            ]
        ),
    ]
)
autocorrelation = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id='autocorr-acf-graph', figure=tsa.fig_acf),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.H4("Интерпретация графика"),
                                html.Br(),
                                dcc.Markdown(),
                            ],
                            style={'max-width': '35%'}
                        ),
                    ]
                ),
            ]
        ),
    ]
)
conclusion_analisys = html.Div(
    [
        html.Br(),
        dcc.Markdown(children=markdown.conclusion_analis_text)
    ]
)

tab3_content = dbc.Card(
    dbc.CardBody(
        [
            html.H3("Анализ данных временного ряда (EDA)"),
            dbc.Tabs(
                [
                dbc.Tab(total_trand, tab_id='total_trand', label='Общая тенденция'),
                dbc.Tab(time_distribution, tab_id='time_distribution', label='Распределение по времени'),
                dbc.Tab(day_distribution, tab_id='day_distribution', label='Распределение по дням недели'),
                dbc.Tab(month_distribution, tab_id='month_distribution', label='Распределение по месяцам'),
                dbc.Tab(autocorrelation, tab_id='autocorrelation', label='Автокорреляция'),
                dbc.Tab(conclusion_analisys, tab_id='conclusion_analisys', label='Вывод'),
                ],
                id='tabs_analisys',
                active_tab='total_trand'
            )
        ]
    ),
    className="mt-3",
)
# ------------------------------------------------------TRAINING,TESTING------------------------------------------------
dammi = html.Div(
    [
        html.Br(),
        html.H4("Результат дамми-кодирования категориальных признаков"),
        html.Br(),
        dash_table.DataTable(data=tsa.df_dammies.to_dict('records'),
                        columns=[{'id':c, 'name':c} for c in tsa.df_dammies.columns],
                        page_size=10,
                        style_header={
                            'backgroundColor': 'rgd(30,30,30)',
                            'color': 'black'
                        },
                        style_data={
                            'backgroundColor': 'rgb(50,50,50)',
                            'color': 'white'
                        },
                        sort_action='native',
                        style_table={'overflowX': 'scroll'},
                        tooltip_header={
                            'trend': 'Тренд количества заказов такси',
                            'seasonal': 'Сезонность количества заказов такси',
                            'month': 'Месяц',
                            'hour': 'Час',
                            'dayofweek': 'День недели',
                            'rolling_mean_D': 'Дневная скользящая средняя',
                            'rolling_mean_W': 'Недельная скользящая средняя'
                        }
        ),
    ]
)
train = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Br(),
                                html.H5('Выбрать модель:'),
                                dbc.Nav(
                                    [
                                        dbc.NavItem(dbc.NavLink("RidgeCV", id="btn_train_ridge", n_clicks=0, className='page-link')),
                                        dbc.NavItem(dbc.NavLink("LassoCV", id="btn_train_lasso", n_clicks=0, className='page-link')),
                                        dbc.NavItem(dbc.NavLink("ElasticNetCV", id="btn_train_elastic", n_clicks=0, className='page-link')),
                                        dbc.NavItem(dbc.NavLink("RandomForestRegressor", id="btn_train_rf", n_clicks=0, className='page-link')),
                                        dbc.NavItem(dbc.NavLink("LGBMRegressor", id="btn_train_lgbm", n_clicks=0, className='page-link')),
                                    ],
                                    id='nav_train',
                                    vertical="md",
                                )
                            ],
                            style={'max-width': '20%'}
                        ),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.Div(id='header_learning_curve_div')
                                            ],
                                            id='cardheader_train'
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Div(id='learning_curve_div')
                                            ],
                                            id='cardbody_train'
                                        ),
                                    ],
                                )
                            ],
                            style={'max-width': '80%'}
                        ),
                    ]
                ),
            ]
        ),
    ]
)
params = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Br(),
                                html.H5('Выбрать модель:'),
                                dbc.Nav(
                                    [
                                        dbc.NavItem(dbc.NavLink("RidgeCV", id="btn_params_ridge", n_clicks=0, className="page-link")),
                                        dbc.NavItem(dbc.NavLink("LassoCV", id="btn_params_lasso", n_clicks=0, className="page-link")),
                                        dbc.NavItem(dbc.NavLink("ElasticNetCV", id="btn_params_elastic", n_clicks=0, className="page-link")),
                                        dbc.NavItem(dbc.NavLink("RandomForestRegressor",id="btn_params_rf", n_clicks=0, className="page-link")),
                                        dbc.NavItem(dbc.NavLink("LGBMRegressor", id="btn_params_lgbm", n_clicks=0, className="page-link")),
                                    ],
                                    vertical="md",
                                    id='nav_params'
                                )
                            ],
                            style={'max-width': '20%'}
                        ),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                html.Div(id='header_params_div')
                                            ],
                                            id='cardheader_params'
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Div(id='params_div')
                                            ],
                                            id='cardbody_params'
                                        ),
                                    ]
                                )
                            ],
                            style={'max-width': '80%'}
                        ),
                    ]
                ),
            ]
        ),
    ]
)
models = html.Div(
    [
        html.Br(),
        html.H4("Сравнительная таблица обучения моделей"),
        html.Br(),
        dbc.Table.from_dataframe(tsa.comparison_table, striped=True, bordered=True, hover=True)
    ]
)
conclusion_train = html.Div(
    [
        html.Br(),
        dcc.Markdown(),
    ]
)

tab4_content = dbc.Card(
    dbc.CardBody(
        [
            html.H3("Моделирование, обучение и тестирование моделей"),
            dbc.Tabs(
                [
                    dbc.Tab(dammi, tab_id='dammi', label='Дамми-кодирование'),
                    dbc.Tab(train, tab_id='train', label='Обучение'),
                    dbc.Tab(params, tab_id='params', label='Параметры моделей'),
                    dbc.Tab(models, tab_id='models', label='Сравнение моделей'),
                    dbc.Tab(conclusion_train, tab_id='conclusion_train', label='Вывод'),
                ],
                id='tabs_train',
                active_tab='dammi'
            )
        ]
    ),
    className="mt-3",
)

#-----------------------------------------------------------РАЗМЕТКА----------------------------------------------------
app.layout = html.Div(
    [
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(logo, style={'max-width': '30%'}, align='start'),
                    dbc.Col(
                        [
                            header_1,
                            header_2
                        ],
                    ),
                ],
                align='center',
                className="p-5",
                style={'max-height': '128px'}
            ),
            style={'max-width': '100%'},
        ),
        dbc.Container(
            html.Div(
                [
                    dbc.Tabs(
                        [
                            dbc.Tab(tab1_content, label='Задача', tab_id='tab_1', tab_style={"marginLeft": "auto"}),
                            dbc.Tab(tab2_content, label='Подготовка данных', tab_id='tab_2'),
                            dbc.Tab(tab3_content, label='Анализ', tab_id='tab_5'),
                            dbc.Tab(tab4_content, label='Обучение и Тестирование', tab_id='tab_6'),
                        ],
                        id='tabs',
                        active_tab="tab_1",
                    ),
                ], style={'max-width': '1300px'},
            ), style={'max-width': '100%'}, className="p-5",
        ),
    ],
)
#------------------------------------------------------------ФУНКЦИИ----------------------------------------------------
@app.callback(
    Output(component_id='mono_div', component_property='children'),
    Input(component_id='mono_btn', component_property='n_clicks')
)
def on_mono_btn_clicks(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    elif tsa.df.index.is_monotonic_increasing:
        return "Временной ряд монотонный"
    else:
        return "Временной ряд немонотонный"

@app.callback(
    Output(component_id='adf_div', component_property='children'),
    Input(component_id='adf_btn', component_property='n_clicks')
)
def on_adf_btn_clicks(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    else:
        return dbc.Table.from_dataframe(tsa.adf_test, striped=True, bordered=True, hover=True)

@app.callback(
    Output(component_id='kpss_div', component_property='children'),
    Input(component_id='kpss_btn', component_property='n_clicks')
)
def on_kpss_btn_clicks(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    else:
        return dbc.Table.from_dataframe(tsa.kpss_test, striped=True, bordered=True, hover=True)

@app.callback(
    Output(component_id='stationarity_div', component_property='children'),
    [Input(component_id='adf_btn', component_property='n_clicks'),
     Input(component_id='kpss_btn', component_property='n_clicks')]
)
def double_btn_clicks(n_clicks_1, n_clicks_2):
    if n_clicks_1 == n_clicks_2 == 1:
        if tsa.adf_test_series.iloc[1] > 0.05 and tsa.kpss_test_series.iloc[1] < 0.05:
            return dcc.Markdown(children=markdown.result_test_1, className='text-primary')
        elif tsa.adf_test_series.iloc[1] < 0.05 and tsa.kpss_test_series.iloc[1] > 0.05:
            return dcc.Markdown(children=markdown.result_test_2, className='text-primary')
        elif tsa.adf_test_series.iloc[1] < 0.05 and tsa.kpss_test_series.iloc[1] < 0.05:
            return dcc.Markdown(children=markdown.result_test_3, className='text-primary')
        else:
            return dcc.Markdown(children=markdown.result_test_4, className='text-primary')
    else:
        raise PreventUpdate

@app.callback(
    [Output("header_learning_curve_div", "children"),
    Output("learning_curve_div", "children")],
    [Input("btn_train_ridge", "n_clicks"),
     Input("btn_train_lasso", "n_clicks"),
     Input("btn_train_elastic", "n_clicks"),
     Input("btn_train_rf", "n_clicks"),
     Input("btn_train_lgbm", "n_clicks")]
)
def show_clicks_train(n1, n2, n3, n4, n5):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ["", ""]
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "btn_train_ridge":
        return [html.H5('Регуляризованная линейная модель гребневой регрессии (L2 регуляризация)'),
                dcc.Graph(id='learning_curve_ridge', figure=tsa.train_graphs[0])]
    elif button_id == "btn_train_lasso":
        return [html.H5('Регуляризованная линейная модель лассо регрессии (L1 регуляризация)'),
                dcc.Graph(id='learning_curve_lasso', figure=tsa.train_graphs[1])]
    elif button_id == "btn_train_elastic":
        return [html.H5('Регуляризованная линейная модель регрессии эластичная сетка (L1, L2 регуляризации)'),
                dcc.Graph(id='learning_curve_elastic', figure=tsa.train_graphs[2])]
    elif button_id == "btn_train_rf":
        return [html.H5('Ансамбль `Random Forest` (мажоритарное голосование)'),
                dcc.Graph(id='learning_curve_rf', figure=tsa.train_graphs[3])]
    else:
        return [html.H5('Ансамбль `Light GBM` (градиентный бустинг)'),
                dcc.Graph(id='learning_curve_lgbm', figure=tsa.train_graphs[4])]

@app.callback(
    [Output("header_params_div", "children"),
     Output("params_div", "children")],
    [Input("btn_params_ridge", "n_clicks"),
     Input("btn_params_lasso", "n_clicks"),
     Input("btn_params_elastic", "n_clicks"),
     Input("btn_params_rf", "n_clicks"),
     Input("btn_params_lgbm", "n_clicks")]
)
def show_clicks_params(n1, n2, n3, n4, n5):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ["", ""]
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "btn_params_ridge":
        return [html.H5('Регуляризованная линейная модель гребневой регрессии (L2 регуляризация)'),\
               dbc.Table.from_dataframe(tsa.params[0], striped=True, bordered=True, hover=True)]
    elif button_id == "btn_params_lasso":
        return [html.H5('Регуляризованная линейная модель лассо регрессии (L1 регуляризация)'),
                dbc.Table.from_dataframe(tsa.params[1], striped=True, bordered=True, hover=True)]
    elif button_id == "btn_params_elastic":
        return [html.H5('Регуляризованная линейная модель регрессии эластичная сетка (L1, L2 регуляризации)'),
                dbc.Table.from_dataframe(tsa.params[2], striped=True, bordered=True, hover=True)]
    elif button_id == "btn_params_rf":
        return [html.H5('Ансамбль `Random Forest` (мажоритарное голосование)'),
                dbc.Table.from_dataframe(tsa.params[3], striped=True, bordered=True, hover=True)]
    else:
        return [html.H5('Ансамбль `Light GBM` (градиентный бустинг)'),
                dbc.Table.from_dataframe(tsa.params[4], striped=True, bordered=True, hover=True)]


if __name__ == '__main__':
    app.run_server(debug=True)