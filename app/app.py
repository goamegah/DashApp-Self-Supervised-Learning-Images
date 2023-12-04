import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import io
import requests
from typing import Optional

from helper_plotting import (plot_accuracy_go,
                             plot_training_loss_go,
                             plot_confusion_matrix_matplotlib)

import numpy as np
import pickle

import base64
from PIL import Image
from io import BytesIO

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Semi Supervised Image classification"

server = app.server
app.config.suppress_callback_exceptions = True

model_list = [
    "LeNet5",
    "ResNet18",
    "SimCLR"
]


#############################################
# Imported required variable for plotting
# Eg. Model config,
#############################################


# #############################################


def post_data_to_api(image_encoded: Optional = None,
                     model_name: str = 'LeNet5'):

    url = "http://127.0.0.1:8000/prediction"  # Mettez votre URL FastAPI ici
    files = {'file': image_encoded}
    data = {'model_name': model_name}

    if image_encoded is not None:
        prediction_response_from_api = requests.post(url, files=files, data=data)
    else:
        url = "http://127.0.0.1:8000/evaluation"
        prediction_response_from_api = requests.post(url, data=data)

    if prediction_response_from_api.status_code == 200:
        return prediction_response_from_api.json()
    else:
        return {'message': f'Erreur lors de la requête POST. Code d\'état : {prediction_response_from_api.status_code}"'}


def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Semi supervised Image classification "),
            html.H3("Welcome to Ssima model  "),
            html.Div(
                id="intro",
                children="Explore the model by uploading image and get the prediction. Model behind the scene is the"
                         "deep convolutional Neural Network train to recognize image Hand-writing images",
            ),
        ],
    )


def generate_control_card():
    """

    :return: A Div containing controls for graphs.

    content:
        - upload button for image
        - model selection & hyperparms
    """

    return html.Div(id="control-card",
                    children=[
                        html.P("Upload image file"),
                        dcc.Upload(
                            id="upload-image",
                            children=[
                                'Drag and Drop or ',
                                html.A('Select a File')
                            ],
                            style={
                                # "color": "darkgray",
                                "width": "100%",
                                "height": "50px",
                                "lineHeight": "50px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "borderColor": "darkgray",
                                "textAlign": "center",
                                "padding": "2rem 0",
                                "margin-bottom": "2rem"
                            }),
                        html.Br(),

                        html.P("Select Model"),
                        dcc.Dropdown(
                            id="model-dropdown",
                            options=[{"label": i, "value": i} for i in model_list],
                            value=model_list[0],
                        ),
                        html.Br(),
                        html.Br(),
                        html.Div(
                            id="predict-btn-outer",
                            children=html.Button(id="predict-button",
                                                 children="Predict",
                                                 n_clicks=0),
                        ),

                        # Image Prediction
                        html.Br(),

                        html.Div(
                            id="prediction",
                            children=[
                                html.B("IMAGE PREDICTION"),
                                html.Hr(),
                                html.Div(id='prediction-output',
                                         style={'whiteSpace': 'pre-line'})
                            ],
                        ),
                    ])


def generate_image_uploaded():
    pass


def generate_image_histogram():
    pass


# App Layout
app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("plotly_logo.png"))],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
                     + [
                         html.Div(
                             ["initial child"], id="output-clientside", style={"display": "none"}
                         )
                     ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[

                html.Div(
                    id="model-summary",
                    children=[
                        # Image uploaded placeholder
                        html.Div(
                            id="div-interactive-image",
                            children=[
                                html.B("Image uploaded"),
                                html.Hr(),
                                html.Div(id='interactive-image'),
                            ],
                        ),

                        html.Div(
                            id="div-learning-curve",
                            children=[
                                html.B("Learning Curve"),
                                html.Hr(),
                                dcc.Graph(id='learning_curve_graph'),
                            ],
                        ),

                        # html.Div(
                        #    id="div-learning-curve",
                        #    children=[
                        #        html.B("Learning Curve"),
                        #        html.Hr(),
                        #        html.Img(id='learning_curve_img'),
                        #    ],
                        # ),

                        html.Div(
                            id="div-accuracy-curve",
                            children=[
                                html.B("Accuracy curve"),
                                html.Hr(),
                                dcc.Graph(id='accuracy_curve'),
                            ],
                        ),

                        html.Div(
                            id="div-confusion-matrix-curve",
                            children=[
                                html.B("Confusion matrix"),
                                html.Hr(),
                                html.Img(id='confusion_matrix'),
                            ],
                        ),

                        # html.Div(
                        #    id="div-accuracy-curve",
                        #    children=[
                        #       html.B("Accuracy curve"),
                        #        html.Hr(),
                        #        html.Img(id='accuracy_curve_img'),
                        #    ],
                        # ),

                    ]
                ),

                html.Br(),
                html.Br(),
            ],
        ),
    ],
)


# Dans la fonction de rappel
@app.callback(
    Output(component_id='interactive-image', component_property='children'),
    Input(component_id='upload-image', component_property='contents')
)
def display_uploaded_image(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))

        # Redimensionner l'image à 28x28 pixels
        image = image.resize((32, 32))

        # Convertir l'image PIL en tableau numpy
        image_array = np.array(image)

        print(image_array.shape)
        print(image_array)

        return html.Img(src=contents, style={'width': '300px', 'height': '300px'})

    return dash.no_update  # Aucune mise à jour si aucun contenu n'est chargé


"""

#############################################
# DETERMINSITIC APPROCH
# 
# dash already know where find artefact file 
# and will plot all visualisation graph
#############################################


@app.callback(
    Output(component_id="learning_curve_graph", component_property='figure'),
    Output(component_id="accuracy_curve", component_property='figure'),
    Output(component_id="confusion_matrix", component_property='src'),
    Input(component_id="model-dropdown", component_property="value")
)
def update_graph_model_summary(selected_model):
    # Read dictionary pkl file corresponding to model name
    with open(f'saved_data/{selected_model}/{selected_model}_summary.pkl', 'rb') as fp:
        model_summary = pickle.load(fp)
        print(f'{selected_model} dictionary')
        print(model_summary)

    # get dict fields
    minibatch_loss_list = model_summary['minibatch_loss_list']
    train_acc_list = model_summary['train_acc_list']
    valid_acc_list = model_summary['valid_acc_list']
    confusion_matrix = model_summary['confusion_matrix']
    num_epochs = model_summary['num_epochs']
    iter_per_epoch = model_summary['iter_per_epoch']
    averaging_iterations = model_summary['averaging_iterations']

    plt_train_loss = plot_training_loss_go(
        minibatch_loss_list=minibatch_loss_list,
        num_epochs=num_epochs,
        iter_per_epoch=iter_per_epoch,
        results_dir=None,
        averaging_iterations=averaging_iterations
    )

    plt_train_valid_acc = plot_accuracy_go(
        train_acc_list=train_acc_list,
        valid_acc_list=valid_acc_list,
        results_dir=None
    )

    class_dict = {0: '0',
                  1: '1',
                  2: '2',
                  3: '3',
                  4: '4',
                  5: '5',
                  6: '6',
                  7: '7',
                  8: '8',
                  9: '9'}

    plt_conf_mat = plot_confusion_matrix_matplotlib(
        figsize=(6.6, 5.5),
        conf_mat=confusion_matrix,
        class_names=class_dict.values()
    )

    return plt_train_loss, plt_train_valid_acc, plt_conf_mat

"""

#############################################
# DYNAMIC VERSION
#
# dash don't know where find files. The
# files will come from API via POST request
#############################################


@app.callback(
    Output(component_id="learning_curve_graph", component_property='figure'),
    Output(component_id="accuracy_curve", component_property='figure'),
    Output(component_id="confusion_matrix", component_property='src'),
    Input(component_id="model-dropdown", component_property="value")
)
def update_graph_model_summary(selected_model):
    # Read dictionary pkl file corresponding to model name

    model_summary = post_data_to_api(image_encoded=None,
                                     model_name=selected_model)

    print(f'{selected_model} dictionary | type: {type(model_summary)}')
    print(hasattr(model_summary, '__dict__'))

    # get dict fields
    minibatch_loss_list = model_summary['minibatch_loss_list']
    train_acc_list = model_summary['train_acc_list']
    valid_acc_list = model_summary['valid_acc_list']
    # Here we need to transform mat conf onto numpy array
    confusion_matrix = np.array(model_summary['confusion_matrix'])
    num_epochs = model_summary['num_epochs']
    iter_per_epoch = model_summary['iter_per_epoch']
    averaging_iterations = model_summary['averaging_iterations']

    plt_train_loss = plot_training_loss_go(
        minibatch_loss_list=minibatch_loss_list,
        num_epochs=num_epochs,
        iter_per_epoch=iter_per_epoch,
        results_dir=None,
        averaging_iterations=averaging_iterations
    )

    plt_train_valid_acc = plot_accuracy_go(
        train_acc_list=train_acc_list,
        valid_acc_list=valid_acc_list,
        results_dir=None
    )

    class_dict = {0: '0',
                  1: '1',
                  2: '2',
                  3: '3',
                  4: '4',
                  5: '5',
                  6: '6',
                  7: '7',
                  8: '8',
                  9: '9'}

    plt_conf_mat = plot_confusion_matrix_matplotlib(
        figsize=(6.6, 5.5),
        conf_mat=confusion_matrix,
        class_names=class_dict.values()
    )

    return plt_train_loss, plt_train_valid_acc, plt_conf_mat


@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('upload-image', 'contents'),
     State('model-dropdown', 'value')]
)
def update_prediction(n_clicks, contents, selected_model):
    if n_clicks is not None and contents is not None:
        # encode image
        image = contents.encode('utf8').split(b';base64,')[1]
        # decode image
        image = base64.b64decode(image)
        # post data to API and get response as prediction
        prediction_response_from_api = post_data_to_api(image, selected_model)

        return f'This picture contain digit: {prediction_response_from_api.get("prediction")}'


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
