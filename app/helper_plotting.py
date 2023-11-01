# os libs
import os
from io import BytesIO
import base64

# graph libs
import plotly.graph_objs as go
import matplotlib  # pip install matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import plotly.express as px

# computation libs
import numpy as np


def fig_to_uri(in_fig, close_all=True, **save_args):
    """
    # type: (plt.Figure) -> str

    Save a figure as a URI
    :param close_all:
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


def plot_confusion_matrix_matplotlib(conf_mat,
                                     hide_spines=False,
                                     hide_ticks=False,
                                     figsize=None,
                                     cmap=None,
                                     colorbar=False,
                                     show_absolute=True,
                                     show_normed=False,
                                     class_names=None):
    if not (show_absolute or show_normed):
        raise AssertionError('Both show_absolute and show_normed are False')
    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError('len(class_names) should be equal to number of'
                             'classes in the dataset')

    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float') / total_samples

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues

    if figsize is None:
        figsize = (len(conf_mat) * 1.25, len(conf_mat) * 1.25)

    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap)

    if colorbar:
        fig.colorbar(matshow)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                cell_text += format(conf_mat[i, j], 'd')
                if show_normed:
                    cell_text += "\n" + '('
                    cell_text += format(normed_conf_mat[i, j], '.2f') + ')'
            else:
                cell_text += format(normed_conf_mat[i, j], '.2f')
            ax.text(x=j,
                    y=i,
                    s=cell_text,
                    va='center',
                    ha='center',
                    color="white" if normed_conf_mat[i, j] > 0.5 else "black")

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)

    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    plt.xlabel('predicted label')
    plt.ylabel('true label')

    out_url = fig_to_uri(fig, dpi=80)
    return out_url


def plot_accuracy_matplotlib(train_acc_list,
                             valid_acc_list,
                             results_dir):
    num_epochs = len(train_acc_list)

    plt.plot(np.arange(1, num_epochs + 1),
             train_acc_list, label='Training')
    plt.plot(np.arange(1, num_epochs + 1),
             valid_acc_list, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # plt.tight_layout()

    # input output stream
    # buf = io.BytesIO()
    # plt.savefig(buf, format="png", dpi=80)
    # acc_fig = base64.b64encode(buf.getbuffer()).decode("ascii")
    # acc_fig_matplotlib = f'data:image/png;base64,{acc_fig}'

    if results_dir is not None:
        image_path = os.path.join(
            results_dir, 'plot_acc_training_validation.pdf')
        plt.savefig(image_path)

    # return acc_fig_matplotlib


def plot_accuracy_go(train_acc_list,
                     valid_acc_list,
                     results_dir):
    num_epochs = len(train_acc_list)

    fig = go.Figure()

    # Ajouter la trace pour la précision de l'entraînement
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, num_epochs + 1),
            y=train_acc_list,
            mode='lines',
            name='Training'
        )
    )

    # Ajouter la trace pour la précision de la validation
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, num_epochs + 1),
            y=valid_acc_list,
            mode='lines',
            name='Validation'
        )
    )

    # Mettre à jour les étiquettes et le titre
    fig.update_layout(
        # title='Accuracy',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        legend=dict(orientation='h',
                    yanchor='top',
                    y=1.1,
                    xanchor='center',
                    x=0.5),  # Placer la légende en haut et au centre
        margin=dict(l=0, r=0, b=0, t=0)  # Ajuster les marges
    )

    if results_dir is not None:
        image_path = os.path.join(results_dir, 'plot_acc_training_validation.pdf')
        fig.write_image(image_path)

    return fig


def plot_training_loss_matplotlib(minibatch_loss_list,
                                  num_epochs,
                                  iter_per_epoch,
                                  results_dir=None,
                                  averaging_iterations=100):
    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_loss_list)),
             minibatch_loss_list, label='Minibatch Loss')

    if len(minibatch_loss_list) > 1000:
        ax1.set_ylim([
            0, np.max(minibatch_loss_list[1000:]) * 1.5
        ])
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    ax1.plot(np.convolve(minibatch_loss_list,
                         np.ones(averaging_iterations, ) / averaging_iterations,
                         mode='valid'),
             label='Running Average')
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()

    # input output stream
    # buf = io.BytesIO()
    # plt.savefig(buf, format="png", dpi=80)
    # loss_fig = base64.b64encode(buf.getbuffer()).decode("ascii")
    # loss_fig_matplotlib = f'data:image/png;base64,{loss_fig}'

    if results_dir is not None:
        image_path = os.path.join(results_dir, 'plot_training_loss.pdf')
        plt.savefig(image_path)

    # return loss_fig_matplotlib


def plot_training_loss_go(minibatch_loss_list,
                          num_epochs,
                          iter_per_epoch,
                          results_dir=None,
                          averaging_iterations=100):
    fig = go.Figure()

    # Ajouter la trace pour la perte du minibatch
    fig.add_trace(
        go.Scatter(
            x=list(range(len(minibatch_loss_list))),
            y=minibatch_loss_list,
            mode='lines',
            name='Minibatch Loss'
        )
    )

    if len(minibatch_loss_list) > 1000:
        max_loss = np.max(minibatch_loss_list[1000:]) * 1.5
        fig.update_yaxes(range=[0, max_loss])

    # Ajouter la trace pour la moyenne mobile (Running Average)
    running_avg = np.convolve(minibatch_loss_list, np.ones(averaging_iterations, ) / averaging_iterations, mode='valid')
    fig.add_trace(
        go.Scatter(
            x=list(range(averaging_iterations - 1, len(minibatch_loss_list))),
            y=running_avg,
            mode='lines',
            name='Running Average'
        )
    )

    # Ajouter les marqueurs pour les époques
    epochs = list(range(num_epochs + 1))
    epoch_positions = [e * iter_per_epoch for e in epochs]

    fig.add_trace(
        go.Scatter(
            x=epoch_positions,
            y=[0] * len(epoch_positions),
            mode='markers',
            name='Epochs',
            marker=dict(symbol='line-ns-open', size=8, color='black'))
    )

    # Mettre à jour les étiquettes et le titre
    fig.update_layout(
        # title='Training Loss',
        xaxis_title='Iterations',
        yaxis_title='Loss',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.1,
            xanchor='center',
            x=0.5
        ),
        # Placer la légende en haut et au centre
        margin=dict(l=0, r=0, b=0, t=0),  # Ajuster les marges
    )

    if results_dir is not None:
        image_path = os.path.join(results_dir, 'plot_training_loss.pdf')
        fig.write_image(image_path)

    return fig


def plot_training_loss_px(minibatch_loss_list,
                          num_epochs,
                          iter_per_epoch,
                          results_dir=None,
                          averaging_iterations=100):
    fig = px.line(
        x=list(range(len(minibatch_loss_list))),
        y=minibatch_loss_list,
        labels={'x': 'Iterations', 'y': 'Loss'},
        title='Training Loss'
    )

    if len(minibatch_loss_list) > 1000:
        max_loss = np.max(minibatch_loss_list[1000:]) * 1.5
        fig.update_yaxes(range=[0, max_loss])

    # Running Average (en avant-plan)
    running_avg = np.convolve(minibatch_loss_list, np.ones(averaging_iterations, ) / averaging_iterations, mode='valid')
    fig.add_scatter(
        x=list(range(averaging_iterations - 1, len(minibatch_loss_list))),
        y=running_avg,
        mode='lines',
        name='Running Average',
        line=dict(color='red', width=2)
    )

    # Ajouter une légende pour "minibatch_loss"
    fig.add_scatter(
        x=list(range(len(minibatch_loss_list))),
        y=minibatch_loss_list,
        mode='lines',
        name='Minibatch Loss'
    )

    # Set second x-axis
    epochs = list(range(num_epochs + 1))
    epoch_positions = [e * iter_per_epoch for e in epochs]
    fig.update_layout(
        xaxis2=dict(
            ticks='outside',
            tickvals=epoch_positions[::10],
            ticktext=epochs[::10],
            anchor='y',
            overlaying='x',
            side='top'
        ),
        xaxis=dict(domain=[0, 1])
    )

    # Placer toutes les légendes en haut du graphique
    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.1,
            xanchor='right', x=1
        )
    )

    # Mettre en avant-plan la courbe de "Running Average"
    fig.update_traces(selector=dict(name='Running Average'),
                      line=dict(width=2))

    if results_dir is not None:
        image_path = os.path.join(results_dir, 'plot_training_loss.pdf')
        fig.write_image(image_path)

    return fig
