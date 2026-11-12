import numpy as np
import datetime

import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)

from modules.mp import *


def heads_tails(consumptions: dict, cutoff, house_idx: list) -> tuple[dict, dict]:
    """
    Split time series into two parts: Head and Tail

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses

    Returns
    --------
    heads: heads of time series
    tails: tails of time series
    """

    heads, tails = {}, {}
    for i in house_idx:
        heads[f'H_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index < cutoff]
        tails[f'T_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index >= cutoff]
    
    return heads, tails


def meter_swapping_detection(heads: dict, tails: dict, house_idx: dict, m: int) -> dict:
    """
    Find the swapped time series pair

    Parameters
    ---------
    heads: heads of time series
    tails: tails of time series
    house_idx: indices of houses
    m: subsequence length

    Returns
    --------
    min_score: time series pair with minimum swap-score
    """

    eps = 0.001

    min_score = {}

    # INSERT YOUR CODE
    
    min_score = {
        'i': None,
        'j': None,
        'score': np.inf,
        'mp_j': None
    }

    # for i in house_idx:
    #     head_i = heads[f'H_{i}'].squeeze().to_numpy().astype(np.float64)
    #     tail_i = tails[f'T_{i}'].squeeze().to_numpy().astype(np.float64)

    #     # baseline: насколько голова i похожа на СВОЙ хвост
    #     mp_self = compute_mp(head_i, m, ts2=tail_i)
    #     min_self = np.min(mp_self['mp'][np.isfinite(mp_self['mp'].astype(np.float64))])

    #     for j in house_idx:
    #         if j == i:
    #             continue

    #         tail_j = tails[f'T_{j}'].squeeze().to_numpy().astype(np.float64)
    #         mp_join = compute_mp(head_i, m, ts2=tail_j)
    #         min_join = np.min(mp_join['mp'][np.isfinite(mp_join['mp'].astype(np.float64))])

    #         swap_score = min_join / (min_self + eps)

    #         if swap_score < min_score['score']:
    #             min_score['i'] = i
    #             min_score['j'] = j
    #             min_score['score'] = swap_score
    #             min_score['mp_j'] = mp_join

    for i in house_idx:
        head_i = heads[f'H_{i}'].squeeze().to_numpy().astype(np.float64)
        tail_i = tails[f'T_{i}'].squeeze().to_numpy().astype(np.float64)
        mp_self_i = compute_mp(head_i, m, ts2=tail_i)
        min_self_i = np.min(mp_self_i['mp'][np.isfinite(mp_self_i['mp'].astype(np.float64))])

    for j in house_idx:
        if j == i:
            continue

        tail_j = tails[f'T_{j}'].squeeze().to_numpy().astype(np.float64)
        mp_join = compute_mp(head_i, m, ts2=tail_j)
        min_join = np.min(mp_join['mp'][np.isfinite(mp_join['mp'].astype(np.float64))])

        # Симметричный score: учитываем и H_i vs T_j, и H_j vs T_i
        head_j = heads[f'H_{j}'].squeeze().to_numpy().astype(np.float64)
        mp_self_j = compute_mp(head_j, m, ts2=tail_j)
        min_self_j = np.min(mp_self_j['mp'][np.isfinite(mp_self_j['mp'].astype(np.float64))])

        mp_join_rev = compute_mp(head_j, m, ts2=tail_i)
        min_join_rev = np.min(mp_join_rev['mp'][np.isfinite(mp_join_rev['mp'].astype(np.float64))])

        score_forward = min_join / (min_self_i + eps)
        score_backward = min_join_rev / (min_self_j + eps)
        swap_score = min(score_forward, score_backward)

        if swap_score < min_score['score']:
            min_score['i'] = i
            min_score['j'] = j
            min_score['score'] = swap_score
            min_score['mp_j'] = mp_join

    return min_score



def plot_consumptions_ts(consumptions: dict, cutoff, house_idx: list):
    """
    Plot a set of input time series and cutoff vertical line

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses
    """

    num_ts = len(consumptions)

    fig = make_subplots(rows=num_ts, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i in range(num_ts):
        fig.add_trace(go.Scatter(x=list(consumptions.values())[i].index, y=list(consumptions.values())[i].iloc[:,0], name=f"House {house_idx[i]}"), row=i+1, col=1)
        fig.add_vline(x=cutoff, line_width=3, line_dash="dash", line_color="red",  row=i+1, col=1)

    fig.update_annotations(font=dict(size=22, color='black'))
    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title='Houses Consumptions',
                      title_x=0.5,
                      title_font=dict(size=26, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)', 
                      height=800,
                      legend=dict(font=dict(size=20, color='black'))
                      )

    fig.show(renderer="colab")
