import matplotlib.pyplot as plt
import pandas as pd

def make_map_of_r2_and_risk(map_df, columns, colors, titles, labels, boundary_map=shpfile):
    fig, ax = plt.subplots(1, len(columns), figsize=(20, 8), constrained_layout=True)
    base_kwargs = {'edgecolor': 'black','linewidth': 0.2,'missing_kwds': {"color": "lightgrey", "label": "No Data"}}

    boundary_kwargs = {'facecolor': 'lightgrey','alpha'    : 0.3,'edgecolor': 'darkgrey',  'linewidth': 0.8}

    for i, col in enumerate(columns):
        map_df.dropna().plot(column=col,ax=ax[i],legend=True,cmap=colors[i],legend_kwds={'label': labels[i],
                             'orientation': 'horizontal','fraction': 0.046,'pad': 0.04},
                             **base_kwargs)
        boundary_map.plot(ax=ax[i], **boundary_kwargs)
        ax[i].set_title(titles[i], fontsize=16)
        ax[i].set_axis_off()

    plt.show()
