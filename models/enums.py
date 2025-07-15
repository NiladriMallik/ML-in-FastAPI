from enum import Enum

class PlotType(str, Enum):
    hist = "hist"
    box = "box"
    scatter = "scatter"
    pairplot = "pairplot"
    heatmap = "heatmap"
    violin = "violin"
    barplot = "barplot"
    lineplot = "lineplot"
    kde = "kde"
