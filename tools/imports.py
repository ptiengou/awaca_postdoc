import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from cycler import cycler
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ttest_ind
import scipy.stats as stats
from scipy.stats import linregress
from scipy.interpolate import griddata
from scipy.integrate import trapezoid
from matplotlib.markers import MarkerStyle
import json
from pprint import pprint
import matplotlib.gridspec as gridspec
from io import StringIO
import os
import matplotlib.ticker as mticker
import geopandas as gpd
import glob
from pathlib import Path
from typing import * #Optional, Dict, List, Tuple, Union, Any, Sequence, Literal, Callable
from datetime import datetime, timedelta
import seaborn as sns
import scipy.ndimage
from collections import defaultdict


#import psyplot.project as psy

## PARAM matplotlib viz
plt.rcParams.update(
        {
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'font.size': 12,
            'figure.dpi': 72.0,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 5.0,
            'xtick.minor.size': 2.5,
            'ytick.major.size': 5.0,
            'ytick.minor.size': 2.5,
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
            'axes.grid': True,
            'axes.titlesize': 'larger',
            'axes.labelsize': 'larger',
            'grid.color': 'dimgray',
            'grid.linestyle': '-',
            'grid.alpha': 0.3,
            'axes.prop_cycle': cycler(
                color=[
                    '#0C5DA5', #blue
                    '#FF2C00', #red
                    '#00B945', #green
                    '#845B97', #purple
                    '#474747', #grey
                    '#9E9E9E', #light grey
                    '#FF9500', #orange/yellow
                ]
            ) * cycler(alpha=[0.8]),
            'scatter.marker': 'x',
            'lines.linewidth': 1.5,
        })
