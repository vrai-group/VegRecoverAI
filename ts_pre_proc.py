import matplotlib.pyplot as plt
import pandas as pd
from pygam import LinearGAM, LogisticGAM, PoissonGAM, GammaGAM, InvGaussGAM, GAM, s, f, l
import statsmodels.api as sm

import numpy as np


lowess = sm.nonparametric.lowess

"""
This script contains the main pre-processing and smoothing functions of the time series.
"""


def remove_outliers(y, frac, alpha=0.2):
    """
    Outlier removal algorithm based on tsclean of forecast package for R
    :param y: time series in pandas.Series format
    :param frac: fraction of data to be used in the smoothin lowess
    :param alpha: factor for defining lower and upper bounds for thresholding
    :return: processed time series in pandas.Dataframe format.
    """
    y_smoothed = lowess(y, y.index, xvals=y.index, frac=frac, is_sorted=True, return_sorted=False)
    resid = pd.Series(y - y_smoothed)
    q25 = resid.quantile(q=0.25)
    q75 = resid.quantile(q=0.75)
    iqr = q75-q25
    lower_lim = q25 - (iqr * alpha)
    upper_lim = q75 + (iqr * alpha)
    outlier_removed = pd.DataFrame(y)
    outlier_removed.dropna(inplace=True)
    for index, row in outlier_removed.iterrows():
        if resid[index] < lower_lim or resid[index] > upper_lim or row[0]<0:
            outlier_removed.drop(index, inplace=True)
    return outlier_removed


def smoothing(series, params: dict):
    """
    Applies outlier removal and smoothing
    :param series: time series in pandas.Series format
    :param params: parameter dictionary (see main.py)
    :return: (processed time series, method info)
    """
    if len(series.dropna().index) == 0:
        return pd.Series(np.zeros((len(series.index)))), "NODATA"

    frac = params["frac"] # 0.04
    frac_outliers = params["frac_outliers"]
    n_splines = params["n_splines"]  # 72
    lam = params["lam"]  # 0.1
    alpha = params["alpha"]  # 0.2
    lams = np.logspace(-3, 5, 5)
    method = params["method"]
    plot = params["plot"]

    if params["remove_outliers"] is True:
        copy = remove_outliers(series, frac_outliers, alpha)
        if plot:
            plt.scatter(copy.index, copy, color="lightblue")
    else:
        copy = series.copy()

    y = None
    x = series.index
    if method == "lowess-gam":
        y = lowess(copy.values.ravel(), x, xvals=x, frac=frac, is_sorted=True, return_sorted=False)
        temp = pd.Series(y).dropna()
        gam = LinearGAM(s(0, n_splines=n_splines, basis="ps", lam=lam)).fit(temp.index[:, None], temp)
        # gam = LinearGAM(s(0, n_splines=n_splines, basis="ps")).gridsearch(temp.index[:, None], temp, lam=lams)
        y = gam.predict(x)
        method_title = "LOWESS " + str(frac) + " -> LinearGAM n_splines: " + str(n_splines) + ", lam: "+str(lam)

    elif method == "gam":
        temp = copy.dropna()
        gam = LinearGAM(s(0, n_splines=n_splines, basis="ps", lam=lam)).fit(temp.index[:, None], temp)
        print(gam.summary())
        y = gam.predict(x)
        method_title = "LinearGAM n_splines: " + str(n_splines) + ", lam: "+str(lam)

    elif method == "lowess":
        y = lowess(copy.values.ravel(), copy.index, xvals=x, frac=frac, is_sorted=True, return_sorted=False)
        method_title = "LOWESS frac: " + str(frac)

    return pd.Series(y), method_title


def smoothing_multi_ts(series_list, params: dict):
    """
    Applies smoothing to the time series in the provided list (obtained from multi_time_series or full_image_time_series)
    :param series_list: list of time series in pandas.Series format
    :param params: parameter dictionary (see in main.py)
    :return: (list of processed time series, method info)
    """
    smoothed = []
    for series in series_list:
        y, method_t = smoothing(series, params)
        smoothed.append(y)
    return smoothed, method_t


def single_time_series(data, slots, x, y, channel):
    """
    Extracts the time series of a single pixel from the provided list of images
    :param date: list of images obtained by the API
    :param slots: list of dates used in API request phase
    :param x: x coordinate of the pixel
    :param y: y coordinate of the pixel
    :param channel: channel from which to extract values for the time series
    :return: time series in pandas.Series format
    """
    series = []
    x_dates = []
    for idx, image in enumerate(data):
        if len(image.shape) == 2:
            series.append(image[y, x])
        else:
            series.append(image[y, x, channel])
        x_dates.append(slots[idx][0])
    return x_dates, pd.Series(series)


def multi_time_series(data, slots, coord_file, channel):
    """
    Extracts and returns the time series of the pixels specified in the provided coordinate text file.
    It also returns the unique list of dates (first elements of tuples in slots).
    :param date: list of images obtained from the API.
    :param slots: list of dates used in API request phase.
    :param coord_file: coordinate file (created with pixel selector)
    :param channel: channel from which to extract values for the time series
    :return: (list of dates, list of time series) each time series is in pandas.Series format
    """
    series_list = []
    x_dates = []
    for idx, image in enumerate(data):
        x_dates.append(slots[idx][0])
    with open(coord_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            series = []
            vals = line.split(",")
            for idx, image in enumerate(data):
                if len(image.shape) == 2:
                    series.append(image[int(vals[1]), int(vals[0])])
                else:
                    series.append(image[int(vals[1]), int(vals[0]), channel])
            series = pd.Series(series)
            series_list.append(series)
    return x_dates, series_list


def full_image_time_series(data, slots, channel):
    """
    Extracts the time series of all pixels in the study area.
    It also returns the unique list of dates (first elements of tuples in slots), height and width of images
    :param date: list of images obtained from the API
    :param slots: list of dates used in API request phase
    :param channel: channel from which to extract values for the time series
    :return: (list of dates, list of time series, height, width)
    """
    image_series = []
    x_dates = []
    height, width = data[0].shape[0], data[0].shape[1]
    print(height, width)
    for idx, image in enumerate(data):
        x_dates.append(slots[idx][0])

    for y in range(height):
        for x in range(width):
            series = []
            for idx, image in enumerate(data):
                if len(image.shape)==2:
                    series.append(image[y, x])
                else:
                    series.append(image[y, x, channel])
            series = pd.Series(series)
            image_series.append(series)
    return x_dates, image_series, height, width


# Test functions


def multi_remove_outliers(series_list, x_dates, params: dict):
    df = pd.DataFrame()
    df["date"] = x_dates
    cols = [df]
    col_names = ["date"]
    for i, series in enumerate(series_list):
        outlier_removed = remove_outliers(series, params["frac"], params["alpha"])
        cols.append(outlier_removed)
        col_names.append(str(i))
    res = pd.concat(cols, axis=1)
    res.columns = col_names
    return res


def yearly_lowess(dataframe, frac=0.3):
    i = 0
    dates = pd.Series(np.unique(dataframe["date"].dt.year))
    cols = [dates]
    dataframe.loc[(dataframe["date"].dt.month < 5) | (dataframe["date"].dt.month >= 9)] = np.NaN
    while str(i) in dataframe.columns:
        series = dataframe[str(i)]
        result = group_by_year(series, dataframe["date"])
        y = lowess(result[2].values, result.index, xvals=np.unique(result.index), frac=0.3, is_sorted=True,
                   return_sorted=False)
        cols.append(pd.Series(y))
        i += 1
    df = pd.concat(cols, axis=1)
    df.columns = dataframe.columns
    return df


def group_by_year(series, dates):
    unique_years = np.unique(dates.dt.year)
    years = dates.dt.year
    result = []
    for i,y in enumerate(unique_years):
        vals = series[(years>=y) & (years<y+1)].values
        for v in vals:
            if not np.isnan(v):
                result.append((i, y, v))
    result = pd.DataFrame(result)
    result = result.set_index(0)
    return result