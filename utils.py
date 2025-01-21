import datetime
import pandas as pd
import os
import json
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from sentinelhub import (
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
    bbox_to_dimensions,
    Geometry
)

"""
This script contains several utility functions for extracting and displaying time series
"""

def intervals_by_chunks(start, end, n_chunks):
    """
    Creates n_chunks intervals between the start and end dates
    Example: start = 2017-01-01; end = 2017-01-03; n_chunks = 3
    [(“2017-01-01, ‘2017-01-02’), (”2017-01-02, “2017-01-03”)]
    :param start: start date
    :param end: end date
    :param n_chunks: number of intervals
    :return: list of length n_chunks-1 of tuples (ds, de)
    """
    start = datetime.datetime.strptime(start)
    end = datetime.datetime.strptime(end)
    tdelta = (end - start) / n_chunks
    edges = [(start + i * tdelta).date().isoformat() for i in range(n_chunks)]
    slots = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    print("Monthly time windows:\n")
    for slot in slots:
        print(slot)

    return slots


def fixed_weeks_per_year(start_year, n_years):
    """
    Creates a tuple list (ds, de) of weekly frequency dates for the specified number of years.
    The number of weeks and the specified dates are the same for each year.
    :param start_year: starting year.
    :param n_years: number of years
    :return: list of length n_years * 52 of tuples (ds, de)
    """
    slots = []
    for y in range(start_year, start_year+n_years):
        start = str(y)+"-01-01"
        end = str(y)+"-12-24"
        dates = pd.date_range(start=start, end=end, freq="7D")
        for d in dates:
            date2 = d + datetime.timedelta(days=7)
            date = d.strftime("%Y-%m-%d")
            date2 = date2.strftime("%Y-%m-%d")
            slots.append((date, date2))
    return slots


def dates_list(start, end, freq="D", interval_size=0):
    """
    Creates a tuple list (ds, de) of dates at variable frequency and interval length.
    :param start: start date
    :param end: end date
    :param freq: interval frequency (interval between the first element of one tuple and the next)
    :param interval_size: interval length (interval between elements of the same tuple)
    :return: list of tuples (ds, de)
    """
    slots = []
    dates = pd.date_range(start=start, end=end, freq=freq)
    for d in dates:
        date2 = d + datetime.timedelta(days=interval_size)
        date = d.strftime("%Y-%m-%d")
        date2 = date2.strftime("%Y-%m-%d")
        slots.append((date, date2))
    return slots


def print_file_names(folder_name):
    """
    Print the names of files in a folder
    :param folder_name: path of the folder
    """
    for folder, _, filenames in os.walk(folder_name):
        for filename in filenames:
            print(os.path.join(folder, filename))


def get_request(config, evalscript, time_interval, geom=None, image_size=None, data_folder=None, mimetype=MimeType.TIFF, data_coll=DataCollection.SENTINEL2_L2A):
    """
    Creates a SentinelHub API request object.
    :param config: sentinelHub API configurations.
    :param evalscript: band acquisition script
    :param time_interval: tuple (start, end) of the time interval from which to extract the image
    :param geom: geojson geometry object
    :param image_size: image size (it is obtained with load_geometry)
    :param data_folder: image cache folder
    :param mimetype: image saving format
    :param data_coll: collections of data from which to search for images
    :return:
    """
    return SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=data_coll,
                time_interval=time_interval,
                mosaicking_order=MosaickingOrder.LEAST_CC, # Image with less cloud cover in the chosen range
            )
        ],
        responses=[SentinelHubRequest.output_response("default", mimetype)],
        bbox=geom.bbox,
        size=image_size,
        # geometry=geom,
        config=config,
        data_folder= data_folder
    )


def load_geometry(file, resolution):
    """
    Loads a geojson and creates the geometry object and image dimensions to be passed to the API request at the specified resolution
    :param file: path of the geojson file
    :param resolution: resolution in meters of the images to be obtained
    :return: (geometry object, image size)
    """
    imported = json.load(open(file))
    geom = Geometry.from_geojson(imported["features"][0]["geometry"])
    image_size = bbox_to_dimensions(geom.bbox, resolution=resolution)
    print(f"Image shape at {resolution} m resolution: {image_size} pixels")
    return geom, image_size


def save_images(data, area, slots, folder, channel=0):
    """
    Saves images downloaded from the API to a specified folder
    :param date: list of images obtained from the API
    :param area: name of the area for saving purposes
    :param slots: list of dates used in API request phase
    :param folder: folder path
    :param channel: channel of the images (in tiff format) to be saved
    """
    script_dir = os.path.dirname(__file__)
    folder = os.path.join(script_dir, folder+"/")
    if not os.path.isdir(folder):
        os.makedirs(folder)
    for idx, image in enumerate(data):
        if len(image.shape) > 2:
            image = Image.fromarray(image[:,:,channel])
        else:
            image = Image.fromarray(image)
        image.save(folder+"/"+area+"_"+slots[idx][0]+"_"+slots[idx][1]+".tiff")


def show_images(data, slots, start_date, image_size, ncols=4, nrows=3):
    """
    Show ncols*nrows images from the list obtained from the API request starting from the specified date
    :param date: list of images obtained from the API
    :param slots: list of dates used at API request stage
    :param start_date: start date from which to show images
    :param image_size: tuple of image sizes
    :param ncols: number of columns in the plot
    :param nrows: number of rows of the plot
    """
    aspect_ratio = image_size[0] / image_size[1]
    subplot_kw = {"xticks": [], "yticks": [], "frame_on": False}

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols * aspect_ratio, 5 * nrows),
                            subplot_kw=subplot_kw)
    i = 1
    for idx, image in enumerate(data):
        s = slots[idx][0]
        e = slots[idx][1]
        if s >= start_date:
            if i >= ncols*nrows:
                break
            ax = axs[i // ncols][i % ncols]
            ax.imshow(image[:, :, 0])
            ax.set_title(f"{s}  -  {e}", fontsize=10)
            i += 1
    plt.tight_layout()
    plt.show()


def plot_multi_series(series_list, color):
    """
    Graphs the time series of the provided list
    :param series_list: list of time series (python list, numpy array, or pandas series)
    :param color: color of the plot
    """
    for series in series_list:
        plt.plot(series, color=color, label=color)


def select_pixels(img, coord_file):
    """
    Tool for selecting and saving pixels of a given image.
    Press R to reset the coordinate file, ESC to confirm selections.
    Avoid closing the window using the close key.
    :param img: the image from which to select pixels.
    :param coord_file: file in which to save the coordinates
    """

    comms = " R: clean file, ESC confirm "
    # mouse callback function
    def red_dot(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            with open(coord_file, "a") as f:
                f.write("{0}, {1}\n".format(x, y))
            cv2.circle(img, (x, y), 0, (0, 0, 255), -1)

    # interactive display
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    clone = img.copy()
    cv2.namedWindow('Pixel selector '+coord_file+comms, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Pixel selector '+coord_file+comms, red_dot)

    # event handler
    while (1):
        cv2.imshow('Pixel selector '+coord_file+comms, img)
        key = cv2.waitKey(1) & 0xFF
        # escape
        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            return
        # refresh dots
        if key == ord('r'):
            with open(coord_file, "w") as f:
                f.write("")
            img = clone.copy()

    cv2.destroyAllWindows()
    return


def select_single_pixel(img):
    """
    Tool to select only one pixel of a given image.
    Press R to cancel, ESC to confirm the selection.
    Avoid closing the window using the close key.
    :param img: the image from which to select the pixel
    :return: (x, y) coordinates of the pixel
    """
    current_pixel = ()
    comms = " R: reset, ESC confirm "

    # mouse callback function
    def red_dot(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            nonlocal img
            img = clone.copy()
            nonlocal current_pixel
            current_pixel = (x, y)
            cv2.circle(img, (x, y), 0, (0, 0, 255), -1)

    # interactive display
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    clone = img.copy()
    cv2.namedWindow('Pixel selector'+comms, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Pixel selector'+comms, red_dot)

    # event handler
    while (1):
        cv2.imshow('Pixel selector'+comms, img)
        key = cv2.waitKey(1) & 0xFF
        # escape
        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            return current_pixel
        # refresh dots
        if key == ord('r'):
            img = clone.copy()

    cv2.destroyAllWindows()
    return current_pixel




