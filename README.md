<img align="left" width="500" src="readme_imgs/vrai_logo_.jpg" style="margin-right:-230px"></br></br></br></br>
<h1> VegRecoverAI: A Deep Learning-Based System for Automated Vegetation Recover Assessment and Prediction after gas pipeline construction</h1></br>

## Overview

This repository contains the code, models, and resources for the project **"VegRecoverAI: A Deep Learning-Based System for Automated Vegetation Recover Assessment and Prediction after gas pipeline construction"**. This research introduces a novel AI-driven approach, VegRecoverAI, to address the ecological challenges posed by gas pipeline construction. This system integrates deep learning algorithms and NDVI time-series data derived from satellite imagery (Sentinel-2, PlanetScope, and Landsat) to monitor vegetation changes and predict recovery trends. The results underline the system's potential to streamline ecological monitoring, support restoration projects, and contribute to global climate change mitigation efforts by ensuring timely interventions and sustainable practices.

<div style="text-align: center;">
    <img src="readme_imgs/change_detection.png?raw=true" alt="Change Detection" width="450" style="display: block; margin: center;"/>
</div>
<div style="text-align: center;">
    <img src="readme_imgs/results1.png?raw=true" alt="Time Series Forecasting" width="800" style="display: block; margin: center;"/>
</div>
<div style="text-align: center;">
    <img src="readme_imgs/results2.gif?raw=true" alt="Valutazione del ripristino" width="200" style="display: block; margin: center;"/>
</div>




## Structure
The code is divided into some scripts to be executed locally related to the extraction of time series from Sentinel-2, Landsat8 and Planet images; and by a Jupyter notebook  (*Change_Detection_Forecasting_Restore_Point.ipynb*) for the change detection, forecasting and restoration status assessment phase of the study areas.

### Time Series Extraction
## Requirements
This project requires **Python 3.10**. Then, install the packages in  *requirements.txt*.

```
pip install -r requirements.txt
```

The script for Sentinel-2 and Landsat8 images is. *main.py*.
First set up the API credentials for SentinelHub.

```
    config = SHConfig()
    # Specify the sentinel API Key in this section.
    '''config.instance_id = '<your instance id>'
    config.sh_client_id = '<your client id>'
    config.sh_client_secret = '<your client secret>'
    config.save()'''
```
[//]: ![](readme_imgs/config.png?raw=true)

Then create the geoJson of the polygon enclosing the study area using QGIS. Place the geoJson in the *geoms* (if absent, create it in the same root as the script).
Then enter the various parameters such as geoJson file name, resolution in meters of the NDVI and color images, time period of the images, and the parameters of the API request. Specifically the *evalscript* that specifies the bands to be acquired (see the *evalscript.py* file for those used), and *data_collection* that specifies the data collection from which to download images (Sentinel-2 L2A, Landsat8, etc...).

```
    # Name of the geojson to be used within the geoms folder.
    area = "cd5"

    # Loading geometries for NDVI/GNDVI images.
    resolution = 10 #meters
    geom, size = utils.load_geometry("geoms/"+area+".geojson", resolution)

    # Loading geometries for reference color image
    resolution_c = 10 #meters
    geom_color, size_color = utils.load_geometry("geoms/"+area+".geojson", resolution_c)

    # Create list of date ranges : list of tuples (start, end)
    slots = utils.dates_list("2017-01-01", "2022-12-31", "7D", 7)

    # Create list of requests to API
    # The important parameters are the evalscript and the data collection specifying the satellite
    list_of_requests = [utils.get_request(config, es.evalscript_raw, slot, geom, size, "cache",
                                        data_coll=DataCollection.SENTINEL2_L2A) for slot in slots]
    request_true_color = utils.get_request(config, es.evalscript_true_color, ("2022-06-01", "2022-06-30"), geom_color, size_color,
                                       "true_color", MimeType.PNG, data_coll=DataCollection.SENTINEL2_L2A)
```
[//]: ![](readme_imgs/area_settings.png?raw=true)

At this point, at the bottom of the script the smoothing parameters can be set and one of several available main functions launched. For extracting the time series of all pixels in the study area, *main_full()* is used. 

```
    params = {
        "frac": 0.06, # Used by the lowess
        "n_splines": 72, # Used by the GAM
        "alpha": 1, # Used in the removal of outliers
        "lam": 0.1, # Used by the GAM
        "remove_outliers": True, # Performing or not removing pre-smoothing outliers
        "frac_outliers": 0.06, # Used by the lowess of outlier removal
        "method": "gam", # Smoothing method to be used
        "plot": True, # Enable some plots of graphs
    }
```
[//]: ![](readme_imgs/run_sentinel.png?raw=true)

Running *main_full()* will generate in the *ts/* folder a csv with the nomenclature *"nameArea_h_w.csv ”* where *h* and *w* are the height and width of the retrieved images, respectively. In the *areas_img* folder instead, the reference color image *"nameArea.png ”* will be saved.
These files are to be used later in the Jupyter notebook.

For Planet images use the *planet.py* script instead. The procedure is similar, however Planet images must currently be downloaded manually from the web portal. Finally, the path to the folder containing the images must be specified in the script. As in the case of SentinelHub, the smoothing parameters should be set and one of the *main* functions defined in the script should be executed.

### Change Detection, Forecasting, State of restoration
For these steps use the Jupyter notebook *Change_Detection_Forecasting_Restore_Point.ipynb* provided in the repository. Follow the instructions within it for execution.

