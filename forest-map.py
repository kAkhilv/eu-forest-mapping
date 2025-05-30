
# Import the libraries

import datetime
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
from sentinelhub import (
    CRS,
    SHConfig,
    DataCollection,
    Geometry,
    MimeType,
    SentinelHubRequest,
    BBoxSplitter,
    UtmZoneSplitter)
from eolearn.core import (
    AddFeatureTask,
    EONode,
    EOPatch,
    EOTask,
    EOWorkflow,
    FeatureType,
    LoadTask,
    OverwritePermission,
    SaveTask,
)
from eolearn.io import SentinelHubInputTask

# Set the configuration for Sentinel Hub

config = SHConfig()
config.sh_client_id = 'baebdf3d-21a9-4e08-8cd2-c7419df585cb'
config.sh_client_secret = 'CgMycZHiVotSQ8ACwNxqEvdUQ9Uizyrd'# Replace with your own credentials
if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")
    
# Load of the coordinates of Area of Interest (AOI) from a GeoJSON file

filename='coordinates.geojson'
file = open(filename, 'r', encoding='utf-8')
geometry = gpd.read_file(file)
aoi_coordinates = Polygon(geometry['geometry'][0].exterior.coords)
# plt.plot(*aoi_coordinates.exterior.xy, color='red', linewidth=2)
# plt.show()

def get_true_color_image(geometry, time_interval = ("2024-05-01", "2024-05-31")):
    evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    } """
    aoi_geometry = Geometry(geometry, crs=CRS.WGS84)
    request_true_color = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    geometry=aoi_geometry,
    config=config
    )
    true_color_imgs = request_true_color.get_data()
    tc_image = true_color_imgs[0]
    return tc_image

tc_image = get_true_color_image(geometry)
    
def aoi_spliiter(aoi_coordinates, max_area=1000000):
    """
    Split the AOI into smaller polygons if it exceeds the maximum area.
    """
    
    splitter = BBoxSplitter(max_area=max_area)
    return splitter.split(aoi_coordinates)


