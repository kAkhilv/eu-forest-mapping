
# Import the libraries

import datetime
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
from ConvLSTM import ConvLSTM
import torch
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
aoi_gdf = gpd.read_file(file)
aoi_coordinates = Polygon(aoi_gdf['geometry'][0].exterior.coords)
aoi_geometry = Geometry(aoi_coordinates, crs=CRS.WGS84)
aoi_crs = aoi_gdf.crs
# plt.plot(*aoi_coordinates.exterior.xy, color='red', linewidth=2)
# plt.show()

def get_true_color_image(aoi_geom, time_interval = ("2024-05-01", "2024-05-31")) -> np.ndarray:
    """
    Function to retrieve a true color image from Sentinel-2 L2A data for a specified geometry and time interval.
    Args:
        geometry (dict or shapely.geometry): The geometry defining the area of interest (AOI), in GeoJSON-like format or as a shapely geometry.
        time_interval (tuple of str, optional): A tuple specifying the start and end dates (YYYY-MM-DD) for the image acquisition period. Defaults to ("2024-05-01", "2024-05-31").
    Returns:
        numpy.ndarray: The true color image as a NumPy array (RGB), corresponding to the specified AOI and time interval.
    Raises:
        Any exceptions raised by SentinelHubRequest or data retrieval will propagate.
    Note:
        Requires Sentinel Hub configuration and dependencies to be properly set up.
    """
    
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
    request_true_color = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    geometry=aoi_geom,
    config=config
    )
    true_color_imgs = request_true_color.get_data()
    tc_image = true_color_imgs[0]
    return tc_image
    
bbox_splitter = BBoxSplitter([aoi_coordinates], aoi_crs, split_shape=(10,10))
bbox_list = np.array(bbox_splitter.get_bbox_list())
info_list = np.array(bbox_splitter.get_info_list())

download_task = SentinelHubInputTask(
    data_collection=DataCollection.SENTINEL2_L2A,
    bands_feature=(FeatureType.DATA, "BANDS"),
    resolution=10,
    time_difference=datetime.timedelta(minutes=120),
    maxcc=0.2,
    bands_dtype=np.uint16,
    bands=["B02", "B03", "B04", "B08"],
    config=config,
)
save_task = SaveTask(path= "./EOpatches", overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)
# Setup the nodes
download_node = EONode(download_task)
save_node = EONode(save_task, inputs=[download_node])

# Define the workflow
workflow = EOWorkflow([download_node, save_node])

class ScaleImageTask(EOTask):
    def __init__(self):
        super().__init__()

    def process_image(self, image):
        img = np.clip(image[:, :, :, :3], a_min=0, a_max=1500)
        img = img / 1500 * 255
        img = np.concatenate((img, image[:, :, :, 3][..., None]), axis=-1)
        img = img.astype(np.uint8)
        return img

    def execute(self, eopatch):
        image = eopatch[FeatureType.DATA]["BANDS"]
        scaled_image = self.process_image(image)
        eopatch[FeatureType.DATA]["SCALED_BANDS"] = scaled_image

        return eopatch
    
scale_image_task = ScaleImageTask()
download_node_with_scale = EONode(download_task)
scale_node = EONode(scale_image_task, inputs=[download_node_with_scale])
save_node_with_scale = EONode(save_task, inputs=[scale_node])
workflow_with_scale = EOWorkflow([download_node_with_scale, scale_node, save_node_with_scale])

# Run the workflow over selected patches only instead of the complete AOI
TILE_IDS = [
    33,
    42,
    51,
    60,
    32,
    41,
    50,
    59,
    31,
    40,
    49,
    58,
    30,
    39,
    48,
    57,
]

for idx in TILE_IDS:
    result = workflow.execute(
        {
            download_node: {"bbox": bbox_list[idx], "time_interval": ["2021-07-01", "2021-09-30"]},
            save_node: {"eopatch_folder": f"eopatch_{idx}"},
        }
    )  
eopatch = EOPatch.load("./EOpatches/eopatch_33")
scaled_image = eopatch[FeatureType.DATA]["BANDS"]

img = np.clip(scaled_image[:, :, :, :3], a_min=0, a_max=1500)
img = img / 1500 * 255
img = np.concatenate((img, scaled_image[:, :, :, 3][..., None]), axis=-1)
img = img.astype(np.uint8)

# Plotting 5 images in a vertical stack
fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(5, 15))

for i in range(5):
    axs[i].imshow(img[i][..., [2, 1, 0]])  # RGB: BGR to RGB
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_aspect("auto")
plt.tight_layout()
plt.show()

    
def convlstm(num_classes=3, in_channels=4) -> ConvLSTM:
    return ConvLSTM(
        512, 512, input_dim=in_channels, hidden_dim=24, nclasses=num_classes, kernel_size=(3, 3), bias=False
    )  
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
class InferenceTask(EOTask):
    def __init__(self, model_file):
        self.model = convlstm(num_classes=4, in_channels=4).to(device)
        self.model.load_state_dict(torch.load(model_file, map_location=device))
        self.model.eval()
        self.add_output = AddFeatureTask((FeatureType.MASK_TIMELESS, "mask"))

    def execute(self, eopatch):
        image = eopatch[FeatureType.DATA]["BANDS"]
        image = np.array(image, dtype=np.float32)
        image /= 65535.0

        image = torch.Tensor(image).unsqueeze(0).to(device)

        output = self.model(image)
        output = torch.argmax(output, dim=1)
        output = output.squeeze(0).unsqueeze(-1).cpu().numpy()

        eopatch = self.add_output(eopatch, output)
        
        return eopatch
    
model_file = "./forest-map/vi_forest_model_weights.pth"
inference_task = InferenceTask(model_file)

save_inference_task = SaveTask("./Predictions", overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)

load_node = EONode(LoadTask("./EOpatches"))
inference_node = EONode(inference_task, inputs=[load_node])
save_inference_node = EONode(save_inference_task, inputs=[inference_node])

inference_workflow = EOWorkflow([load_node, inference_node, save_inference_node])

for tile_id in TILE_IDS:
    inference_workflow.execute(
        {
            load_node: {"eopatch_folder": f"eopatch_{tile_id}"},
            save_inference_node: {"eopatch_folder": f"eopatch_{tile_id}"},
        }
    )
    
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))

for i, tile_id in enumerate(TILE_IDS):
    eopatch = EOPatch.load(f"./EOpatches/eopatch_{tile_id}")
    scaled_image = eopatch[FeatureType.DATA]["BANDS"]
    img = np.clip(scaled_image[:, :, :, :3], a_min=0, a_max=1500)
    img = img / 1500 * 255
    img = np.concatenate((img, scaled_image[:, :, :, 3][..., None]), axis=-1)
    img = img.astype(np.uint8)

    ax = axs[i // 4][i % 4]
    ax.imshow(img[0][..., [2, 1, 0]])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")

fig.subplots_adjust(wspace=0, hspace=0)

COLOR_ENCODING = {
    0: [255, 255, 255],
    1: [70, 158, 74],
    2: [28, 92, 36],
    3: [255, 255, 255],
}


def labelVisualize(img, num_class=3):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = COLOR_ENCODING[i]
    return img_out.astype(np.uint8)

fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
for i, tile_id in enumerate(TILE_IDS):
    inferenced_eopatch = EOPatch.load(f"./Predictions/eopatch_{tile_id}")
    output = inferenced_eopatch[FeatureType.MASK_TIMELESS]["mask"].squeeze(-1)
    output[output == 2] = 1
    output = labelVisualize(output)
    ax = axs[i // 4][i % 4]
    ax.imshow(output)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")

fig.subplots_adjust(wspace=0, hspace=0)

fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
for i, tile_id in enumerate(TILE_IDS):
    inferenced_eopatch = EOPatch.load(f"./Predictions/eopatch_{tile_id}")
    output = inferenced_eopatch[FeatureType.MASK_TIMELESS]["mask"].squeeze(-1)
    output = labelVisualize(output)
    ax = axs[i // 4][i % 4]
    ax.imshow(output)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")

fig.subplots_adjust(wspace=0, hspace=0)