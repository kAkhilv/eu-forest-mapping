
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


import torch
import torch.nn as nn
class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.i_dim = input_dim
        self.h_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.i_dim + self.h_dim,
            out_channels=4 * self.h_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.h_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(torch.nn.Module):
    def __init__(self, height, width, input_dim=13, hidden_dim=16, nclasses=4, kernel_size=(3, 3), bias=False):
        super(ConvLSTM, self).__init__()

        self.inconv = torch.nn.Conv3d(input_dim, hidden_dim, (1, 3, 3))

        self.cell = ConvLSTMCell(
            input_size=(height, width), input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias
        )

        self.final = torch.nn.Conv2d(hidden_dim, nclasses, (3, 3))

    def forward(self, x, hidden=None, state=None):
        x = x.permute(0, 4, 1, 2, 3)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), "constant", 0)
        x = self.inconv.forward(x)

        b, c, t, h, w = x.shape
        if hidden is None:
            hidden = torch.zeros((b, c, h, w))
        if state is None:
            state = torch.zeros((b, c, h, w))

        if torch.cuda.is_available():
            hidden = hidden.cuda()
            state = state.cuda()

        for iter in range(t):
            hidden, state = self.cell.forward(x[:, :, iter, :, :], (hidden, state))

        x = torch.nn.functional.pad(state, (1, 1, 1, 1), "constant", 0)
        x = self.final.forward(x)

        return x
    
def convlstm(num_classes=3, in_channels=4):
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