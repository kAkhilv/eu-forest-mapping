# EU Forest Mapping
Identify the type of forest cover using Sentinel-2 data and Deep Learning.
The model uses a pre-trained model for forests more focussed in the EU-region.

## Prerequisites
[Pytorch](https://pytorch.org/get-started/locally/) <br>
A [Sentinelhub](https://www.sentinel-hub.com/) account to use the API and corresponding libraries. <br>

## Run the code locally
Clone the repo <br>
```
git clone https://github.com/kAkhilv/eu-forest-mapping.git
 ```
Create a virtual environment
```
cd eu-forest-mapping
python -m venv eoenv
source eoenv/bin/activate
```
Install the requirements
```
pip install -r requirements.txt
```
Replace the CRS coordinates of the Area of Interest (AoI) in ```coordinates.geojson ``` file.
The AoI can be split into number of smaller bounding boxes using the ```sentinelhub.areas.BBoxSplitter``` class.

## Workflow
### Acquiring the raw data
Sentinel-2 imagery for the defined Area of Interest (AoI) is acquired by dividing the region into smaller bounding box patches. This patch-wise approach streamlines data retrieval and ensures efficient coverage. To optimize reuse and minimize redundant downloads, it is recommended to store the fetched images locally. A sample patch of the raw image is as shown below:
<img src="https://raw.githubusercontent.com/kAkhilv/eu-forest-mapping/main/raw_image.png" alt="Raw Image from a patch" width="400"/>

### The ConvLSTM model
ConvLSTM is a deep learning model known for its effectiveness in capturing spatio-temporal dependencies, making it particularly suitable for tasks such as vegetation forecasting and climate-related predictions. In this project, a pretrained ConvLSTM model—provided by [Vision—Impulse](https://www.vision-impulse.com/en-us/) is used for both training and inference, leveraging pretrained weights to accelerate development and improve performance. 

### Output
The pretrained model is used for running inference on the downloaded data. The inference generates individual results for each patch. 
The result show the coniferous trees class in light green and broadleaf trees class in dark green.
<img src="https://raw.githubusercontent.com/kAkhilv/eu-forest-mapping/main/output-png.png" alt="Raw Image from a patch" width="400"/>

### References
[eolearn](https://eo-learn.readthedocs.io/en/latest/) <br>
[ConvLSTM](https://proceedings.neurips.cc/paper_files/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)







