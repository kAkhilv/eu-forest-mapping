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
Replace the CRS coordinates of the Area of Interest(aoi) in ```coordinates.geojson ``` file.
The aoi can be splpit into number of smaller bounding boxes using the ```sentinelhub.areas.BBoxSplitter``` class.

## The ConvLSTM model





