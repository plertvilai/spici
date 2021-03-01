# spcserver

SPCServer - Downloads and uploads data to spc.ucsd.edu

## System requirement

1. Python 3.5 or higher
2. Example: Create a python environment named 'spici_env' and install required libraries using pip:
    - `virtualenv spici_env`
    - `source spici_env/bin/activate`
    - `pip3 install -r requirements.txt`

Conda also works as well.

## Usage

Run `spc_go.py` to download images from the spc.ucsd.edu. 
Create an `SPCServer` to upload images to the spc.ucsd.edu


### Download
The input requires one file:
1. A search parameter text file containing the start and end dates, minimum/maximum organism length, and camera choice


This example shows you how to both download and upload images to spc.ucsd.edu using python.
```
    python spc_go.py --search-param-file=examples/summer2019/time_period.txt --image-output-path=examples/summer2019/images --meta-output-path=examples/summer2019/meta_data.csv -d=True
```

This downloads images for the 3 following time intervals to `examples/summer2019/images`
```
    2019-05-23 09:53:00,2019-05-23 10:27:00,0.03,1.0,SPCP2
    2019-05-28 10:11:00,2019-05-28 10:45:00,0.03,1.0,SPCP2
    2019-05-30 09:58:00,2019-05-30 10:32:00,0.03,1.0,SPCP2
```
It will also output the meta data csv file to `examples/summer2019/meta_data.csv`

The output consists of two files:

1. A meta data file (in csv file formating) containing the image's id, min/max length, timestamp, etc.
2. A collection of images pulled from the website based off the desired search parameters.

### Upload
The input requires two files:

1. A predictions text file containing the image file names and their respective enumerated labels
2. A labels text file containing the enumerated labels mapped to their class names

Once you have images ready to be uploaded, you can create an `SPCServer` 
object and use its method
`upload` to upload your desired images with their labels.

An example is provided below:

`#todo give explanation of example`

```
# Create SPCServer
spc = SPCServer()

# Initialize inputs
account_info = {'username': 'kevin', 'password': 'ceratium'}
login_url = 'http://spc.ucsd.edu/data/admin/?next=/data/admin'
predictions = 'examples/prorocentrum/predictions.txt'
label_file = 'examples/prorocentrum/labels.txt'

# Initialize submission dictionary parameters
spc.submit_dict['name'] = 'kevin_test'  # NAME OF LABELING INSTANCE
spc.submit_dict['tag'] = 'kevin_test'
spc.submit_dict['is_machine'] = True  # BOOL FOR MACHINE UPLOAD. IF FALSE, USES USER LOG-IN AS ANNOTATOR NAME
spc.submit_dict['machine_name'] = 'kevin_test'

# Upload images
spc.upload(login_url=login_url,
           account_info=account_info,
           textfile=predictions,
           label_file=label_file)

