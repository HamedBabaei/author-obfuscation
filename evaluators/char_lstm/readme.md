# Character based LSTM for authorship identification

## Prepare the data
Run the following command to build the datasets from the original PAN datasets.
The first argument is the path to the root of the PAN dataset. 
```bash
python prepare_data.py path_to_the_root_of_the_dataset
```
It assumes the dataset is the PAN dataset for the obfuscation task. 

### Parameters
`--output`: the output path, default is `data_obf`
`obf`: a parameter which determines if the data is from obfuscation or not.
The default is `true`

## train
To train the model run the following command
```bash
python train.py
```
The trained model will be saved in `models_obf` folder.
Run the command with `-h` parameter to see the options

## test
To test the model run the following command
```bash
python test.py
```
Run the command with `-h` parameter to see the options

## Prerequisites
You need to have `pytorch` installed to be able to use this program.
