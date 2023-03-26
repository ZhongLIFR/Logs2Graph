# Logs2Graph

Steps to use our code:

## Step0: Check requirements
Please check you have the specified environment, which is described in requirements.txt

## Step1: Download Dataset
Please download dataset from this link: [zenodo](https://doi.org/10.5281/zenodo.7771548), and put them under the root_path (namely where all python scripts are located) with a name "Data". Particularly, if the downloaded zip file has a name other than "Data" after unziping it, you should change it to "Data".  

## Step2: Replace root_path

Replace the variable "root_path" at the beginning of each python script with your own "root_path". For example, 
```
root_path = r'/Users/YourName/Desktop/OCDIGCN'
```

## Step3: Testing
1. for testing Logs2Graph on HDFS: run GraphGeneration_HDFS.py, and then run main_HDFS.py.


## References
Our code is developed based on [GLAM](https://github.com/sawlani/GLAM) and [DiGCN](https://github.com/flyingtango/DiGCN)
