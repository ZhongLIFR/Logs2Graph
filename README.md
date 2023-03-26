# Logs2Graph

Steps to use our code:

## Step1: Download Dataset
Please download dataset from this link: https://doi.org/10.5281/zenodo.7771388, and put it under the root_path 

## Step2: Replace root_path

Replace the variable "root_path" at the beginning of each python script with your own "root_path". For example, 
```
root_path = r'/Users/YourName/Desktop/OCDIGCN'
```

## Step3: Testing
1. for testing Logs2Graph on HDFS: run GraphGeneration_HDFS.py first, and then run main_HDFS.py.


## References
Our code is developed based on [GLAM](https://github.com/sawlani/GLAM) and [DiGCN](https://github.com/flyingtango/DiGCN)
