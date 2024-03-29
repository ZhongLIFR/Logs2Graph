# Logs2Graphs

This repository contains the lab code for the manuscript "Graph Neural Networks based Log Anomaly Detection and Explanation", which has been accepted by [ICSE'24](https://conf.researchr.org/home/icse-2024) poster track (short paper). The full version is under review with the [EAAI journal](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence). Importantly, we are deploying this algorithm on lithography systems with our individual partners [ASML](https://www.asml.com/en) and [VDL](https://www.vdlgroep.com/en).

Readers can follow these steps to use our code:

## Step0: Check requirements
Please ensure that you have the specified environment, which is described in requirements.txt

## Step1: Download Dataset
Please download the dataset "Data.zip" from this link: [zenodo](https://doi.org/10.5281/zenodo.7771548), and put them under the root_path (namely where all python scripts are located) with a name "Data". If the downloaded zip file has a name other than "Data" after unziping it, you should change it to "Data".  

## Step2: Replace root_path

Replace the variable "root_path" at the beginning of each python script with your own "root_path". For example, 
```
root_path = r'/Users/YourName/Desktop/Logs2Graph'
```

## Step3: Testing
1. for testing Logs2Graph on HDFS: run GraphGeneration_HDFS.py, and then run main_HDFS.py. 
2. for testing Logs2Graph on Hadoop: run GraphGeneration_Hadoop.py, and then run main_Hadoop.py.
3. for testing Logs2Graph on Spirit: run GraphGeneration_Spirit.py, and then run main_Spirit.py.
4. for testing Logs2Graph on BGL: run GraphGeneration_BGL.py, and then run main_BGL.py.
5. for testing Logs2Graph on Thunderbird: run GraphGeneration_Thunderbird.py, and then run main_Thunderbird.py.

## References
Our code is developed based on [GLAM](https://github.com/sawlani/GLAM) and [DiGCN](https://github.com/flyingtango/DiGCN).
