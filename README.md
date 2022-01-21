# WSDM Cup Submission 2022
Based on the bench GNN framework [link](https://github.com/xkcd1838/bench-DGNN).

## 1. Installation
### 1.1 Requirements
1. Install [Singularity 3.5.3](https://github.com/hpcng/singularity/blob/master/INSTALL.md).
2. Install [CUDA 11.0](https://developer.nvidia.com/cuda-11.0-download-archive) (Optional for CPU training)

Memory requirements differ between datasets and models. 64GB RAM and 24GB vRAM is required for dataset B. While dataset A can train on less than 16Gb vRAM.

### 1.2 Build container
In the root directury, run the following script to build the singularity container. Expect the building to take a couple of hours. The script builds the container and places it in the parent directory (outside of the repository). The container takes up roughly 7.5GB disk space.
```
./build_container.sh 
```

If you do not wish to use a singularity container, you can look at the installation process in the container definition file, `container.def` and install the dependencies manually. 

## 2. Download datasets
Place the downloaded datasets in the `experiment/data` folder. There is a subfolder for each WSDM dataset.
Input datasets should be named accoding to their sets e.g. input_A_intermediate.csv.gz and input_B_test.csv.gz

## 4. Quick run
Code and scripts relevant to the experiment is found in the `experiment` folder. You can run a quick test run (3 min) with the `quick.yaml` found in the `config` folder. Run scripts are found in the `run_scripts` folder. How to run a quick run is shown below.

```
./run_scripts/run_cpu.sh config/quick.yaml #Change to run_gpu.sh if you want to untilize the GPU
```

If you wish to run the quick run without the helper script that can be done with the following command.
```
singularity exec ../../container.sif python run_exp.py --config_file config/quick.yaml
```

## 5. Running in the background
Helper scripts for convenient background running is also included.
```
./run_scripts/nohup_run_gpu.sh config/quick.yaml
```

## 6. Reproducing results
For dataset A:
```
./run_scripts/run_gpu.sh config/wsdm/wsdm-A_gclstm.yaml
```
Dataset B:
```
./run_scripts/run_gpu.sh config/wsdm/wsdm-B_gclstm.yaml
```

### 6.2 Final results
The predictions will be saved in the experiments/predictions folder.
Under the submission folder there is a prepare script which prepared the saved predictions for submission.
```
./prepare.sh path_to_dataset_a_predictions.csv.gz path_to_dataset_b_predictions.csv.gz name_of_zipfile.zip
```
