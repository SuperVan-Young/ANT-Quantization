# ANT Simulator

This repository contains the source code for a research paper that published at MICRO2022.

## Prerequisite

+ Ubuntu 18.04.5 LTS
+ Andconda 4.10.1
+ Python 3.8
+ gcc 7.5.0

## Getting Started

```shell
$ # Environment.
$ conda create -n ant_sim python=3.8
$ conda activate ant_sim  
$ pip install -r  requirements.txt
$ # Cacti for the memory simulation.
$ cd ./ant_isca_dev/ant_simulator
$ git clone https://github.com/HewlettPackard/cacti ./bitfusion/sram/cacti/
$ make -C ./bitfusion/sram/cacti/
$ # Run ANT simulation.
$ python ./run_ant.py
```

## Evaluation

The script `run_ant.py` generates statistic data and stores it in file `./result/ant_res.csv`. Note that BiScaled only test on VGG16 and ResNet50.

In `./result/ant_res.csv`, Line 3 shows the **cycle** data that normalized with AdaFloat. Line 7-10 shows the **energy** data that normalized with AdaFloat.

As shown below, the `./result/ANT-simulator.xlsx` provides the template. You can fill it with the numbers of `./result/ant_res.csv` to generate Figure 13 in the paper.

<div>
<img src=./docs/img/evaluation.png width=100%>
</div>