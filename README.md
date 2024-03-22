# PROMO
Official Repo of Prompt Tuning for Item Cold-start Recommendation （PROMO）

## Datasets
The data pre-processing script is included. 

For example, you could download KuaiRand-Pure data from [here](https://kuairand.com/) and run the script to produce the csv format data.
```shell
cd data/KuaiRand
sh data_preprocess.sh
```  

## Model Training
To train our model on KuaiRand (with default hyper-parameters):
```shell
sh train.sh
```  

