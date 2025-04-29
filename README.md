# PROMO
Prompt Tuning for Item Cold-start Recommendation （PROMO）

## Datasets
The data pre-processing script is included. 

For example, you could download KuaiRand-Pure data from [here](https://kuairand.com/) and run the script to produce the csv format data.
```shell
cd data
python preprocess.py --dataset=KuaiRand --preprocess=True --save=True
python preprocess.py --dataset=MovieLens100k --preprocess=True --save=True
```  

## Model Training
To train our model on KuaiRand (with default hyper-parameters):
```shell
sh train.sh
```  

