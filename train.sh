python main_baseline.py --dataset KuaiRand --model_name SASRec --train_dir SASRec --exp_name base --num_epochs 30 --device cuda:0 --l2_emb 0.01
python main_pretrain.py --dataset KuaiRand --model_name DSSM_SASRec --train_dir DSSM_SASRec --exp_name base --num_epochs 20 --device cuda:0 --l2_emb 0.01 --pretrain_model_path ./KuaiRand_SASRec/base/best.pth
python main.py --dataset KuaiRand --model_name DSSM_SASRec_PTCR --train_dir DSSM_SASRec_PTCR --exp_name base --num_epochs 20 --device cuda:0 --l2_emb 0.01 --pretrain_model_path ./KuaiRand_DSSM_SASRec/base/best.pth --alpha 0.01 --beta 0.01
