# GRIT
Here we present the source codes for GRIT. GRIT is an end-to-end model designed for social relation inference. The paper can be found [here](https://github.com/IFBigData/GRIT/blob/master/8731paper.pdf).

![Alt text](./GRIT_framework.jpg)

The model weights can be downloaded from [Google Drive](https://drive.google.com/file/d/1l9piylE1ZwmGHWBPXlE0F_kOQXjA-hj4/view?usp=sharing) or [BaiduYunPan](https://pan.baidu.com/s/1H-BISGrijHQQ6LS1wAbYhQ)[password: 8731]

-- python version is 3.6.9

-- torch version is 1.10.0

## Training
To train GRIT-SW224 on PISC_Fine dataset, run run_main.sh or the following
```
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --master_port 8730 main_ddp.py --output_dir=Transformer_output --img_size=224 --dataset=pisc_fine --backbone=swin_transformer
```
To train GRIT-SW384 on PISC_Fine dataset, run the following
```
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --master_port 8730 main_ddp.py --output_dir=Transformer_output --img_size=384 --dataset=pisc_fine --backbone=swin_transformer
```
To train GRIT-R101 on PISC_Fine dataset, run the following
```
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --master_port 8730 main_ddp.py --output_dir=Transformer_output --img_size=448 --dataset=pisc_fine --backbone=resnet101
```

## Evaluation
To evaluate the trained GRIT-SW224 model on PISC_Fine dataset, run the following
```angular2
python main_eval.py --model_path=path_to_GRIT-SW224 --dataset=pisc_fine --img_size=224
```
To evaluate the trained GRIT-SW384 model on PISC_Fine dataset, run the following
```angular2
python main_eval.py --model_path=path_to_GRIT-SW384 --dataset=pisc_fine --img_size=384
```

## Visualization
To visualize the attention map from transformer's decoder, run the following
```angular2
python visualization.py --model_path=path_to_GRIT-SW224 --dataset=pisc_fine --img_size=224
```

