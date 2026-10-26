# DMFTrack

## Usage
### Installation
Create and activate a conda environment:
```
conda create -n DMFTrack python=3.8
conda activate DMFTrack
```

Install the required packages:
```
pip install -r requirements.txt
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST         
   ```

### Path Setting
Run the following command to set paths:
```
cd <PATH_of_DMFTrack>
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Training
Download pre-trained [DeiT-Tiny weights](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth), [Eva02-Tiny weights](https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/pt/eva02_Ti_pt_in21k_p14.pt) , and [ViT-Tiny weights](https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz)  and put it under `$USER_ROOT$/.cache/torch/hub/checkpoints/. 
```
# Training DMFTrack-DeiT
# DMFTrack Stage1
python tracking/train.py --script dmftrack_st1 --config deit_tiny_patch16_224  --save_dir ./output --mode single
# DMFTrack Stage2
python tracking/train.py --script dmftrack_st2 --config deit_tiny_patch16_224  --save_dir ./output --mode single --script_prv dmftrack_st1 --config_prv deit_tiny_patch16_224

```


### Testing

Put the downloaded weights on `<PATH_of_DMFTrack>/output/checkpoints/train/dmftrack_st2/deit_tiny_patch16_224`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

 Testing examples:
- VisDrone2018 or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py dmftrack_st2 deit_tiny_patch16_224 --dataset visdrone2018 --threads 4 --num_gpus 1
python tracking/analysis_results.py # need to modify tracker configs and names
```
- UAV123
```
python tracking/test.py dmftrack_st2 deit_tiny_patch16_224 --dataset uav123 --threads 4 --num_gpus 1
```


### Test FLOPs, and Params.

```
# Profiling DMFTrack-DeiT
python tracking/profile_model.py --script dmftrack_st2 --config deit_tiny_patch16_224
```


## Acknowledgment
* This repo is based on [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library which are excellent works and help us to quickly implement our ideas.

* We use the implementation of the DeiT, Eva02, and ViT from the [Timm](https://github.com/rwightman/pytorch-image-models) repo. 


