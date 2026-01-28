# LAMDA
# Method
<p align="center">
  <b>Method of LAMDA</b>
</p>
<p align="center">
  <img src="./method.pdf/" style="width:100%"/>
</p>
# Datasets
* **CITYSCAPES**: Follow the instructions in [Cityscapes](https://www.cityscapes-dataset.com/)
  to download the images and semantic segmentation ground-truths. Please follow the dataset directory structure:
  ```html
    <CITYSCAPES_DIR>/             % Cityscapes dataset root
  ├── leftImg8bit/              % input image (leftImg8bit_trainvaltest.zip)
  └── gtFine/                   % semantic segmentation labels (gtFine_trainvaltest.zip)
  ```

* **ACDC**: Download ACDC images and ground truths from [ACDC](https://acdc.vision.ee.ethz.ch/download). Please follow the dataset directory structure:
  ```html
  <ACDC_DIR>/                   % ACDC dataset root
  ├── rbg_anon/                 % input image (rgb_anon_trainvaltest.zip)
  └── gt/                       % semantic segmentation labels (gt_trainval.zip)
  ```
* **GTA5**: Download GTA5 images and ground truths from [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/). Please follow the dataset directory structure:
  ```html
  <GTA5_DIR>/                   % GTA5 dataset root
  ├── images/                   % input image 
  └── labels/                   % semantic segmentation labels
  ```
# Running DHPM
## source training
```
python3 main.py \
  --dataset <source_dataset> \
  --data_root <path_to_source_dataset> \
  --data_aug \
  --lr 0.1 \
  --crop_size 768 \
  --batch_size 2 \
  --freeze_BB \
  --ckpts_path saved_ckpts
```
## Feature optimization
```
python3 PIN_aug.py \
--dataset <source_dataset> \
--data_root <path_to_source_dataset> \
--total_it 100 \
--resize_feat \
--domain_desc <target_domain_description>  \
--save_dir <directory_for_saved_statistics>
```
## Model adaptation
```
python3 main.py \
--dataset <source_dataset> \
--data_root <path_to_source_dataset> \
--ckpt <path_to_source_checkpoint> \
--batch_size 4 \
--lr 0.01 \
--ckpts_path adapted \
--freeze_BB \
--train_aug \
--total_itrs 2000 \ 
--path_mu_sig <path_to_augmented_statistics>
--proto_path <path_to _proto_memory>
```
## Evaluation
```
python3 main.py \
--dataset <dataset_name> \
--data_root <dataset_path> \
--ckpt <path_to_tested_model> \
--test_only \
--val_batch_size 1 \
--ACDC_sub <ACDC_subset_if_tested_on_ACDC>   
```
## Inference&Visualization
```
python3 predict.py \
--ckpt <ckpt_path> \
--save_val_results_to <directory_for_saved_output_images>
```
