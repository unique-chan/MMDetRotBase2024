<h1 align="center" style="font-weight: 500; line-height: 1.4;">
  Baseline Code for Remote Sensing Object Detection
</h1>

<p align="center">
  <a href="#"><img alt="Python3.7+" src="https://img.shields.io/badge/Python-3.7+-blue?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch1.6~1.10.2" src="https://img.shields.io/badge/PyTorch-≥1.6, ≤1.10-orange?logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="MMDetection2.28.2" src="https://img.shields.io/badge/MMDetection-2.28.2-red?logo=mmlab&logoColor=white"></a>
  <a href="#"><img alt="MMRotate0.3.4" src="https://img.shields.io/badge/MMRotate-0.3.4-hotpink?logo=mmlab&logoColor=white"></a>
  <a href="#"><img alt="Apache-2.0" src="https://img.shields.io/badge/License-Apache 2.0-green?logo=Apache"></a>
</p>

<p align="center">
  <b><a href="https://github.com/unique-chan">Yechan Kim</a></b> 
</p>

### This repo includes:
* Training & test code for oriented object detection
* **[NOTE]** We only consider single-machine multi-GPU scenarios and do not address cases involving multiple nodes and using Slurm. For explanations related to multi-machine setups and Slurm, please refer to the official documentation of [MMDetection](https://mmdetection.readthedocs.io/en/v2.28.2/) and [MMRotate](https://mmrotate.readthedocs.io/en/v0.3.4/).

### Announcement:
* 5 Feb. 2025: Now, confusion_matrix.py in MMRotate properly uses `args.color_theme`. Besides, you can find two types of confusion matrix results with file extensions, `*.png` and `*.svg`!
* 5 Feb. 2025: We modified _crop_data() of `RRandomCrop` in the original MMRotate code (v0.3.4). Now we can use the transform named `RRandomCrop` in train/val/test pipelines of your configuration file without bbox mismatch error! 👏
* 4 Feb. 2025: We modified `print_map_summary()` of eval_map.py in the original MMRotate code (v0.3.4) in order to avoid runtime error when #classes of model > #classes of given data! 👏

### Preliminaries:


* **Step 1**. Create a conda environment with Python 3.8 and activate it.
    ~~~shell
    conda create --name MMBase python=3.8 -y
    conda activate MMBase
    ~~~

* **Step 2.** Install PyTorch with TorchVision following [official instructions](https://pytorch.org/get-started/locally/). The below is an example. We do not recommend PyTorch 2.x for our code.
    ~~~shell
    pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html  
    ~~~

* **Step 3.** Install `MMDetection (v2.28.2)` ([v2.28.2](https://mmdetection.readthedocs.io/en/v2.28.2/) is the latest version of 2024 suited to MMRotate).
    ~~~shell
    # ⚠️ No need to clone MMDet (e.g. "git clone -b 2.x https://github.com/open-mmlab/mmdetection; rm -rf mmdetection/.git"). Already cloned! 
    pip install -U openmim==0.3.9
    mim install mmcv-full==1.7.2
    pip install -v -e mmdetection/
    ~~~

* **Step 4.** Install `MMRotate (v0.3.4)` ([v0.3.4](https://mmrotate.readthedocs.io/en/v0.3.4/) is the latest version of 2024). 
    ~~~shell
    # ⚠️ No need to clone MMRot (e.g. "git clone https://github.com/open-mmlab/mmrotate; rm -rf mmrotate/.git"). Already cloned!
    pip install -v -e mmrotate/
    ~~~

    <details>
      <summary> To verify whether MMRotate is installed correctly, you may try the following things: </summary>
    
    * Download config and checkpoint files.
        ~~~shell
        mim download mmrotate --config oriented_rcnn_r50_fpn_1x_dota_le90 --dest .
        ~~~
    * Verify the inference demo.
        ~~~shell
        python mmrotate/demo/image_demo.py \
        mmrotate/demo/demo.jpg oriented_rcnn_r50_fpn_1x_dota_le90.py \
        oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file result.jpg
        ~~~
    * If **result.jpg** is generated correctly, it means that the environment is set up properly.
    </details>

### Test a model:
You can use the following commands to infer a dataset.
~~~shell
# Single-gpu
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# Multi-gpu
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
~~~

Examples:
<details>
  <summary> Prepare DOTA-v1.0 dataset: </summary>

  * Go to the [official site](https://captain-whu.github.io/DOTA/dataset.html) and download training, validation, test sets (via GoogleDrive). 
  * Create a directory named `data` under **MMDetRotBase2024** and a directory named `DOTA` under **data**.
  * Move `train`, `val`, `test` directories into `data/DOTA`.
  * Run the following code to unzip compressed images and labels in each subdirectory. (Tip: For DOTA-v1.5, unzip **labelTxt-v1.5** instead for all splits!)
    ~~~shell
    # For train,
    cd data/DOTA/train/images
    unzip part1.zip -d temp1; mv temp1/images/* .
    unzip part2.zip -d temp2; mv temp2/images/* .
    unzip part3.zip -d temp3; mv temp3/images/* .
    rm -rf 1 temp1 temp2 temp3 part1.zip part2.zip part3.zip
    cd ..
    mkdir labelTxt
    unzip labelTxt-v1.0/labelTxt.zip -d labelTxt
    cd ../../..
    ~~~
    
    ~~~shell
    # For val,
    cd data/DOTA/val/images
    unzip part1.zip -d temp1; mv temp1/images/* .
    rm -rf temp1 part1.zip
    cd ..
    mkdir labelTxt
    unzip labelTxt-v1.0/labelTxt.zip -d labelTxt
    cd ../../..
    ~~~
    
    ~~~shell
    # For test,
    cd data/DOTA/test
    mkdir images
    unzip part1.zip -d temp1; mv temp1/images/* images/.
    unzip part2.zip -d temp2; mv temp2/images/* images/.
    rm -rf temp1 temp2 part1.zip part2.zip
    cd ../../..
    ~~~

  * Then, run the following codes to crop the images into 1024x1024 patches with an overlap of 200:
    ~~~shell
    pip install shapely
    ~~~

    ~~~shell
    python mmrotate/tools/data/dota/split/img_split.py --base-json \
      mmrotate/tools/data/dota/split/split_configs/ss_trainval.json
    ~~~ 
    ~~~shell
    python mmrotate/tools/data/dota/split/img_split.py --base-json \
      mmrotate/tools/data/dota/split/split_configs/ss_test.json
    ~~~ 
    
    ~~~shell
    mv data/split_ss_dota data/split_1024_dota1_0
    ~~~
    
</details>

* Inference OrientedRCNN on DOTA-v1.0 **test** split (without labels), for [online submission](https://captain-whu.github.io/DOTA/evaluation.html).
    ~~~shell
    python ./mmrotate/tools/test.py  \
      oriented_rcnn_r50_fpn_1x_dota_le90.py \
      oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --format-only \
      --eval-options submission_dir=work_dirs/Task1_results 
    ~~~
  <details>
      <summary> For multi-gpu parallel inference, </summary>
  
  ~~~shell
  mmrotate/tools/dist_test.sh  \
    oriented_rcnn_r50_fpn_1x_dota_le90.py \
    oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth 1 --format-only \
    --eval-options submission_dir=work_dirs/Task1_results
  ~~~
    </details>

  <details>
      <summary> For visualization (inference on images), </summary>
  
  ~~~shell
  python ./mmrotate/tools/test.py \
    oriented_rcnn_r50_fpn_1x_dota_le90.py \
    oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth \
    --show-dir work_dirs/vis
  ~~~
    </details>

* Inference OrientedRCNN on DOTA-v1.0 **validation** split (with given labels), for the offline evaluation.
  * Important: Change the two paths, `ann_file` and `img_prefix` of **data/test** in the config file (.py) for **val** or **trainval** sets.
   ~~~shell
   python ./mmrotate/tools/test.py  \
     oriented_rcnn_r50_fpn_1x_dota_le90.py \
     oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --eval mAP
   ~~~
  <details>
      <summary> For multi-gpu parallel inference, </summary>
  
  ~~~shell
  mmrotate/tools/dist_test.sh  \
    oriented_rcnn_r50_fpn_1x_dota_le90.py \
    oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth 1 --eval mAP
  ~~~
    </details>



### Train a model:
You can use the following commands to train a model from the dataset.
~~~shell
# Single-gpu
python tools/train.py ${CONFIG_FILE} [optional arguments]

# Multi-gpu
./tools/dist_test.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
~~~

**Optional arguments** are:
* `--no-validate` (not recommended): No validation (evaluation) during the training.
* `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
* `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
  * Difference between **resume-from** and **load-from**: resume-from loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally. load-from only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

**Launch multiple jobs on a single machine**: If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs, you need to specify different ports (29500 by default) for each job to avoid communication conflict.
~~~shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
~~~

Examples:
<details>
  <summary> Prepare DOTA-v1.0 dataset: </summary>

  * Go to the [official site](https://captain-whu.github.io/DOTA/dataset.html) and download training, validation, test sets (via GoogleDrive). 
  * Create a directory named `data` under **MMDetRotBase2024** and a directory named `DOTA` under **data**.
  * Move `train`, `val`, `test` directories into `data/DOTA`.
  * Run the following code to unzip compressed images and labels in each subdirectory. (Tip: For DOTA-v1.5, unzip **labelTxt-v1.5** instead for all splits!)
    ~~~shell
    # For train,
    cd data/DOTA/train/images
    unzip part1.zip -d temp1; mv temp1/images/* .
    unzip part2.zip -d temp2; mv temp2/images/* .
    unzip part3.zip -d temp3; mv temp3/images/* .
    rm -rf 1 temp1 temp2 temp3 part1.zip part2.zip part3.zip
    cd ..
    mkdir labelTxt
    unzip labelTxt-v1.0/labelTxt.zip -d labelTxt
    cd ../../..
    ~~~
    
    ~~~shell
    # For val,
    cd data/DOTA/val/images
    unzip part1.zip -d temp1; mv temp1/images/* .
    rm -rf temp1 part1.zip
    cd ..
    mkdir labelTxt
    unzip labelTxt-v1.0/labelTxt.zip -d labelTxt
    cd ../../..
    ~~~
    
    ~~~shell
    # For test,
    cd data/DOTA/test
    mkdir images
    unzip part1.zip -d temp1; mv temp1/images/* images/.
    unzip part2.zip -d temp2; mv temp2/images/* images/.
    rm -rf temp1 temp2 part1.zip part2.zip
    cd ../../..
    ~~~

  * Then, run the following codes to crop the images into 1024x1024 patches with an overlap of 200:
    ~~~shell
    pip install shapely
    ~~~

    ~~~shell
    python mmrotate/tools/data/dota/split/img_split.py --base-json \
      mmrotate/tools/data/dota/split/split_configs/ss_trainval.json
    ~~~ 
    ~~~shell
    python mmrotate/tools/data/dota/split/img_split.py --base-json \
      mmrotate/tools/data/dota/split/split_configs/ss_test.json
    ~~~ 
    
    ~~~shell
    mv data/split_ss_dota data/split_1024_dota1_0
    ~~~
    
</details>


* Train OrientedRCNN on **trainval** or **train** sets of DOTA-v1.0.
   * Simple example 
        ~~~shell
        python ./mmrotate/tools/train.py  \
          oriented_rcnn_r50_fpn_1x_dota_le90_changed.py 
        ~~~
   * More useful example (User-customized configurations)
     * **[NOTE]** When passing configuration arguments in the form of **`key=value`** after --cfg-options, make sure not to include spaces around the equals sign. Formats like `key =value` or `key = value` are not allowed.
     ~~~shell
     python ./mmrotate/tools/train.py  \
       oriented_rcnn_r50_fpn_1x_dota_le90_changed.py \
       --cfg-options optimizer.lr=0.0001 data.samples_per_gpu=2 \
                     log_config.interval=50 
     ~~~
  <details>
      <summary> For multi-gpu parallel training, </summary>
  
  ~~~shell
  mmrotate/tools/dist_train.sh  \
    oriented_rcnn_r50_fpn_1x_dota_le90.py 1 
  ~~~
    </details>
