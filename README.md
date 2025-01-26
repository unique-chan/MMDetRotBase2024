<h1 align="center" style="font-weight: 500; line-height: 1.4;">
  Baseline Code for Research on Oriented Object Detection
</h1>

<p align="center">
  <a href="#"><img alt="Python3.8+" src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white"></a>
  <a href="#"><img alt="PyTorch2.0+" src="https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="MMDetection2.28.2" src="https://img.shields.io/badge/MMDetection-2.28.2-red?logo=mmlab&logoColor=white"></a>
  <a href="#"><img alt="MMRotate0.3.4" src="https://img.shields.io/badge/MMRotate-0.3.4-hotpink?logo=mmlab&logoColor=white"></a>
  <a href="#"><img alt="MIT" src="https://img.shields.io/badge/License-MIT-green?logo=MIT"></a>
</p>

<p align="center">
  <b><a href="https://github.com/unique-chan">Yechan Kim</a></b> 
</p>

### This repo includes:
* Training & test code for oriented object detection

### Preliminaries:


* **Step 1**. Create a conda environment with Python 3.8 and activate it.
    ~~~
    conda create --name MMBase python=3.8 -y
    conda activate MMBase
    ~~~

* **Step 2.** Install PyTorch with TorchVision following [official instructions](https://pytorch.org/get-started/locally/). The below is an example.
    ~~~
    pip install torch==2.1.0 torchvision==0.16.0
    ~~~

* **Step 3.** Install `MMDetection (2.28.2)`.
    ~~~
    # ⚠️ Do not need to clone MMDet (e.g. "git clone -b 2.x https://github.com/open-mmlab/mmdetection"). Already cloned! 
    pip install -U openmim==0.3.9
    mim install mmcv-full==1.7.2
    pip install -v -e mmdetection/
    ~~~

* **Step 4.** Install `MMRotate (0.3.4)`. 
    ~~~
    # ⚠️ Do not need to clone MMRot (e.g. "git clone https://github.com/open-mmlab/mmrotate"). Already cloned!
    pip install -v -e mmrotate/
    ~~~

    <details>
      <summary> To verify whether MMRotate is installed correctly, you may try the following things: </summary>
    
    * ~~~
      mim download mmrotate --config oriented_rcnn_r50_fpn_1x_dota_le90 --dest .
      ~~~
    * ~~~
      python mmrotate/demo/image_demo.py mmrotate/demo/demo.jpg oriented_rcnn_r50_fpn_1x_dota_le90.py oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file result.jpg
      ~~~
    If **result.jpg** is generated correctly, it means that the environment is set up properly.
    </details>

