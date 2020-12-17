# Foot-PoseNet

This is a Keras implementation of a modified version of [PersonLab](https://arxiv.org/abs/1803.08225) called Foot-PoseNet for Multi-Person Pose Estimation.
The model predicts heatmaps and offsets which allow for computation of 23 joints also known as keypoints, 17 for face and body, and 6 for feet. See the paper for more details.

### Demo

* Download the [model](https://drive.google.com/file/d/1viDeWyRVNwAV6uEw4OcRYrssiSIqSR_a/view?usp=sharing)
* Run 'python demo.py' to run the demo and visualize the model results

### Result

**Pose**

![pose](https://github.com/BrunoMelicio/FootPoseNet/blob/main/src/demo_results/keypoints_test.png)

## Training a model
If you want to use Resnet101 as the base, first download the imagenet initialization weights from [here](https://drive.google.com/open?id=1ulygah5BTWjhSGGpN20-eYV5NAozdE8Z) and copy it to your `~/.keras/models/` directory.

First, construct the dataset in the correct format by running the [generate_hdf5.py](generate_hdf5.py) script. Before running, just set the `ANNO_FILE` and `IMG_DIR` constants at the top of the script to the paths to the COCO person_keypoints annotation file and the image folder respectively.

Edit the [config.py](config.py) to set options for training, e.g. input resolution, number of GPUs, whether to freeze the batchnorm weights, etc. More advanced options require altering the [train.py](train.py) script. For example, changing the base network can be done by adding an argument to the get_personlab() function, see the documentation [there](model.py#L162).

After eveything is configured to your liking, go ahead and run the train.py script.

## Testing a model

See the [demo.ipynb](demo.ipynb) for sample inference and visualizations.

## Technical Debts
Several parts of this codebase are borrowed from [PersonLab Keras](https://github.com/octiapp/KerasPersonLab)

## Environment
This code was tested in the following environment and with the following software versions:

* Ubuntu 16.04
* CUDA 8.0 with cudNN 6.0
* Python 2.7
* Tensorflow 1.7
* Keras 2.1.3
* OpenCV 2.4.9
* Tensorflow 1.80
* pycocotools  2.0
* skimage  0.13.0
* python-opencv 3.4.1





### Training

* Download the COCO 2017 dataset

  http://images.cocodataset.org/zips/train2017.zip

  http://images.cocodataset.org/zips/val2017.zip

  http://images.cocodataset.org/annotations/annotations_trainval2017.zip

  training images in `coco2017/train2017/` , val images in `coco2017/val2017/`, training annotations in `coco2017/annotations/`

* Download the [Resnet101](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz) pretrained model, put the model in `./model/101/resnet_v2_101.ckpt`

* Edit the [config.py](https://github.com/scnuhealthy/Tensorflow_PersonLab/blob/master/config.py) to set options for training, e.g. dataset position, input tensor shape, learning rate. 
* Run the train.py script





### Citation

```
@inproceedings{papandreou2018personlab,
  title={PersonLab: Person pose estimation and instance segmentation with a bottom-up, part-based, geometric embedding model},
  author={Papandreou, George and Zhu, Tyler and Chen, Liang-Chieh and Gidaris, Spyros and Tompson, Jonathan and Murphy, Kevin},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={269--286},
  year={2018}
}
```
