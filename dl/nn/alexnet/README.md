

# Alexnet example

This example is extracted from: [kratzert](https://github.com/kratzert/finetune_alexnet_with_tensorflow) (thanks!)


All you need are the pretrained weights, which you can find [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/) or convert yourself from the caffe library using [caffe-to-tensorflow](https://github.com/ethereon/caffe-tensorflow).
If you convert them on your own, take a look on the structure of the `.npy` weights file (dict of dicts or dict of lists).

## TensorBoard support

The code has TensorFlows summaries implemented so that you can follow the training progress in TensorBoard. (--logdir in the config section of `finetune.py`)

## Content

- `alexnet.py`: Class with the graph definition of the AlexNet.
- `finetune.py`: Script to run the finetuning process.
- `datagenerator.py`: Contains a wrapper class for the new input pipeline.
- `caffe_classes.py`: List of the 1000 class names of ImageNet (copied from [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)).
- `validate_alexnet_on_imagenet.ipynb`: Notebook to test the correct implementation of AlexNet and the pretrained weights on some images from the ImageNet database.
- `images/*`: contains three example images, needed for the notebook.

## Usage

All you need to touch is the `finetune.py`, although I strongly recommend to take a look at the entire code of this repository. In the `finetune.py` script you will find a section of configuration settings you have to adapt on your problem.
If you do not want to touch the code any further than necessary you have to provide two `.txt` files to the script (`train.txt` and `val.txt`). Each of them list the complete path to your train/val images together with the class number in the following structure.

```
Example train.txt:
/path/to/train/image1.png 0
/path/to/train/image2.png 1
/path/to/train/image3.png 2
/path/to/train/image4.png 0
.
.
```
were the first column is the path and the second the class label.

The other option is that you bring your own method of loading images and providing batches of images and labels, but then you have to adapt the code on a few lines.
