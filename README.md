# uchile_vision

Vision features for robots at Universidad de Chile Homebreakers Robotic Team. Many thanks to [JP][Jpcaceres] for making this repository possible.

This package has a class dedicated to making detections with Yolo v5 (other utilities are work in progress). It also includes a custom type of message made for easier communication with other ROS nodes and topics. For more info on how the message type works refer to the [darknet official repository][darknet].

## Setup uchile_vision repository

`uchile_vision` repository needs to be in `soft_ws` workspace of Uchile Robotics Workspace.

In `soft_ws` directory open a terminal:

```
cd src/
git clone https://github.com/uchile-robotics/uchile_vision
```

Once installed, we need to run the next set of bash commands:

```
cd yolov5/
sudo python setup.py install
cd src/yolov5/
python2.7 -m pip --no-cache-dir install -r requirements.txt
python2.7 -m pip install future
```
> Note: It's important to run the pip commands exactly how they are exposed, because there is a known bug where installing `pytorch 1.4` without `--no-cache-dir` crashes the computer.

With this done, Yolo v5 should work with ROS Melodic and Python 2.7 (Don't forget to compile every workspace!)

# How do I use Yolo v5 in my robot?

In `/uchile_vision/yolov5/src/yolov5/detect.py` is where the `Yolov5()` class is stated. For using it in a script it has to be imported.

In the same file we can choose between setting `self.device` to either `"cpu"` or `"cuda"`. As you can already be thinking, this variable sets the GPU with cuda or CPU functionality of the neural network. 

Last but not least, when creating the `Yolov5()` whe can set the `weights` variable that we desire. From default, the neural network uses `ycb_v8.pt` weights that where trained in Gazebo using YCB dataset. In this same package exists another weight file called `yolov5_jp.pt`. This weights are the default weights of Yolo v5 neural network, which where trained with COCO dataset.

# Solving common issues
When using `yolov5_jp.pt` we can get an error where a so called SiLU function does not exist. For solving it open a terminal and execute the following commands:

```
cd 
cd .local/lib/python2.7/site_packages/torch/nn/modules
```

Here edit `activation.py` file and add the following lines at the end of the file:

```
class SiLU(Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
```

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [darknet]: <https://github.com/leggedrobotics/darknet_ros/tree/master/darknet_ros_msgs>
   [Jpcaceres]: <https://github.com/Jpcaceres>     
