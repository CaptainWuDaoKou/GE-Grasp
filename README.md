# GE-Grasp: Efficeint Target Oriented Grasping in Dense Clutters

This repository is for the paper

**[GE-Grasp: Efficeint Target Oriented Grasping in Dense Clutters]**<br/>


Grasping in dense clutters is a fundamental skill for autonomous robots. However, the crowdedness and occlusions in the cluttered scenario cause significant difficulties to generate valid grasp poses without collisions, which results in low efficiency and high failure rates. To address these, we present a generic framework called GE-Grasp for grasp pose generation in dense clutters, where we leverage diverse action primitives for occluded object removal and present the generator-evaluator architecture to avoid spatial collisions. Therefore, our GE-Grasp is capable of grasping objects in dense clutters efficiently with promising success rates. Specifically, we define three action primitives: target-oriented grasping for picking up the target directly, pushing and nontarget-oriented grasping to reduce the crowdedness and occlusions. The generators select the preferred action primitive set via a spatial correlation test (SCT), which effectively provide various motion candidates aiming at grasping target objects in clutters. Meanwhile, the evaluators assess the selected action primitive candidates, where optimal action is implemented by the robot. 

<p align="center">
    <img src="./images/overview.png" width=100% alt="overview">
    <i>System Overview</i>  
</p>


## Dependencies
```
- Ubuntu 16.04
- Python 3
- PyTorch 1.6
```
The file of the conda environment is environment.yml. We use [V-REP 3.5.0] as the simulation environment.

## Code
We do experiments on a NVIDIA 1080 Ti GPU. It requires at least 6GB of GPU memory to run the code.

First download the pre-trained models and the segmentation module from "https://drive.google.com/file/d/1KSVV-dduiYWG1K4FxWOP5XnyIbNvzx45/view?usp=sharing", and unzip the two folders to the root directory.

Then download V-REP 3.5.0 from "https://drive.google.com/file/d/1nDkkNO4FxpNSl6eB3sm0-r_BfKqMaSgS/view?usp=sharing" and open the file ```simulation/simulation.ttt``` with V-REP to start the simulation.


### Testing

To test the "random clutters" task, run 

```
python test.py
```

The files of the test cases are available in ```simulation/random```.

To test the "challenging clutters" task, run
```
python test.py --test_preset_cases 
```
The files of the test cases are available in ```simulation/preset```.



## Acknowledgments
We use the following code in our project

- [Visual Pushing and Grasping Toolbox][1]

- [A Deep Learning Method to Grasp the Invisible][2]

- [Light-Weight RefineNet for Real-Time Semantic Segmentation][3]


[1]: https://github.com/andyzeng/visual-pushing-grasping
[2]: https://github.com/choicelab/grasping-invisible
[3]: https://github.com/DrSleep/light-weight-refinenet
