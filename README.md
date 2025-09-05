[English](./README.md) | 简体中文

# Function Introduction

The **hand_landmarks_mediapipe** package is an example of a monocular RGB hand landmark detection algorithm developed using the **hobot_dnn** package.  
On the RDK series development boards, the model and image data are processed by the BPU processor for inference.  
The model outputs the detected hand landmark results.

The example subscribes to image data (`image msg`), publishes customized perception results (`hobot ai msg`), and users can subscribe to the published `ai msg` for application development.

# Model and Platform Support

| Model Type           | Supported Platform |
| :------------------- | ------------------ |
| mediapipe            | RDK S100           |

# Bill of Materials

| Item Name            | Manufacturer | Reference Link                                                                 |
| :------------------- | ------------ | ------------------------------------------------------------------------------ |
| RDK S100             | Multiple     | [RDK S100](https://developer.horizon.cc/rdks100)                               |
| Camera               | Multiple     | [MIPI Camera](https://developer.horizon.cc/nodehubdetail/168958376283445781)<br>[USB Camera](https://developer.horizon.cc/nodehubdetail/168958376283445777) |

# Preparation

- RDK has been flashed with Ubuntu 22.04 system image  
- The camera is properly connected to the RDK S100

# Usage

**1. Install the package**

After starting the robot, connect to it via SSH terminal or VNC, click the "One-click Deploy" button in the upper right corner of this page, then copy the following command to run on the RDK system to complete the installation of the relevant Node.

```bash
sudo apt update
sudo apt install -y tros-humble-hand_landmarks_mediapipe
```

**2.Run mediapipe hand landmark detection**

**Using MIPI Camera to publish images**

```shell
# Configure tros.b humble environment
source /opt/tros/humble/setup.bash

# Copy the configuration files required to run the example from the tros.b installation path.
cp -r /opt/tros/${TROS_DISTRO}/lib/palm_detection_mediapipe/config/ .
cp -r /opt/tros/${TROS_DISTRO}/lib/hand_landmarks_mediapipe/config/ .

# Configure MIPI camera
export CAM_TYPE=mipi

# Start the launch file
ros2 launch hand_landmarks_mediapipe hand_landmarks.launch.py

```

**Using USB Camera to publish images**

```shell
# Configure tros.b humble environment
source /opt/tros/humble/setup.bash

# Copy the configuration files required to run the example from the tros.b installation path.
cp -r /opt/tros/${TROS_DISTRO}/lib/palm_detection_mediapipe/config/ .
cp -r /opt/tros/${TROS_DISTRO}/lib/hand_landmarks_mediapipe/config/ .

# Configure USB camera
export CAM_TYPE=usb

# Start the launch file
ros2 launch hand_landmarks_mediapipe hand_landmarks.launch.py
```

**Using local playback images**

Only supported on tros humble version.

```shell
# Copy the configuration files required to run the example from the tros.b installation path.
cp -r /opt/tros/${TROS_DISTRO}/lib/palm_detection_mediapipe/config/ .
cp -r /opt/tros/${TROS_DISTRO}/lib/hand_landmarks_mediapipe/config/ .

# Configure local playback image
export CAM_TYPE=fb

# Start the launch file
ros2 launch hand_landmarks_mediapipe hand_landmarks.launch.py publish_image_source:=config/example.jpg publish_image_format:=jpg publish_output_image_w:=640 publish_output_image_h:=480

```

**4.View the results**

On a computer in the same network, open the browser and visit http://IP:8000
 to see the real-time visual recognition results.
Here, IP is the RDK's IP address.

# Interface Description

## Topics

The hand landmark detection results are published through the topic hobot_msgs/ai_msgs/msg/PerceptionTargets
.
The detailed definition of this topic is as follows:
```shell
# Perception result

# Message header
std_msgs/Header header

# Processing frame rate of perception results
# fps val is invalid if fps is less than 0
int16 fps

# Performance statistics, such as recording the inference time of each model
Perf[] perfs

# Collection of perception targets
Target[] targets

```

| Name                        | Message Type                                                                                                                          | Description                                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| /hand\_landmarks\_mediapipe | [hobot\_msgs/ai\_msgs/msg/PerceptionTargets](https://github.com/D-Robotics/hobot_msgs/blob/develop/ai_msgs/msg/PerceptionTargets.msg) | Publishes detected target information                                                                        |
| /hbmem\_img                 | [hobot\_msgs/hbm\_img\_msgs/msg/HbmMsg1080P](https://github.com/D-Robotics/hobot_msgs/blob/develop/hbm_img_msgs/msg/HbmMsg1080P.msg)  | When `is_shared_mem_sub == 1`, subscribes to image data from the previous node via shared memory             |
| /image\_raw                 | hsensor\_msgs/msg/Image                                                                                                               | When `is_shared_mem_sub == 0`, subscribes to image data from the previous node via standard ROS subscription |

## Parameters


| Parameter Name            | Type        | Description                                                                                                                                                 | Required | Supported Configurations          | Default Value               |
| ------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------------------------------- | --------------------------- |
| is\_sync\_mode            | int         | Synchronous/Asynchronous inference mode. 0: asynchronous; 1: synchronous                                                                                    | No       | 0/1                               | 0                           |
| model\_file\_name         | std::string | Model file used for inference                                                                                                                               | No       | Configured with actual model path | config/hand\_224\_224.hbm   |
| is\_shared\_mem\_sub      | int         | Whether to subscribe to image messages via shared memory. 0: off; 1: on. When enabled/disabled, topic names are `/hbmem_img` and `/image_raw` respectively. | No       | 0/1                               | 1                           |
| ai\_msg\_pub\_topic\_name | std::string | Topic name for publishing AI messages containing hand landmark perception results                                                                           | No       | Configured per deployment         | /hand\_landmarks\_mediapipe |
| ros\_img\_topic\_name     | std::string | ROS image topic name                                                                                                                                        | No       | Configured per deployment         | /image\_raw                 |
| image\_gap                | int         | Frame skipping interval, indicating the frequency of image processing. 1 = every frame, 2 = every other frame, etc.                                         | No       | Configured per deployment         | 1                           |
