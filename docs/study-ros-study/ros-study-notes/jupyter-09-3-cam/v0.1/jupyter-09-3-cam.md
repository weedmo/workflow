# 9차시_3_cam_실행

> **Migrated from Quartz wiki** (`wiki/study/ros-study/jupyter_ros/jupyter-09-3-cam.md`, updated 2026-09-26)  
> tags: ros, ros2, notebook, jupyter_ros, study  
> source: raw/2026-09-26-ros-study/jupyter_ros/9차시_3_cam_실행.ipynb  
> source: https://github.com/weedmo/ROS-study  

ROS 2 Humble - 카메라 이미지 퍼블리셔 & 서브스크라이버

사용 메시지 타입: sensor_msgs/msg/Image

이미지 처리 라이브러리: cv_bridge, OpenCV

카메라 데이터: 직접 캡처(OpenCV) 또는 실제 카메라 노드 사용 가능

ros2 pkg create --build-type ament_python my_cam_pubsub --dependencies rclpy sensor_msgs cv_bridge

1. 퍼블리셔 노드 (카메라 데이터 publish)

2. 서브스크라이버 노드 (카메라 데이터 수신)

설치 필요 패키지

```bash
sudo apt install ros-humble-cv-bridge python3-opencv
```

setup.py 수정

console_script 에 아래내용 추가
```bash
'cam_pub = my_cam_pubsub.cam_pub:main',
'cam_sub = my_cam_pubsub.cam_sub:main',
```

#### colcon build 실행

```bash
colcon build --symlink-install --packages-select my_cam_pubsub 

source install/setup.bash
```

#### 실행 명령어(터미널 3개 열고실행)

```bash
ros2 run my_cam_pubsub cam_pub

다른 Terminal 에서

ros2 run my_cam_subsub cam_sub

rqt 에서 Image 출력력

rviz 에서 Image 출력
```

### 테스트 순서
- 퍼블리셔 노드 실행 (카메라 영상 송출)
- 서브스크라이버 노드 실행 (영상 수신 및 출력)
- rqt 실행
- image view 시청
