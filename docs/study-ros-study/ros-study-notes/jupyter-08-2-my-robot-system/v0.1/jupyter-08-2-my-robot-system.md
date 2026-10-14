# 8차시_2_my_robot_system_설명

> **Migrated from Quartz wiki** (`wiki/study/ros-study/jupyter_ros/jupyter-08-2-my-robot-system.md`, updated 2026-09-26)  
> tags: ros, ros2, notebook, jupyter_ros, study  
> source: raw/2026-09-26-ros-study/jupyter_ros/8차시_2_my_robot_system_설명.ipynb  
> source: https://github.com/weedmo/ROS-study  

### 시스템 구성 요약

#### 구성 요소	역할

```bash
Topic (/temperature)	센서 노드에서 주기적으로 온도 publish
Subscriber	온도 데이터를 받아서 30도 이상일 경우 처리 시작
Service (/cooler_motor)	30도 이상일 때 호출되는 쿨러 모터 제어용 서비스
Action (/switch_control)	goal로 스위치 on/off 제어 요청 (예: 일정 시간 동안 ON 등)

전체 구성

[SENSOR_NODE] ----(temperature topic)----> [MANAGER_NODE]
                                             |
                                             +--> call /cooler_motor (service)
                                             |
                                             +--> send_goal /switch_control (action)

```


```bash
my_robot_system/
├── sensor_node.py
├── manager_node.py
├── cooler_service.py
├── switch_action_server.py
├── action/
│   └── SwitchControl.action
```


1. Action 정의: action/SwitchControl.action


```python
# Goal
bool turn_on
---
# Result
bool success
---
# Feedback
string status
```

2. sensor_node.py (토픽 퍼블리셔)


```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import random


class SensorNode(Node):
    def __init__(self):
        super().__init__("sensor_node")
        self.publisher = self.create_publisher(Float32, "temperature", 10)
        self.timer = self.create_timer(1.0, self.publish_temperature)

    def publish_temperature(self):
        temp = random.uniform(25.0, 35.0)
        self.get_logger().info(f"Publishing Temperature: {temp:.2f}")
        msg = Float32()
        msg.data = temp
        self.publisher.publish(msg)


def main():
    rclpy.init()
    sensor_node = SensorNode()
    rclpy.spin(sensor_node)
    rclpy.shutdown()
```

 3. cooler_service.py (서비스 서버)


```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class CoolerService(Node):
    def __init__(self):
        super().__init__("cooler_service")
        self.srv = self.create_service(Trigger, "cooler_motor", self.handle_request)

    def handle_request(self, request, response):
        self.get_logger().info("Cooler activated!")
        response.success = True
        response.message = "Cooler turned on"
        return response


def main():
    rclpy.init()
    cooler_service = CoolerService()
    rclpy.spin(cooler_service)
    rclpy.shutdown()
```

 4. switch_action_server.py (액션 서버)


```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from my_robot_interfaces.action import SwitchControl
import time


class SwitchActionServer(Node):
    def __init__(self):
        super().__init__("switch_action_server")
        self._action_server = ActionServer(
            self, SwitchControl, "switch_control", self.execute_callback
        )

    async def execute_callback(self, goal_handle):
        turn_on = goal_handle.request.turn_on
        status = ["Switch ON"] if turn_on else ["Switch OFF"]
        self.get_logger().info(f"[Action]서버 상태: {status}")

        feedback_msg = SwitchControl.Feedback()
        feedback_msg.status = status
        goal_handle.publish_feedback(feedback_msg)

        time.sleep(2)

        goal_handle.succeed()
        result = SwitchControl.Result()
        result.success = True
        return result


def main():
    rclpy.init()
    switch_action_server = SwitchActionServer()
    rclpy.spin(switch_action_server)
    rclpy.shutdown()

```

 5. manager_node.py (온도 판단 → 서비스/액션 호출)


```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from rclpy.action import ActionClient
from my_robot_interfaces.action import SwitchControl


class ManagerNode(Node):
    def __init__(self):
        super().__init__("manager_node")
        self.subscriber = self.create_subscription(
            Float32, "temperature", self.temp_callback, 10
        )
        self.cooler_client = self.create_client(Trigger, "cooler_motor")
        self.switch_client = ActionClient(self, SwitchControl, "switch_control")

    def temp_callback(self, msg):
        temp = msg.data
        self.get_logger().info(f"Received temperature(현재 보드온도): {temp:.2f}")
        if temp > 30.0:
            self.call_cooler_service()
            self.send_switch_goal(True)

    def call_cooler_service(self):
        while not self.cooler_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for cooler_motor service...")
        req = Trigger.Request()
        future = self.cooler_client.call_async(req)

        def callback(future):
            try:
                res = future.result()
                self.get_logger().info(
                    f"Cooler service called : 팬 동작: {res.success}, 스위치: {res.message}"
                )
            except Exception as e:
                self.get_logger().error(f"Service call failed: {e}")

        future.add_done_callback(callback)

    def send_switch_goal(self, turn_on):
        if not self.switch_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Switch control action server not available!")
            return

        goal_msg = SwitchControl.Goal()
        goal_msg.turn_on = ["Switch ON"]
        self.switch_client.send_goal_async(goal_msg)


def main():
    rclpy.init()
    manager_node = ManagerNode()
    rclpy.spin(manager_node)
    rclpy.shutdown()
```

실행 순서
```bash
ros2 run my_robot_system sensor_node
ros2 run my_robot_system manager_node
ros2 run my_robot_system cooler_service
ros2 run my_robot_system switch_action_server
```

디렉토리 구조 가정

```bash
my_robot_system/
├── launch/
│   └── system_launch.py   ←launch 파일
├── sensor_node.py
├── manager_node.py
├── cooler_service.py
├── switch_action_server.py
```


launch/system_launch.py


```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_system',
            executable='sensor_node',
            name='sensor_node',
            output='screen'
        ),
        Node(
            package='my_robot_system',
            executable='manager_node',
            name='manager_node',
            output='screen'
        ),
        Node(
            package='my_robot_system',
            executable='cooler_service',
            name='cooler_service',
            output='screen'
        ),
        Node(
            package='my_robot_system',
            executable='switch_action_server',
            name='switch_action_server',
            output='screen'
        )
    ])
```

setup.py 설정 확인
entry_points에 각 노드 등록이 되어 있어야 해:


```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 👇 launch 파일 등록
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Sensor to Service and Action example system',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_node = my_robot_system.sensor_node:main',
            'manager_node = my_robot_system.manager_node:main',
            'cooler_service = my_robot_system.cooler_service:main',
            'switch_action_server = my_robot_system.switch_action_server:main',
        ],
    },
)
```

실행 방법


```python
ros2 launch my_robot_system system.launch.py
```

추가로
만약 launch 폴더가 없다면 꼭 CMakeLists.txt 또는 setup.py에 다음 추가해:

setup.py:


```python
data_files=[
    ('share/ament_index/resource_index/packages', ['resource/my_robot_system']),
    ('share/my_robot_system', ['package.xml']),
    ('share/my_robot_system/launch', ['launch/system.launch.py']),
],
```

실행 방법


```python
colcon build --packages-select my_robot_system
source install/setup.bash
ros2 launch my_robot_system system.launch.py
```
