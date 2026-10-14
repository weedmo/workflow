# 8차시_1_temperature_example

> **Migrated from Quartz wiki** (`wiki/study/ros-study/jupyter_ros/jupyter-08-1-temperature.md`, updated 2026-09-26)  
> tags: ros, ros2, notebook, jupyter_ros, study  
> source: raw/2026-09-26-ros-study/jupyter_ros/8차시_1_temperature_example.ipynb  
> source: https://github.com/weedmo/ROS-study  

### ROS 2 Temperature Publisher/Subscriber Example in Jupyter
가상의 온도 센서값을 1초 간격으로 퍼블리시하고, 동시에 이를 서브스크라이브하는 구조입니다.

#### 환경 설정

셀 1: ROS 2 환경 준비

먼저 터미널에서 ROS 2 환경을 source 한 다음 Jupyter를 실행해야 합니다:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

셀 2: 기본 임포트 및 초기화


```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import random
import time
```

셀 3: 노드 정의


```python
class TemperatureNode(Node):
    def __init__(self):
        super().__init__('temperature_node')

        self.publisher = self.create_publisher(Float32, 'temperature', 10)
        self.subscription = self.create_subscription(Float32, 'temperature', self.listener_callback, 10)
        self.timer = self.create_timer(1.0, self.publish_temperature)
        self.temperature_value = 25.0  # 초기 온도

    def publish_temperature(self):
        self.temperature_value += random.uniform(-0.5, 0.5)
        msg = Float32()
        msg.data = self.temperature_value
        self.publisher.publish(msg)
        print(f'Published temperature: {msg.data:.2f} °C')

    def listener_callback(self, msg):
        print(f'Received temperature: {msg.data:.2f} °C')
```

셀 4: 노드 실행 및 ROS 초기화


```python
rclpy.init()
node = TemperatureNode()
```

 셀 5: 스핀 루프 (1초마다 1회 실행, 30초 동안)


```python
try:
    for _ in range(30):  # 총 30초 동안 동작
        rclpy.spin_once(node)
        time.sleep(1.0)
finally:
    node.destroy_node()
    rclpy.shutdown()
```

#### 결과
Jupyter에서 위 셀들을 순서대로 실행하면,
temperature 토픽으로 온도 데이터를 퍼블리시하고
같은 토픽을 서브스크라이브하여 출력할 수 있습니다.

Tip: Jupyter에서 log는 node.get_logger().info()로는 출력이 잘 안 보일 수 있으니 print()로 바꿔도 괜찮아요.
