# 9차시_2_주피터노트북_사용시_주의사항

> **Migrated from Quartz wiki** (`wiki/study/ros-study/jupyter_ros/jupyter-09-2-notebook-cautions.md`, updated 2026-09-26)  
> tags: ros, ros2, notebook, jupyter_ros, study  
> source: raw/2026-09-26-ros-study/jupyter_ros/9차시_2_주피터노트북_사용시_주의사항.ipynb  
> source: https://github.com/weedmo/ROS-study  

## 주피터 노트북으로 ROS2 사용시 주의해야할 사항들 정리
#### rclpy 초기화/종료
ROS 2 Python 노드를 작성할 때


```python
rclpy.init()
node = rclpy.create_node('my_node')
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()
```

#### Jupyter에서는 이렇게 나눠서 실행


```python
import rclpy
from rclpy.node import Node
```


```python
rclpy.init()
```


```python
class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('Node initialized!')

node = MyNode()
```

#### 콜백이 있다면 spin 필요


```python
rclpy.spin(node)
```

#### 종료 시점에 수동으로


```python
node.destroy_node()
rclpy.shutdown()
```

 rclpy.spin()은 블로킹이기 때문에, Jupyter에서는 rclpy.spin_once()를 주기적으로 호출하는 방식이 더 적합할 수 있음 (아래 예시 참고)

그다음 셀에서


```python
import time
for _ in range(10):
    rclpy.spin_once(node)
    time.sleep(1.0)
```

####Publisher/Subscriber 사용

퍼블리셔 예시


```python
from std_msgs.msg import String

publisher = node.create_publisher(String, 'chatter', 10)

msg = String()
msg.data = 'Hello from Jupyter!'
publisher.publish(msg)
```

서브스크라이버 예시


```python
def callback(msg):
    print(f"Received: {msg.data}")

subscriber = node.create_subscription(String, 'chatter', callback, 10)
```

launch나 lifecycle 관련 노드 실행은 피하거나 별도로

Launch file 기반으로 실행되는 복잡한 노드는 Jupyter에서 직접 실행하기 어렵기 때문에, 

ROS Launch로 띄우고 Jupyter에서 통신만 하는 식이 더 적합

### 요약: Jupyter에서 ROS 2 노드 실행 체크리스트

- ROS 환경 활성화	setup.bash source 후 jupyter lab 실행
- rclpy.init()/shutdown() 수동 관리	셀 분리해서 실행
- rclpy.spin() 대신 spin_once() + loop	인터랙티브 실행 가능
- 블로킹 함수 주의	spin()은 막히기 때문에 반복문으로 대체
- 타이머/퍼블리셔/서브스크라이버는 가능	대부분 잘 작동
- launch 파일은 별도로 실행	Jupyter에서는 통신 위주로 활용
