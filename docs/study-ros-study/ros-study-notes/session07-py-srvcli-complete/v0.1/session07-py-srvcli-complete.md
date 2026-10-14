# 7차시_주피터완성본_py_srvcli

> **Migrated from Quartz wiki** (`wiki/study/ros-study/session07/session07-py-srvcli-complete.md`, updated 2026-09-26)  
> tags: ros, ros2, notebook, session07, study  
> source: raw/2026-09-26-ros-study/7차시_수업코드_개선_/7차시 수업코드(개선)/7차시_주피터완성본_py_srvcli.ipynb  
> source: https://github.com/weedmo/ROS-study  

## Service code 를 주피터에서 실행될수 있도록 수정한 코드
실행방법 : 코드를 실행하고 다음 셀에서 정수를 입력하고 실행한다.


```python
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from example_interfaces.srv import AddTwoInts
import threading
import time

# Client Node
class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__("minimal_client_async")
        self.cli = self.create_client(AddTwoInts, "add_two_ints")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting again...")
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        return self.cli.call_async(self.req)

# 전역에서 init()은 한 번만 호출
if not rclpy.ok():
    rclpy.init()

executor = SingleThreadedExecutor()

def run_client(a, b):
    client_node = MinimalClientAsync()
    future = client_node.send_request(a, b)
    executor.add_node(client_node)

    def spin_until_future_complete():
        while rclpy.ok() and not future.done():
            executor.spin_once(timeout_sec=0.1)
            time.sleep(0.1)

    # spin을 백그라운드에서 실행
    spin_thread = threading.Thread(target=spin_until_future_complete)
    spin_thread.start()
    spin_thread.join()

    if future.done():
        response = future.result()
        client_node.get_logger().info(
            f"Result of add_two_ints: for {a} + {b} = {response.sum}"
        )
        print(f"✅ {a} + {b} = {response.sum}")

    executor.remove_node(client_node)
    client_node.destroy_node()

```


```python
run_client(3, 5)
```

    ✅ 3 + 5 = 8


    [WARN] [1745447581.357879883] [rcl.logging_rosout]: Publisher already registered for provided node name. If this is due to multiple nodes with the same name then all logs for that logger name will go out over the existing publisher. As soon as any node with that name is destructed it will unregister the publisher, preventing any further logs for that name from being published on the rosout topic.
    [INFO] [1745447581.414803233] [minimal_service]: Incoming request: a = 3, b = 5
    [INFO] [1745447581.543660360] [minimal_client_async]: Result of add_two_ints: for 3 + 5 = 8



