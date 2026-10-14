# triangle_turtle

> **Migrated from Quartz wiki** (`wiki/study/ros-study/jupyter_ros/jupyter-triangle-turtle.md`, updated 2026-09-26)  
> tags: ros, ros2, notebook, jupyter_ros, study  
> source: raw/2026-09-26-ros-study/jupyter_ros/triangle_turtle.ipynb  
> source: https://github.com/weedmo/ROS-study  

```python
import rclpy
from geometry_msgs.msg import Twist
import time
```


```python

if not rclpy.ok():  # 또는 hasattr(rclpy, '_rclpy') 등으로 체크
    rclpy.init()
node = Node('triangle_turtle')
pub = node.create_publisher(Twist, '/turtle1/cmd_vel', 10)
msg = Twist()
```

    [WARN] [1745315101.346933587] [rcl.logging_rosout]: Publisher already registered for provided node name. If this is due to multiple nodes with the same name then all logs for that logger name will go out over the existing publisher. As soon as any node with that name is destructed it will unregister the publisher, preventing any further logs for that name from being published on the rosout topic.



```python
def main():
    rclpy.init()

    # 노드 생성
    node = rclpy.create_node('triangle_drawer')
    publisher = node.create_publisher(Twist, '/turtle1/cmd_vel', 10)
    
    move_cmd = Twist()
    turn_cmd = Twist()
    
    move_cmd.linear.x = 2.0
    turn_cmd.angular.z = 2.0

    # 삼각형 그리기
    for _ in range(3):
        publisher.publish(move_cmd)
        time.sleep(1.5)

        publisher.publish(Twist())  # 정지
        time.sleep(0.5)

        publisher.publish(turn_cmd)
        time.sleep(1.2)

        publisher.publish(Twist())  # 정지
        time.sleep(0.5)

    print("삼각형 그리기 완료")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[66], line 34
         31     rclpy.shutdown()
         33 if __name__ == '__main__':
    ---> 34     main()


    Cell In[66], line 2, in main()
          1 def main():
    ----> 2     rclpy.init()
          4     # 노드 생성
          5     node = rclpy.create_node('triangle_drawer')


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/__init__.py:88, in init(args, context, domain_id, signal_handler_options)
         86     else:
         87         signal_handler_options = SignalHandlerOptions.NO
    ---> 88 context.init(args, domain_id=domain_id)
         89 # Install signal handlers after initializing the context because the rclpy signal
         90 # handler only does something if there is at least one initialized context.
         91 # It is desirable for sigint or sigterm to be able to terminate the process if rcl_init
         92 # takes a long time, and the default signal handlers work well for that purpose.
         93 install_signal_handlers(signal_handler_options)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/context.py:70, in Context.init(self, args, initialize_logging, domain_id)
         65     raise RuntimeError(
         66         'Domain id ({}) should not be lower than zero.'
         67         .format(domain_id))
         69 if self.__context is not None:
    ---> 70     raise RuntimeError('Context.init() must only be called once')
         72 self.__context = _rclpy.Context(
         73     args if args is not None else sys.argv,
         74     domain_id if domain_id is not None else _rclpy.RCL_DEFAULT_DOMAIN_ID)
         75 if initialize_logging and not self._logging_initialized:


    RuntimeError: Context.init() must only be called once



```python
node.create_timer(0.1, timer_callback)
threading.Thread(target=rclpy.spin, args=(node,)).start()
```


```python

```
