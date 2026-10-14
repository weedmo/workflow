# SubscriptionTest

> **Migrated from Quartz wiki** (`wiki/study/ros-study/jupyter_ros/jupyter-subscription-test.md`, updated 2026-09-26)  
> tags: ros, ros2, notebook, jupyter_ros, study  
> source: raw/2026-09-26-ros-study/jupyter_ros/SubscriptionTest.ipynb  
> source: https://github.com/weedmo/ROS-study  

```python
import rclpy # Ros2를 python에서 사용할 수 있게 해주는 module
from turtlesim.msg import Pose
```


```python
if not rclpy.ok():  # 또는 hasattr(rclpy, '_rclpy') 등으로 체크
    rclpy.init()
test_node = rclpy.create_node('sub_test')
```

    [WARN] [1744163682.294190713] [rcl.logging_rosout]: Publisher already registered for provided node name. If this is due to multiple nodes with the same name then all logs for that logger name will go out over the existing publisher. As soon as any node with that name is destructed it will unregister the publisher, preventing any further logs for that name from being published on the rosout topic.



```python
def callback1(data):
    print('---')
    print('/turtle1/pose :',data)
    print('x : ', data.x)
    print('y : ', data.y)
    print('theta : ', data.theta)
```


```python
cnt = 0
def callback2(data):
    global cnt
    cnt += 1
    print('>', cnt, '-> X :' , data.x, ',Y : ', data.y)
    if cnt > 3:
        raise Exception('subscription Stop')
```


```python
test_node.create_subscription(Pose, '/turtle1/pose',callback1,10)
```




    <rclpy.subscription.Subscription at 0x755184771b70>




```python
# node 연결
# <data_type> <topic_name> <callback> <QoS History>
test_node.create_subscription(Pose, '/turtle1/pose',callback2,10)
```




    <rclpy.subscription.Subscription at 0x7551847f0370>




```python
rclpy.spin_once(test_node)
```

    > 6 -> X : 5.544444561004639 ,Y :  5.544444561004639



    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    Cell In[27], line 1
    ----> 1 rclpy.spin_once(test_node)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/__init__.py:206, in spin_once(node, executor, timeout_sec)
        204 try:
        205     executor.add_node(node)
    --> 206     executor.spin_once(timeout_sec=timeout_sec)
        207 finally:
        208     executor.remove_node(node)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py:751, in SingleThreadedExecutor.spin_once(self, timeout_sec)
        750 def spin_once(self, timeout_sec: float = None) -> None:
    --> 751     self._spin_once_impl(timeout_sec)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py:748, in SingleThreadedExecutor._spin_once_impl(self, timeout_sec)
        746 handler()
        747 if handler.exception() is not None:
    --> 748     raise handler.exception()


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/task.py:254, in Task.__call__(self)
        251 if inspect.iscoroutine(self._handler):
        252     # Execute a coroutine
        253     try:
    --> 254         self._handler.send(None)
        255     except StopIteration as e:
        256         # The coroutine finished; store the result
        257         self.set_result(e.value)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py:447, in Executor._make_handler.<locals>.handler(entity, gc, is_shutdown, work_tracker)
        444 gc.trigger()
        446 try:
    --> 447     await call_coroutine(entity, arg)
        448 finally:
        449     entity.callback_group.ending_execution(entity)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py:372, in Executor._execute_subscription(self, sub, msg)
        370 async def _execute_subscription(self, sub, msg):
        371     if msg:
    --> 372         await await_or_execute(sub.callback, msg)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py:107, in await_or_execute(callback, *args)
        104     return await callback(*args)
        105 else:
        106     # Call a normal function
    --> 107     return callback(*args)


    Cell In[18], line 7, in callback2(data)
          5 print('>', cnt, '-> X :' , data.x, ',Y : ', data.y)
          6 if cnt > 3:
    ----> 7     raise Exception('subscription Stop')


    Exception: subscription Stop



```python
# 노드 구독 once는 한번만 그냥 spin은 무한 반복
rclpy.spin(test_node)
```

    > 1 -> X : 5.544444561004639 ,Y :  5.544444561004639
    > 2 -> X : 5.544444561004639 ,Y :  5.544444561004639
    > 3 -> X : 5.544444561004639 ,Y :  5.544444561004639
    > 4 -> X : 5.544444561004639 ,Y :  5.544444561004639



    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    Cell In[20], line 2
          1 # 노드 구독 once는 한번만 그냥 spin은 무한 반복
    ----> 2 rclpy.spin(test_node)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/__init__.py:226, in spin(node, executor)
        224     executor.add_node(node)
        225     while executor.context.ok():
    --> 226         executor.spin_once()
        227 finally:
        228     executor.remove_node(node)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py:751, in SingleThreadedExecutor.spin_once(self, timeout_sec)
        750 def spin_once(self, timeout_sec: float = None) -> None:
    --> 751     self._spin_once_impl(timeout_sec)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py:748, in SingleThreadedExecutor._spin_once_impl(self, timeout_sec)
        746 handler()
        747 if handler.exception() is not None:
    --> 748     raise handler.exception()


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/task.py:254, in Task.__call__(self)
        251 if inspect.iscoroutine(self._handler):
        252     # Execute a coroutine
        253     try:
    --> 254         self._handler.send(None)
        255     except StopIteration as e:
        256         # The coroutine finished; store the result
        257         self.set_result(e.value)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py:447, in Executor._make_handler.<locals>.handler(entity, gc, is_shutdown, work_tracker)
        444 gc.trigger()
        446 try:
    --> 447     await call_coroutine(entity, arg)
        448 finally:
        449     entity.callback_group.ending_execution(entity)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py:372, in Executor._execute_subscription(self, sub, msg)
        370 async def _execute_subscription(self, sub, msg):
        371     if msg:
    --> 372         await await_or_execute(sub.callback, msg)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py:107, in await_or_execute(callback, *args)
        104     return await callback(*args)
        105 else:
        106     # Call a normal function
    --> 107     return callback(*args)


    Cell In[18], line 7, in callback2(data)
          5 print('>', cnt, '-> X :' , data.x, ',Y : ', data.y)
          6 if cnt > 3:
    ----> 7     raise Exception('subscription Stop')


    Exception: subscription Stop



```python

```
