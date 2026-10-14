# PublisherTest

> **Migrated from Quartz wiki** (`wiki/study/ros-study/jupyter_ros/jupyter-publisher-test.md`, updated 2026-09-26)  
> tags: ros, ros2, notebook, jupyter_ros, study  
> source: raw/2026-09-26-ros-study/jupyter_ros/PublisherTest.ipynb  
> source: https://github.com/weedmo/ROS-study  

```python
import rclpy 
from geometry_msgs.msg import Twist

if not rclpy.ok():  # 또는 hasattr(rclpy, '_rclpy') 등으로 체크
    rclpy.init()
test_node = rclpy.create_node('pub_test')
```

    1745315536.778231 [25]    python3: selected interface "lo" is not multicast-capable: disabling multicast



```python
msg = Twist()
print(msg)
```

    geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0), angular=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0))



```python
msg.linear.x = 0.0
print(msg)
```

    geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0), angular=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0))



```python
pub = test_node.create_publisher(Twist, '/turtle1/cmd_vel', 10)
```


```python
msg.linear.x = 2.0
msg.angular.z = 2.0
pub.publish(msg)

```


```python
cnt = 0
def timer_callback():
    global cnt

    cnt += 1

    print(cnt)
    pub.publish(msg)

    if cnt > 5:
        raise Exception('publisher stop')
```


```python
timer_period = 0.1
timer = test_node.create_timer(0.1, timer_callback)
rclpy.spin(test_node)
```

    1
    2
    3
    4
    5
    6



    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    Cell In[7], line 3
          1 timer_period = 0.1
          2 timer = test_node.create_timer(0.1, timer_callback)
    ----> 3 rclpy.spin(test_node)


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


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py:361, in Executor._execute_timer(self, tmr, _)
        360 async def _execute_timer(self, tmr, _):
    --> 361     await await_or_execute(tmr.callback)


    File /opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py:107, in await_or_execute(callback, *args)
        104     return await callback(*args)
        105 else:
        106     # Call a normal function
    --> 107     return callback(*args)


    Cell In[6], line 11, in timer_callback()
          8 pub.publish(msg)
         10 if cnt > 5:
    ---> 11     raise Exception('publisher stop')


    Exception: publisher stop



```python
cnt = 0
phase = 0  # 짝수: 직진 / 홀수: 회전

def timer_callback():
    global cnt, phase

    print(f"[{cnt}] Phase: {phase}")

    if phase % 2 == 0:
        # 직진 단계
        msg.linear.x = 2.0
        msg.angular.z = 0.0
    else:
        # 회전 단계 (120도 회전 → 약 1.5초 동안 angular.z = 2.0)
        msg.linear.x = 0.0
        msg.angular.z = 1.5

    pub.publish(msg)
    cnt += 1

    # 각 단계의 지속 시간 설정 (0.1초 타이머 기준)
    if (phase % 2 == 0 and cnt >= 20):      # 직진 2초
        cnt = 0
        phase += 1
    elif (phase % 2 == 1 and cnt >= 15):    # 회전 1.5초
        cnt = 0
        phase += 1

    if phase >= 6:
        print("삼각형 그리기 완료 - shutdown")
        rclpy.shutdown()
        
timer = test_node.create_timer(0.1, timer_callback)
rclpy.spin(test_node)
```

    [0] Phase: 0
    [1] Phase: 0
    [2] Phase: 0
    [3] Phase: 0
    [4] Phase: 0
    [5] Phase: 0
    [6] Phase: 0
    [7] Phase: 0
    [8] Phase: 0
    [9] Phase: 0
    [10] Phase: 0
    [11] Phase: 0
    [12] Phase: 0
    [13] Phase: 0
    [14] Phase: 0
    [15] Phase: 0
    [16] Phase: 0
    [17] Phase: 0
    [18] Phase: 0
    [19] Phase: 0
    [0] Phase: 1
    [1] Phase: 1
    [2] Phase: 1
    [3] Phase: 1
    [4] Phase: 1
    [5] Phase: 1
    [6] Phase: 1
    [7] Phase: 1
    [8] Phase: 1
    [9] Phase: 1
    [10] Phase: 1
    [11] Phase: 1
    [12] Phase: 1
    [13] Phase: 1
    [14] Phase: 1
    [0] Phase: 2
    [1] Phase: 2
    [2] Phase: 2
    [3] Phase: 2
    [4] Phase: 2
    [5] Phase: 2
    [6] Phase: 2
    [7] Phase: 2
    [8] Phase: 2
    [9] Phase: 2
    [10] Phase: 2
    [11] Phase: 2
    [12] Phase: 2
    [13] Phase: 2
    [14] Phase: 2
    [15] Phase: 2
    [16] Phase: 2
    [17] Phase: 2
    [18] Phase: 2
    [19] Phase: 2
    [0] Phase: 3
    [1] Phase: 3
    [2] Phase: 3
    [3] Phase: 3
    [4] Phase: 3
    [5] Phase: 3
    [6] Phase: 3
    [7] Phase: 3
    [8] Phase: 3
    [9] Phase: 3
    [10] Phase: 3
    [11] Phase: 3
    [12] Phase: 3
    [13] Phase: 3
    [14] Phase: 3
    [0] Phase: 4
    [1] Phase: 4
    [2] Phase: 4
    [3] Phase: 4
    [4] Phase: 4
    [5] Phase: 4
    [6] Phase: 4
    [7] Phase: 4
    [8] Phase: 4
    [9] Phase: 4
    [10] Phase: 4
    [11] Phase: 4
    [12] Phase: 4
    [13] Phase: 4
    [14] Phase: 4
    [15] Phase: 4
    [16] Phase: 4
    [17] Phase: 4
    [18] Phase: 4
    [19] Phase: 4
    [0] Phase: 5
    [1] Phase: 5
    [2] Phase: 5
    [3] Phase: 5
    [4] Phase: 5
    [5] Phase: 5
    [6] Phase: 5
    [7] Phase: 5
    [8] Phase: 5
    [9] Phase: 5
    [10] Phase: 5
    [11] Phase: 5
    [12] Phase: 5
    [13] Phase: 5
    [14] Phase: 5
    삼각형 그리기 완료 - shutdown



```python

```


```python

```
