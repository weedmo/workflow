# ros2_function

> **Migrated from Quartz wiki** (`wiki/study/ros-study/jupyter_ros/jupyter-ros2-function.md`, updated 2026-09-26)  
> tags: ros, ros2, notebook, jupyter_ros, study  
> source: raw/2026-09-26-ros-study/jupyter_ros/ros2_function.ipynb  
> source: https://github.com/weedmo/ROS-study  

다음은 **ROS 2 Humble**에서 자주 사용하는 **Topic, Service, Action** 관련 Python API 함수들을 정리한 노트. 개발할 때 빠르게 참고할 수 있도록 요약.

### Topic 관련 함수
#### 퍼블리셔(Publisher)


```python
publisher = self.create_publisher(MsgType, 'topic_name', qos_profile)
publisher.publish(msg)
```

- `create_publisher(MsgType, topic_name, qos_profile)`: 퍼블리셔 생성
- `publish(msg)`: 메시지 전송

#### 서브스크라이버(Subscriber)


```python
self.subscription = self.create_subscription(
    MsgType,
    'topic_name',
    self.callback,
    qos_profile)
```

- `create_subscription(MsgType, topic_name, callback, qos_profile)`: 서브스크라이버 생성
- `callback(msg)`: 수신 메시지 처리 함수

### Service 관련 함수
#### 클라이언트(Client)


```python
self.cli = self.create_client(SrvType, 'service_name')

# 서비스 요청 준비
req = SrvType.Request()
req.param = value

# 서버가 준비될 때까지 대기
while not self.cli.wait_for_service(timeout_sec=1.0):
    self.get_logger().info('Waiting for service...')

# 요청 보내기
future = self.cli.call_async(req)
future.add_done_callback(response_callback)
```

- `create_client(SrvType, service_name)`: 서비스 클라이언트 생성
- `SrvType.Request()`: 요청 메시지 생성
- `call_async(req)`: 비동기 요청

#### 서버(Server)


```python
self.srv = self.create_service(SrvType, 'service_name', self.callback)
```

- `create_service(SrvType, service_name, callback)`: 서비스 서버 생성
- `callback(request, response)`: 요청 처리 함수
- `response.param = value` → 응답 내용 설정

### Action 관련 함수
#### 액션 클라이언트(Action Client)


```python
self._action_client = ActionClient(self, ActionType, 'action_name')

# 서버 대기
self._action_client.wait_for_server()

# goal 생성
goal_msg = ActionType.Goal()
goal_msg.param = value

# goal 전송
self._send_goal_future = self._action_client.send_goal_async(
    goal_msg,
    feedback_callback=self.feedback_callback)

# 결과 기다리기
self._send_goal_future.add_done_callback(self.goal_response_callback)
```

- `ActionClient(self, ActionType, action_name)`: 액션 클라이언트 생성
- `send_goal_async(goal_msg, feedback_callback)`: 목표 전송
- `goal_response_callback(future)`: 응답 처리
- `feedback_callback(feedback_msg)`: 피드백 수신 처리
- `get_result_async()`: 결과 요청

#### 액션 서버(Action Server)


```python
self._action_server = ActionServer(
    self,
    ActionType,
    'action_name',
    execute_callback=self.execute_callback)
```

- `ActionServer(self, ActionType, action_name, execute_callback)`: 액션 서버 생성
- `execute_callback(goal_handle)`: goal 처리 함수
  - 내부에서 `goal_handle.publish_feedback()`으로 피드백 전송
  - `goal_handle.succeed()` 또는 `goal_handle.abort()`
  - `return result` 로 결과 전달

#### 참고용 예시 메시지 타입
- Topic: `std_msgs.msg.String`, `sensor_msgs.msg.Image`
- Service: `example_interfaces.srv.AddTwoInts`
- Action: `example_interfaces.action.Fibonacci`
