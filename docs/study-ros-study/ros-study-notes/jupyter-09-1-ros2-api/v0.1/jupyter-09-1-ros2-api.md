# 9차시_1_ROS2_API함수정리

> **Migrated from Quartz wiki** (`wiki/study/ros-study/jupyter_ros/jupyter-09-1-ros2-api.md`, updated 2026-09-26)  
> tags: ros, ros2, notebook, jupyter_ros, study  
> source: raw/2026-09-26-ros-study/jupyter_ros/9차시_1_ROS2_API함수정리.ipynb  
> source: https://github.com/weedmo/ROS-study  

**ROS 2 Humble**에서 자주 사용하는 **Topic, Service, Action** 관련 Python API 함수들을 정리한 노트.

개발할 때 빠르게 참고할 수 있도록 요약.

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
self.cli = self.create_client(SrvType, "service_name")

# 서비스 요청 준비
req = SrvType.Request()
req.param = value

# 서버가 준비될 때까지 대기
while not self.cli.wait_for_service(timeout_sec=1.0):
    self.get_logger().info("Waiting for service...")

# 요청 보내기
future = self.cli.call_async(req)
future.add_done_callback(response_callback)
```

- `create_client(SrvType, service_name)`: 서비스 클라이언트 생성
- `SrvType.Request()`: 요청 메시지 생성
- `call_async(req)`: 비동기 요청

- cli.call_async()는 ROS 2에서 서비스 클라이언트가 비동기적으로 서비스 요청을 보낼 때 사용하는 메서드입니다.
- cli.call_async(req)는 서비스 서버에 요청을 보내고, 결과를 기다리는 Future 객체를 반환합니다.
- 결과는 나중에 add_done_callback()이나 spin_until_future_complete()로 받습니다.

#### Service Request(서비스 요청)

서비스 클라이언트 + add_done_callback()


```python
future = cli.call_async(req)
future.add_done_callback(response_callback)
```

핵심 요약
```bash
항목	         설명
call_async(req)	비동기 서비스 호출, Future 반환
반환값	Future 객체 (future.result()로 결과 얻음)
처리 방법	rclpy.spin_once() 루프 안에서 future.done() 체크
```

#### Service 서버(Server)


```python
self.srv = self.create_service(SrvType, 'service_name', self.callback)
```

- `create_service(SrvType, service_name, callback)`: 서비스 서버 생성
- `callback(request, response)`: 요청 처리 함수
- `response.param = value` → 응답 내용 설정

| 역할       | 서버                                | 클라이언트                          |
|------------|-------------------------------------|-------------------------------------|
| 생성       | `create_service()`                  | `create_client()`                  |
| 요청 핸들링 | `callback(request, response)`       | `req = Request(); call_async(req)` |
| 응답 전송   | `return response`                   | `future.result()`로 결과 받음       |
| 필드 설정   | `response.param = value`            | `req.param = value`                |


### Action 관련 함수
#### 액션 클라이언트(Action Client)


```python
self._action_client = ActionClient(self, ActionType, "action_name")

# 서버 대기
self._action_client.wait_for_server()

# goal 생성
goal_msg = ActionType.Goal()
goal_msg.param = value

# goal 전송
self._send_goal_future = self._action_client.send_goal_async(
    goal_msg, feedback_callback=self.feedback_callback
)

# 결과 기다리기
self._send_goal_future.add_done_callback(self.goal_response_callback)
```

- `ActionClient(self, ActionType, action_name)`: 액션 클라이언트 생성
- `send_goal_async(goal_msg, feedback_callback)`: 목표 전송
- `goal_response_callback(future)`: 응답 처리
- `feedback_callback(feedback_msg)`: 피드백 수신 처리
- `get_result_async()`: 결과 요청

액션 클라이언트 + send_goal_async()


```python
goal_future = action_client.send_goal_async(goal_msg)
goal_future.add_done_callback(goal_response_callback)
```

```bash
add_done_callback(fn)	Future가 완료될 때 호출할 함수를 등록
fn(future)	콜백 함수는 future 객체를 인자로 받음
쓰임새	비동기 서비스/액션 결과 처리
```

액션 클라이언트 실행 흐름
```bash
send_goal_async(goal_msg)
        └──> feedback_callback() ← 피드백 받을 때마다 호출
        └──> goal_response_callback(future)
                  └──> get_result_async()
                          └──> get_result_callback(future)
```

콜백 예시 전체 코드


```python
def goal_response_callback(future):
    goal_handle = future.result()
    if not goal_handle.accepted:
        print('❌ Goal was rejected')
        return

    print('✅ Goal accepted')
    result_future = goal_handle.get_result_async()
    result_future.add_done_callback(get_result_callback)

def get_result_callback(future):
    result = future.result().result
    print(f'🎉 Result: {result.sequence}')

goal_future = action_client.send_goal_async(goal_msg)
goal_future.add_done_callback(goal_response_callback)
```

#### 액션 서버(Action Server)


```python
self._action_server = ActionServer(
    self,
    ActionType,
    'action_name',
    self.execute_callback,        # execute 단계에서 실행됨
    goal_callback=self.on_goal,   # goal 도착시 실행
)
```

- `ActionServer(self, ActionType, action_name, execute_callback)`: 액션 서버 생성
- `execute_callback(goal_handle)`: goal 처리 함수
  - 내부에서 `goal_handle.publish_feedback()`으로 피드백 전송
  - `goal_handle.succeed()` 또는 `goal_handle.abort()`
  - `return result` 로 결과 전달

액션 서버 콜백

여기서도 결국 executor가:

1. goal이 오면 → wait

2. goal을 DDS에서 → take

3. self.on_goal(goal_request) 실행 → execute

이 줄은 ROS 2의 액션 서버를 구현할 때 필요한 핵심 클래스와 상수를 임포트하는 구문


```python
from rclpy.action import ActionServer, GoalResponse, CancelResponse
```

```bash
ActionServer	액션 서버를 생성하는 클래스. 노드 안에서 특정 액션 타입을 처리할 수 있게 만듦
GoalResponse	goal을 수락할지, 거부할지를 나타내는 상수 (ACCEPT, REJECT)
CancelResponse	클라이언트가 요청한 goal 취소 요청에 대해 응답하는 상수 (ACCEPT, REJECT)
```

```bash
GoalResponse 상수
GoalResponse.ACCEPT
GoalResponse.REJECT
CancelResponse 상수
CancelResponse.ACCEPT
CancelResponse.REJECT
```

```bash
 rclpy.spin(node)을 호출하면:

while rclpy.ok():
    # 1. wait: DDS WaitSet으로부터 이벤트 기다림
    # 2. take: 누가 publish한 메시지 등 도착함
    # 3. execute: 등록된 콜백 함수 실행
```

#### 참고용 예시 메시지 타입
- Topic: `std_msgs.msg.String`, `sensor_msgs.msg.Image`
- Service: `example_interfaces.srv.AddTwoInts`
- Action: `example_interfaces.action.Fibonacci`

### ROS 2에서의 goal_handle

goal_handle은 action 서버가 클라이언트로부터 받은 목표(goal)에 대해 추적 및 상태를 관리하기 위해 사용하는 객체입니다.

작동 흐름 요약

1. 클라이언트가 send_goal_async(goal_msg)로 goal 전송

2. 서버는 execute_callback(goal_handle) 호출

3. 서버는 goal_handle.request를 통해 전달된 goal 내용을 읽음

동작 흐름
```bash
Client                        Server
  │                             │
  ├── send_goal_async() ──────▶│
  │                             │
  │        Future (goal_handle) │
  ├◀───────────────────────────┤
  │                             │
  ├─ goal_handle.get_result_async() ──▶ (결과 기다림)
  ```

goal_handle에서 자주 쓰는 속성과 메서드
```bash
속성/메서드	설명
goal_handle.request	클라이언트가 보낸 goal 요청 내용
goal_handle.accepted	이 goal이 수락되었는지 여부
goal_handle.publish_feedback(feedback_msg)	클라이언트에 피드백 보내기
goal_handle.succeed()	goal이 정상적으로 완료됨을 알림
goal_handle.abort()	goal 수행 중 실패를 알림
goal_handle.canceled()	클라이언트가 goal을 취소했음을 알림
goal_handle.is_cancel_requested	클라이언트가 취소 요청을 했는지 확인
ActionServer	액션 서버 생성 클래스
goal_callback	goal을 수락할지 거절할지 결정
cancel_callback	클라이언트가 goal을 취소하려고 할 때 처리
execute_callback	실제 작업 수행 / 피드백 보내고 결과 반환

```

예제 흐름 (서버에서)


```python
def goal_callback(self, goal_request):
    self.get_logger().info(f"Goal received: {goal_request}")
    return GoalResponse.ACCEPT

def cancel_callback(self, goal_handle):
    self.get_logger().info("Cancel request received.")
    return CancelResponse.ACCEPT

async def execute_callback(self, goal_handle):
```
