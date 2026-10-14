# 9차시_4_hangman_api_list

> **Migrated from Quartz wiki** (`wiki/study/ros-study/jupyter_ros/jupyter-09-4-hangman-api.md`, updated 2026-09-26)  
> tags: ros, ros2, notebook, jupyter_ros, study  
> source: raw/2026-09-26-ros-study/jupyter_ros/9차시_4_hangman_api_list.ipynb  
> source: https://github.com/weedmo/ROS-study  

## hangman_game/letter_publisher.py


```python
self.publisher_ = self.create_publisher(String, 'letter_topic', 10)
self.timer = self.create_timer(1.0, self.publish_letter)
```

letter_topic이라는 이름의 토픽을 만들고, std_msgs/msg/String 타입의 데이터를 퍼블리시할 퍼블리셔 객체 생성

10: 큐 사이즈 (토픽 메시지 버퍼 용량)

1초마다 self.publish_letter()를 실행하는 타이머 생성


```python
msg = String()
msg.data = chr(self.current_letter)
self.publisher_.publish(msg)
```

현재 알파벳 문자를 메시지에 담아 letter_topic으로 퍼블리시

## hangman_game/word_service.py


```python
self.service = self.create_service(CheckLetter, "check_letter", self.check_letter_callback)
self.subscription = self.create_subscription(String, "letter_topic", self.letter_callback, 10)
self.progress_publisher = self.create_publisher(Progress, "progress", 10)
```

check_letter라는 이름으로 서비스 서버 생성 (요청 받으면 check_letter_callback 실행)

서비스 타입: CheckLetter.srv (예: 문자 체크)

letter_topic을 구독하고, 새로운 메시지가 올 때 letter_callback을 호출함

progress라는 토픽에 퍼블리시할 퍼블리셔 생성 (Progress.msg 타입 사용)


```python
data = Progress()
self.progress_publisher.publish(data)
```

초기 Progress 메시지를 퍼블리시 (ex: 게임 시작 상태)


```python
 # Publish progress
progress_msg = Progress()
progress_msg.current_state = response.updated_word_state
progress_msg.attempts_left = self.attempts_left
self.progress_publisher.publish(progress_msg)
```

서비스 처리 결과(updated_word_state, 남은 시도 수)를 포함해 progress 토픽으로 게임 진행 상황 퍼블리시

## hangman_game/user_input.py


```python
self.cli = self.create_client(CheckLetter, "check_letter")
while not self.cli.wait_for_service(timeout_sec=1.0):
self.get_logger().info("Service not available, waiting...")
self.req = CheckLetter.Request()
```

check_letter 서비스를 호출할 수 있는 클라이언트 생성

서비스 서버가 뜰 때까지 대기 (1초마다 확인)

CheckLetter 서비스의 요청 메시지 인스턴스를 생성


```python
future = self.cli.call_async(self.req)
future.add_done_callback(self.callback_future)
```

비동기 방식으로 서비스 요청을 보내고 결과를 기다리는 Future 객체 반환
서비스 호출이 완료되면 결과 처리를 위한 콜백 등록


```python
response = future.result()
```

Future에서 실제 응답 데이터 추출 (보통 콜백 함수 내에서 사용)

## hangman_game/progress_action_server.py


```python
self._action_server = ActionServer(self, GameProgress, "game_progress", self.execute_callback)
```

game_progress라는 이름의 액션 서버 생성

액션 타입: GameProgress.action

콜백: 목표 수신 시 execute_callback() 실행


```python
self.subscription = self.create_subscription(Progress, "progress", self.progress_callback, 10)
```

progress 토픽을 구독해 게임 진행 상황을 모니터링

#### Action Server 의 주요 정의부(execute_callback함수)


```python
def execute_callback(self, goal_handle):
```

액션 서버에서 클라이언트로부터 goal을 받았을 때 호출되는 함수


```python
feedback_msg = GameProgress.Feedback()
```


```python
goal_handle.publish_feedback(feedback_msg)
```

피드백 메시지를 액션 클라이언트에 전송


```python
goal_handle.succeed()
```

액션 목표가 성공적으로 완료되었음을 알림

## hangman_game/progress_action_client.py


```python
self._action_client = ActionClient(self, GameProgress, "game_progress")
self.send_goal()
```

game_progress 액션 서버에 연결할 액션 클라이언트 생성

goal을 서버로 보낼 함수 호출


```python
self._action_client.wait_for_server()
goal_msg = GameProgress.Goal()
```

서버가 준비될 때까지 대기

액션 목표를 담을 메시지 인스턴스 생성


```python
self._send_goal_future = self._action_client.send_goal_async(goal_msg,feedback_callback=self.feedback_callback)
self._send_goal_future.add_done_callback(self.goal_response_callback)
```

비동기 방식으로 goal 전송, 피드백 수신용 콜백 설정

goal 전송 결과(수락 여부) 처리를 위한 콜백 등록


```python
def goal_response_callback(self, future):
```


```python
goal_handle = future.result()
```

goal 전송 결과에서 goal handle 추출


```python
def goal_response_callback(self, future):
```


```python
self._get_result_future = goal_handle.get_result_async()
self._get_result_future.add_done_callback(self.get_result_callback)
```

목표 결과를 비동기로 요청

목표 처리 결과 수신 시 실행할 콜백 등록


```python
def get_result_callback(self, future):
```


```python
result = future.result().result
```

액션 처리 결과 추출
