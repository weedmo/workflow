# 4장 객체 추적과 움직임 (Object tracking and motion)

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/intermediate/ai-app-04-object_tracking.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, intermediate, study  
> source: raw/2026-09-26-rokey-ai/intermediate/강의_3기_AI응용_4차시__object_tracking_.ipynb  
> source: https://github.com/weedmo/Rokey-AI  


```python
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
```

## 객체 추적


```python
folder = "fig"
```

### 평균 이동 (Mean shift)


```python
# (function) def meanShift( probImage: MatLike, window: Rect, criteria: TermCriteria
# ) -> tuple[int, Rect]

# probImage: 히스토그램 역투영 영상 (확률 영상)
# window: 초기 검색 영역 윈도우
# criteri: 종료 기준
```


```python
import sys
import numpy as np
import cv2


# 비디오 파일 열기
# cap = cv2.VideoCapture('./fig/Billard.mp4')
cap = cv2.VideoCapture(Path(folder, "Billard.mp4"))


if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

ret, frame = cap.read()

if not ret:
    print('frame read failed!')
    sys.exit()

(x, y, w, h) = cv2.selectROI('ROI', frame)
rc = (x, y, w, h)



roi = frame[y:y+h, x:x+w]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# HS 히스토그램 계산
channels = [0, 1]
ranges = [0, 180, 0, 256]
hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)

# Mean Shift 알고리즘 종료 기준
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # HS 히스토그램에 대한 역투영
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)

    # Mean Shift
    _, rc = cv2.meanShift(backproj, rc, term_crit)

    # 추적 결과 화면 출력
    cv2.rectangle(frame, rc, (0, 0, 255), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(20) == 27:
        break

cap.release()
cv2.destroyAllWindows()

```

### 캠시프트 (Camshift)


```python
import sys
import numpy as np
import cv2


# 비디오 파일 열기
cap = cv2.VideoCapture(Path(folder, "Billard.mp4"))

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

ret, frame = cap.read()

if not ret:
    print('frame read failed!')
    sys.exit()

(x, y, w, h) = cv2.selectROI('ROI', frame)
rc = (x, y, w, h)

roi = frame[y:y+h, x:x+w]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# HS 히스토그램 계산
channels = [0, 1]
ranges = [0, 180, 0, 256]
hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)

# CamShift 알고리즘 종료 기준
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # HS 히스토그램에 대한 역투영
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)

    # CamShift
    ret, rc = cv2.CamShift(backproj, rc, term_crit)

    # 추적 결과 화면 출력
    cv2.rectangle(frame, rc, (0, 0, 255), 2)
    cv2.ellipse(frame, ret, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(60) == 27:
        break

cap.release()
cv2.destroyAllWindows()

```

## 모션 벡터 (Motion vector)

### Lucas-Kanade optical flow


```python
# def calcOpticalFlowPyrLK(
#     prevImg: MatLike, 첫 번째 frame
#     nextImg: MatLike, 두 번째 frame
#     prevPts: MatLike, 첫 번째 points
#     nextPts: MatLike,
#     status: MatLike | None = ...,
#     err: MatLike | None = ...,
#     winSize: Size = ...,
#     maxLevel: int = ...,
#     criteria: TermCriteria = ...,
#     flags: int = ...,
#     minEigThreshold: float = ...
# ) -> tuple[nextPts:MatLike, status:atLike, err:MatLike]
```


```python
import sys
import numpy as np
import cv2


# src1 = cv2.imread('fig/frame1.jpg')
# src2 = cv2.imread('fig/frame2.jpg')
src1 = cv2.imread(Path(folder, "frame1.jpg"))
src2 = cv2.imread(Path(folder, "frame2.jpg"))

if src1 is None or src2 is None:
    print('Image load failed!')
    sys.exit()

gray1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)

pt1 = cv2.goodFeaturesToTrack(gray1, 50, 0.01, 10)
pt2, status, err = cv2.calcOpticalFlowPyrLK(src1, src2, pt1, None)

dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

pt1
status
pt2
for i in range(pt2.shape[0]):
    if status[i, 0] == 0:
        continue

    cv2.circle(dst, tuple(pt1[i, 0].astype(int)), 4, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(dst, tuple(pt2[i, 0].astype(int)), 4, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.arrowedLine(dst, tuple(pt1[i, 0].astype(int)), tuple(pt2[i, 0].astype(int)), (0, 255, 0), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

```

### Dense optical flow


```python
# def calcOpticalFlowFarneback(
#     prev: MatLike, # 첫 번째 영상
#     next: MatLike, # 두 번째 영상
#     flow: MatLike, # None
#     pyr_scale: float, # 피라미드 영상 축소 비율
#     levels: int, # 피라미드 개수
#     winsize: int, # window 크기
#     iterations: int, # 알고리즘 반복 횟수
#     poly_n: int,  # 5 또는 7
#     poly_sigma: float, # 1.1 또는 1.5
#     flags: int # 0
# ) -> MatLike: ...
```


```python
import sys
import numpy as np
import cv2


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 255), lineType=cv2.LINE_AA)


    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 128, 255), -1, lineType=cv2.LINE_AA)

    return vis


# cap = cv2.VideoCapture('fig/vtest.avi')
cap = cv2.VideoCapture(Path(folder, "vtest.avi"))

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# # cap.set(cv2.CAP_PROP_FPS, 30)


if not cap.isOpened():
    print('Camera open failed!')
    sys.exit()

ret, frame1 = cap.read()

if not ret:
    print('frame read failed!')
    sys.exit()

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame2 = cap.read()

    if not ret:
        print('frame read failed!')
        sys.exit()

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 13, 3, 5, 1.1, 0)

    cv2.imshow('frame2', draw_flow(gray2, flow))
    if cv2.waitKey(20) == 27:
        break

    gray1 = gray2

cv2.destroyAllWindows()

```
