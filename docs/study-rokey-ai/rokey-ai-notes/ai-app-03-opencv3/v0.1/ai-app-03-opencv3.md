# 3장 OpenCV를 이용한 객체 검출,  동영상 처리

> **Migrated from Quartz wiki** (`wiki/study/rokey-ai/intermediate/ai-app-03-opencv3.md`, updated 2026-09-26)  
> tags: rokey, ai, notebook, intermediate, study  
> source: raw/2026-09-26-rokey-ai/intermediate/강의_3기_AI응용_3차시__OpenCV3_.ipynb  
> source: https://github.com/weedmo/Rokey-AI  


```python
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from pathlib import Path
```

## 객체 검출


```python
folder = "fig"
```

### 소벨 필터


```python
kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], np.float32)
```


```python

# src = cv2.imread("./fig/bamboo.jpg", cv2.IMREAD_GRAYSCALE)
src = cv2.imread(Path(folder, "bamboo.jpg"), cv2.IMREAD_GRAYSCALE)


# kernel


dst = cv2.filter2D(src, -1, kernel)

cv2.imshow("dst", src)

while True:
    if cv2.waitKey() == 27:
        break

cv2.destroyAllWindows()
```


```python
src = cv2.imread(Path(folder, "bamboo.jpg"), cv2.IMREAD_GRAYSCALE)
# src = cv2.imread("./fig/plates.png", cv2.IMREAD_GRAYSCALE)


kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], np.float32)
# print(kernel)
# print(kernel.T)

dst1 = cv2.filter2D(src, -1, kernel)
dst2 = cv2.filter2D(src, -1, kernel.T)

cv2.imshow("bamboo", src)
cv2.imshow("dst1", dst1)
cv2.imshow("dst2", dst2)


while True:
    if cv2.waitKey() == 27:
        break

cv2.destroyAllWindows()
```


```python
## Sobel filter
# cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delt, borderType) -> dst
# src : 입력영상
# ddepth : 출력영상의 데이터 타입 (-1)
# dx : x 방향 미분차수
# dy : x 방향 미분차수
# dst : 출력영상
# ksize : 커널의 크기
# scale : 연산결과에 추가적으로 곱할 값
# delta : 연산결과에 추가적으로 더할 값
# borderType : 가장자리 픽셀확장 방식

# magnitude(x, y, magnitude) -> magnitude
```


```python
# src = cv2.imread("./fig/son.jpg", cv2.IMREAD_GRAYSCALE)
src = cv2.imread(Path(folder, "son.jpg"), cv2.IMREAD_GRAYSCALE)


dx = cv2.Sobel(src, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(src, cv2.CV_32F, 0, 1)

mag = cv2.magnitude(dx, dy)
mag = np.clip(mag,  0, 255).astype(np.uint8)

ret, dst = cv2.threshold(mag, 150, 255, cv2.THRESH_BINARY)
background = np.zeros((mag.shape[0], mag.shape[1]), np.uint8)
background[mag > 150] = 255

cv2.imshow("src", src)
# cv2.imshow("dx", dx)
# cv2.imshow("dy", dy)
cv2.imshow("mag", mag)
cv2.imshow("dst", dst)
cv2.imshow("background", background)


# plt.imshow(mag, cmap = "gray")
# plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
```

### Canny 에지필터


```python
## Canny edge
# Canny(image, threshold1, threshold2, edges, apertureSize, L2gradient) -> edges
# image : 입력 영상
# threshold1: 에지결정 하한값
# threshold1: 에지결정 상한값
# edges: None
# apertureSize: 커널사이즈
# L2gradient: gradient 크기 계산, False
```


```python
src = cv2.imread(Path(folder, "son.jpg"), cv2.IMREAD_GRAYSCALE)
dst = cv2.Canny(src, 150, 180)

cv2.imshow("src", src)
cv2.imshow("dst", dst)

cv2.waitKey()
cv2.destroyAllWindows()
```

### Hough 변환: 직선검출


```python
## 직선 검출, 곡선 검출; 허프 변환 (Hough transform)
# HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap) -> lines
# image: 입력 에지영상
# rho: 축적배열에서 rho의 간격
# theta: 축적배열에서 theta의 간격
# threshold: 직선판단할 임계값
# lines: 선분의 끝좌표 (x1, y1, x2, y2)
# srn = None, stn = None
# minLineLength: 검출한 선분의 최소 길이
# maxLineGap: 직선으로 간주할 최대 에지 점 간격 (끝어진 점을 연결할 기준)
```


```python
# img = cv2.imread("./fig/checkerboard.png", cv2.IMREAD_GRAYSCALE)
img = cv2.imread(Path(folder, "checkerboard.png"), cv2.IMREAD_GRAYSCALE)

edge = cv2.Canny(img, 100, 200)
# ret, edge = cv2.threshold(edge, )

lines = cv2.HoughLinesP(edge, 1, np.pi/360, 100, minLineLength = 10, maxLineGap = 30)
print(lines.shape[0]) # 라인갯수

dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

for i in range(lines.shape[0]):
    pt1 = (lines[i][0][0], lines[i][0][1])
    pt2 = (lines[i][0][2], lines[i][0][3])
    cv2.line(dst, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.imshow('edge', edge)
cv2.imshow("dst", dst)

cv2.waitKey()
cv2.destroyAllWindows()
```

    52


### Hough 변환: 곡선검출


```python
# HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
# image: 입력영상
# method: cv2.HOUGH_GRADIENT,
# dp: 입력영상에 대한 실제 영상처리 배율, 1,  2 설정
# minDist: 검출된 원들간의 최소거리
# circles: 원좌표 (cx, cy, r), shape = (1, N, 3), dtype = np.float32
# param1: Canny edge max 값
# param2: 축척배열에서 원검출 임계값
# minRadius: 원 크기의 최소값
# maxRadius: 원 크기의 최대값
```


```python
## Hough transform, 원 검출

# src = cv2.imread("./fig/plates.png")
src = cv2.imread(Path(folder, "plates.png"))

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# src_gray =  255 - src_gray
circles = cv2.HoughCircles(src_gray, cv2.HOUGH_GRADIENT, 1, 
                           50, param1=200, param2= 100, minRadius=50, maxRadius=150)

print(circles.shape[1])

for i in range(circles.shape[1]):
    cx, cy, radius = circles[0][i]
    cv2.circle(src, (int(cx), int(cy)), int(radius), (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow('src', src)
cv2.imshow("src-gray", src_gray)

cv2.waitKey()
cv2.destroyAllWindows()
```

    6


### 레이블링 (labeling)


```python
## 레이블링 (labeling)

# connectedComponentsWithStats(image[, labels[, stats[, centroids[, connectivity[, ltype]]]]]) -> retval, labels, stats, centroids
```


```python
# src = cv2.imread("./fig/symbols.png")
src = cv2.imread(Path(folder, "symbols.png"))

# src = cv2.GaussianBlur(src, (0, 0), 1)

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(src_gray, 100, 255, cv2.THRESH_BINARY)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)


cnts, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
print(stats[1])

for i in range(1, cnts):
    (x, y, w, h, area) = stats[i]

    if area <= 50:
        continue

    cv2.rectangle(src, (x, y, w, h), (0, 0, 255), 2)

print("count = ", cnts)

cv2.imshow("src", src)
# cv2.imshow("gray", src_gray)
cv2.imshow("mask", mask)

cv2.waitKey()
cv2.destroyAllWindows()
```

    [     0      0    512    512 217198]
    count =  2


### 외곽선 검출


```python
## 외곽선 검출
# findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy
    # image: 입력 영상. non-zero 픽셀을 객체로 간주함.
    # mode: 외곽선 검출 모드. cv2.RETR_로 시작하는 상수. 
    #     (cv2.RETR_EXTERNAL, cv2.RETR_LIST,cv2.RETR_CCOMP, cv2.RETR_TREE)
    # method: 외곽선 근사화 방법. cv2.CHAIN_APPROX_로 시작하는 상수.
    # contour: 검출된 외곽선 좌표. numpy.ndarray로 구성된 리스트. 
    # contours[i].shape=(K, 1, 2). contours[i].dtype=numpy.int32.
    # hierarchy: 외곽선 계층 정보. numpy.ndarray. shape=(1, N, 4). dtype=numpy.int32.
    # hierarchy[0, i, 0] ~ hierarchy[0, i, 3]이 순서대로 next, prev, child, parent
    # 외곽선 인덱스를 가리킴. 해당 외곽선이 없으면 -1.
    # offset: 좌표 값 이동 옵셋. 기본값은 (0, 0).
    
# drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) -> image
```


```python
# src = cv2.imread("./fig/shape.png")
src = cv2.imread(Path(folder, "shape.png"))

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(src_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
contours, hierachy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    cv2.drawContours(src, contours, i, (0, 0, 255), 1)
    cv2.putText(src, str(i), contours[i][0][0], cv2.FONT_HERSHEY_COMPLEX, 1,
                (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('src', src)
cv2.imshow('src_gray', src_gray)
cv2.imshow("mask", mask)
cv2.waitKey()
cv2.destroyAllWindows()
```

### 기하학적 모멘트


```python
## 기하학적 모멘트 (Hu 불변 모멘트)
# obj = cv2.imread("./fig/spades.png", cv2.IMREAD_GRAYSCALE)
# src = cv2.imread("./fig/symbols.png", cv2.IMREAD_GRAYSCALE)
obj = cv2.imread(Path(folder, "spades.png"), cv2.IMREAD_GRAYSCALE)
src = cv2.imread(Path(folder, "symbols.png"), cv2.IMREAD_GRAYSCALE)



_, obj_bin = cv2.threshold(obj, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
obj_contours, _ = cv2.findContours(obj_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
obj_pts = obj_contours[0]
_, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
src_contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

for pts in src_contours:
    if cv2.contourArea(pts) < 1000:
        continue
    rc = cv2.boundingRect(pts)
    cv2.rectangle(dst, rc, (255, 0, 0), 1)
    dist = cv2.matchShapes(obj_pts, pts, cv2.CONTOURS_MATCH_I3, 0)
    cv2.putText(dst, str(round(dist, 3)), (rc[0], rc[1] - 3), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                (255, 0, 0), 1, cv2.LINE_AA)
    
    if dist < 0.1:
        cv2.rectangle(dst, rc, (0, 0, 255), 2)
        cv2.putText(dst, str(round(dist, 3)), (rc[0], rc[1] - 3), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                (0, 0, 255), 2, cv2.LINE_AA)
        
cv2.imshow("obj", obj)
cv2.imshow("src", src)
cv2.imshow("dst", dst)
# cv2.imshow("obj_bin", obj_bin)

cv2.waitKey()
cv2.destroyAllWindows()
```

## 동영상 처리

### 동영상 객체 검출


```python
# cv2.VideoCapture(index/filename, apiPreference=None) -> retval
# index: camera_id or filename
# apiPreference=None

# cv2.VideoWriter(filename, fourcc, fps, framesize, isColor=None) -> retval
# filename: 저장할 이름
# fourcc: cv2.VideoWriter_fourcc(*'DIVX') 를 사용
# fps: 초당 프레임 수 e.g. 30
# frameSize: 프레임 크기 e.g., [640, 480])
# isColor: Color 영상
```


```python

# cap = cv2.VideoCapture("./fig/PETS2000.avi")
cap = cv2.VideoCapture(Path(folder, "PETS2000.avi"))

if not cap.isOpened():
    print("Video open failed")
    sys.exit()

ret, background = cap.read()
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background_gray_G = cv2.GaussianBlur(background_gray, (0, 0), 1.) 
# cv2.imshow("background", background)
# cv2.waitKey()
while True:
    ret, frame = cap.read() # fps: frame per second

    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray_G = cv2.GaussianBlur(frame_gray, (0, 0), 1.)
    diff_G = cv2.absdiff(frame_gray_G, background_gray_G)
    ret_g, mask_g = cv2.threshold(diff_G, 50, 255, cv2.THRESH_BINARY)
    cnts, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_g)
    for i in range(1, cnts):
        x, y, w, h, area = stats[i]
        if area <= 200:
            continue
        cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("diff", diff_G)
    cv2.imshow("mask_g", mask_g)

    if cv2.waitKey(20) == 27:
        break

cv2.destroyAllWindows()
cap.release()

```

### 이동 평균 배경 차분


```python
# accumulateWeighted(src, dst, alpha, mask) -> dst
# src: 입력영상
# dis: 출력영상 (32bit, 64bit)
# alpha : 축적가중치
# mask: 마스트 영상
```


```python
## 이동 평균 배경 차분
cap = cv2.VideoCapture(Path(folder, "PETS2000.avi"))

if not cap.isOpened():
    print("Video open failed")
    sys.exit()
ret, back = cap.read()
back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
back = cv2.GaussianBlur(back, (0, 0), 1.)
fback = back.astype(np.float32)
while True:
    ret, frame = cap.read()
    if not ret:
        print("frame is None")
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (0, 0), 1)
    cv2.accumulateWeighted(frame_gray, fback, 0.01)
    back = fback.astype(np.uint8)
    diff = cv2.absdiff(frame_gray, back)
    ret, mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    cnts, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    for i in range(1, cnts):
        x, y, w, h, area = stats[i]
        if area < 100:
            continue
        cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)

    cv2.imshow("frame",frame)
    cv2.imshow("mask",mask)
    cv2.imshow("back",back)
    
    if cv2.waitKey(20) == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
```

### Webcam 열기


```python
# 카메라 열기
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 30)
# cap = cv2.VideoCapture('raining.mp4')

if not cap.isOpened(): #True or Falose
    print("Camera open failed")
    cap.release()
    sys.exit()


# 카메라 프레임 크기 출력
print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


# 카메라 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        
        break

    edge = cv2.Canny(frame, 50, 150)       
    # inversed = ~frame  # 반전
    
    cv2.imshow('frame', frame)
    cv2.imshow('frame1', edge)
#     cv2.imshow('inversed', inversed)

    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
```

    Frame width: 640
    Frame height: 480


### 동영상 저장


```python
# cv2.VideoWriter(filename, fourcc, fps, framesize, isColor=None) -> retval
# filename: 저장할 이름
# fourcc: cv2.VideoWriter_fourcc(*'DIVX') 를 사용
# fps: 초당 프레임 수 e.g. 30
# frameSize: 프레임 크기 e.g., [640, 480])
# isColor: Color 영상
```


```python
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

# cap.get(cv2.CAP_PROP_FRAME_WIDTH) -> float type 반환
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fps = cap.get(cv2.CAP_PROP_FPS) # Frame per second
fps = 30


# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
# fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') # *'DIVX' 


delay = round(1000 / fps) #frame 간 시간 간격, ms 단위

out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h), 
                      isColor = True)

if not out.isOpened():
    print('File open failed!')
    cap.release()
    sys.exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

#     inversed = ~frame
#     edge = cv2.Canny(frame, 50, 150)
#     edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    out.write(frame) #소리는 capture가 안됨
#     out.write(inversed)
#     out.write(edge_color)

    cv2.imshow('frame', frame)
#     cv2.imshow('inversed', inversed)
#     cv2.imshow('edge', edge)
    if cv2.waitKey(delay) == 27:
        break


out.release()
cv2.destroyAllWindows()
```

### 카툰 필터 카메라


```python
# 카툰 필터 카메라
import sys 
import numpy as np
import cv2


def cartoon_filter(img):
    h, w = img.shape[:2]
#     img2 = cv2.resize(img, (w//2, h//2))

    blr = cv2.bilateralFilter(img, -1, 20, 7)
    edge = 255 - cv2.Canny(img, 80, 120)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    dst = cv2.bitwise_and(blr, edge)
    dst = cv2.resize(dst, (w, h), interpolation=cv2.INTER_NEAREST)

    return dst


def pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gray, (0, 0), 3)
    dst = cv2.divide(gray, blr, scale=255)
    return dst


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print('video open failed!')
    sys.exit()

cam_mode = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if cam_mode == 1:
        frame = cartoon_filter(frame)
    elif cam_mode == 2:
        frame = pencil_sketch(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord(' '):
        cam_mode += 1
        if cam_mode == 3:
            cam_mode = 0


cap.release()
cv2.destroyAllWindows()
```
