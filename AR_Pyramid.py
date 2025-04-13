import numpy as np
import cv2 as cv

# === 비디오와 캘리브레이션 데이터 로드 ===
video_file = "D:\\ComputerVision\\CV_image\\my_chessboard.mp4"
data = np.load('calibration_data.npz')
K = data['mtx']
dist_coeff = data['dist']

# === 체스보드 설정 ===
board_pattern = (8, 6)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# === 비디오 열기 ===
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# === 체스보드 3D 좌표 생성 ===
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# === 피라미드 3D 포인트 정의 ===
# === 밑면 1 (3x3 큰 밑면) ===
base_bottom = board_cellsize * np.array([
    [3, 1, 0],
    [6, 1, 0],
    [6, 4, 0],
    [3, 4, 0],
])

# === 밑면 2 (2x2 작은 밑면, 중간 층) ===
base_middle = board_cellsize * np.array([
    [3.5, 1.5, -1],
    [5.5, 1.5, -1],
    [5.5, 3.5, -1],
    [3.5, 3.5, -1],
])
apex = board_cellsize * np.array([[4.5, 2.5, -2]])  # 꼭짓점 (정사각형 중심에서 2m 위)
pyramid_pts_3d = np.vstack([base_bottom, base_middle, apex])  # 총 9개 점

# === 프레임 반복 처리 ===
while True:
    valid, img = video.read()
    if not valid:
        break

    # === 체스보드 찾기 및 포즈 추정 ===
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # === 피라미드 3D -> 2D 투영 ===
        pyramid_pts_2d, _ = cv.projectPoints(pyramid_pts_3d, rvec, tvec, K, dist_coeff)
        pts = np.int32(pyramid_pts_2d).reshape(-1, 2)

        # === 밑면 채우기 ===
        cv.fillConvexPoly(img, pts[:4], color=(0, 255, 255))  # 노란색 (BGR: 0,255,255)

        # === 측면 1층 (큰 밑면 → 중간 밑면 연결) ===
        for i in range(4):
            side = np.array([pts[i], pts[(i+1)%4], pts[(i+1)%4 + 4], pts[i + 4]])
            cv.fillConvexPoly(img, side, (0, 200, 255))  # 어두운 노란색

        # === 밑면 2 (중간 밑면) ===
        cv.fillConvexPoly(img, pts[4:8], (0, 255, 180))  # 밝은 노랑-초록

        # === 측면 2층 (중간 밑면 → 꼭짓점 연결) ===
        for i in range(4):
            triangle = np.array([pts[i + 4], pts[(i+1)%4 + 4], pts[8]])
            cv.fillConvexPoly(img, triangle, (0, 255, 200))  # 초록 노랑 느낌

        # === 선 그리기 (테두리) ===
        cv.polylines(img, [pts[0:4]], isClosed=True, color=(255, 0, 0), thickness=2)
        cv.polylines(img, [pts[4:8]], isClosed=True, color=(255, 100, 0), thickness=2)
        for i in range(4):
            cv.line(img, pts[i], pts[i + 4], (100, 255, 255), 1)       # 1층 측면
            cv.line(img, pts[i + 4], pts[8], (0, 100, 255), 2)         # 2층 측면

        # === 카메라 위치 계산 및 출력 ===
        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # === 결과 출력 및 키 입력 처리 ===
    cv.imshow('Pose Estimation (Chessboard with Pyramid)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27:  # ESC
        break

video.release()
cv.destroyAllWindows()
