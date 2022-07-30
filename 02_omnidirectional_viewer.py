# coding: UTF-8
#!/usr/bin/python3
import copy
import time
import argparse

import cv2
import numpy as np

g_wheel = 0
g_drag_flag = False
g_prev_x, g_prev_y = None, None
g_diff_x, g_diff_y = None, None


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--viewpoint", type=float, default=-1.0)
    parser.add_argument("--imagepoint", type=float, default=1.0)

    parser.add_argument("--width", type=float, default=640)
    parser.add_argument("--height", type=float, default=360)

    parser.add_argument("--sensor_size", type=float, default=0.561)

    parser.add_argument("--image", type=str, default='sample.png')
    parser.add_argument("--movie", type=str, default=None)

    args = parser.parse_args()

    return args


def callback_mouse_event(event, x, y, flags, param):
    global g_drag_flag, g_prev_x, g_prev_y, g_diff_x, g_diff_y, g_wheel

    # ホイール
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            g_wheel += 1
        else:
            g_wheel -= 1

    # 左ドラッグ
    if event == cv2.EVENT_LBUTTONDOWN:
        g_drag_flag = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if g_drag_flag:
            if g_prev_x is not None and g_prev_y is not None:
                g_diff_x, g_diff_y = g_prev_x - x, g_prev_y - y
    elif event == cv2.EVENT_LBUTTONUP:
        g_drag_flag = False
        g_diff_x, g_diff_y = 0, 0
    g_prev_x, g_prev_y = x, y


def create_rotation_matrix(roll, pitch, yaw):
    roll = roll * np.pi / 180
    pitch = pitch * np.pi / 180
    yaw = yaw * np.pi / 180

    matrix01 = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)],
    ])

    matrix02 = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)],
    ])

    matrix03 = np.array([
        [np.cos(yaw), np.sin(yaw), 0],
        [-np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    matrix = np.dot(matrix03, np.dot(matrix02, matrix01))

    return matrix


def calculate_phi_and_theta(
    viewpoint,
    imagepoint,
    sensor_width,
    sensor_height,
    output_width,
    output_height,
    rotation_matrix,
):
    width = np.arange(
        (-1) * sensor_width,
        sensor_width,
        sensor_width * 2 / output_width,
    )
    height = np.arange(
        (-1) * sensor_height,
        sensor_height,
        sensor_height * 2 / output_height,
    )

    ww, hh = np.meshgrid(width, height)

    point_distance = (imagepoint - viewpoint)
    if point_distance == 0:
        point_distance = 0.1

    a1 = ww / point_distance
    a2 = hh / point_distance
    b1 = -a1 * viewpoint
    b2 = -a2 * viewpoint

    a = 1 + (a1**2) + (a2**2)
    b = 2 * ((a1 * b1) + (a2 * b2))
    c = (b1**2) + (b2**2) - 1

    d = ((b**2) - (4 * a * c))**(1 / 2)

    x = (-b + d) / (2 * a)
    y = (a1 * x) + b1
    z = (a2 * x) + b2

    xd = rotation_matrix[0][0] * x + rotation_matrix[0][
        1] * y + rotation_matrix[0][2] * z
    yd = rotation_matrix[1][0] * x + rotation_matrix[1][
        1] * y + rotation_matrix[1][2] * z
    zd = rotation_matrix[2][0] * x + rotation_matrix[2][
        1] * y + rotation_matrix[2][2] * z

    phi = np.arcsin(zd)
    theta = np.arcsin(yd / np.cos(phi))

    xd[xd > 0] = 0
    xd[xd < 0] = 1
    yd[yd > 0] = np.pi
    yd[yd < 0] = -np.pi

    offset = yd * xd
    gain = -2 * xd + 1
    theta = gain * theta + offset

    return phi, theta


def remap_image(image, phi, theta):
    input_height, input_width = image.shape[:2]

    phi = (phi * input_height / np.pi + input_height / 2)
    phi = phi.astype(np.float32)
    theta = (theta * input_width / (2 * np.pi) + input_width / 2)
    theta = theta.astype(np.float32)

    output_image = cv2.remap(image, theta, phi, cv2.INTER_CUBIC)

    return output_image


def main():
    global g_diff_x, g_diff_y, g_wheel

    # コマンドライン引数
    args = get_args()

    viewpoint = args.viewpoint
    imagepoint = args.imagepoint + viewpoint
    base_imagepoint = imagepoint

    output_width = args.width
    output_height = args.height

    sensor_width = args.sensor_size
    sensor_height = args.sensor_size * output_height / output_width

    image_path = args.image
    movie_path = args.movie

    # 画面操作用変数
    roll = 0
    pitch = 0
    yaw = 0

    drag_rate = int(output_width / 200)
    wheel_rate = int(output_width / 50)

    # ウィンドウ生成
    window_name = 'Omnidirectional Viewer Using OpenCV'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, callback_mouse_event)

    # 映像・画像読み込み
    video_capture = None
    if movie_path is not None:
        video_capture = cv2.VideoCapture(movie_path)
    else:
        image = cv2.imread(image_path)

    while True:
        start_time = time.time()

        # GUI操作：ピッチ・ヨー操作
        if g_diff_y is not None and g_diff_y != 0:
            pitch += (g_diff_y / drag_rate)
            pitch %= 360
            if pitch > 180:
                pitch -= 360
            g_diff_y = 0
        if g_diff_x is not None and g_diff_x != 0:
            yaw -= (g_diff_x / drag_rate)
            yaw %= 360
            if yaw > 180:
                yaw -= 360
            g_diff_x = 0

        # GUI操作：ズーム操作
        imagepoint = base_imagepoint + (g_wheel / wheel_rate)
        if imagepoint < viewpoint:
            imagepoint = viewpoint
            g_wheel += 1

        # 画像読み込み
        if movie_path is not None:
            ret, frame = video_capture.read()
            if not ret:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, frame = video_capture.read()
        else:
            frame = copy.deepcopy(image)

        # 回転行列生成
        rotation_matrix = create_rotation_matrix(
            roll,
            pitch,
            yaw,
        )

        # 角度座標φ, θ算出
        phi, theta = calculate_phi_and_theta(
            viewpoint,
            imagepoint,
            sensor_width,
            sensor_height,
            output_width,
            output_height,
            rotation_matrix,
        )

        # 画像変換
        output_image = remap_image(frame, phi, theta)

        elapsed_time = time.time() - start_time

        # 情報表示
        cv2.putText(
            output_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1,
            cv2.LINE_AA)
        cv2.putText(output_image, "Viewpoint : " + '{:.1f}'.format(viewpoint),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(output_image,
                    "Imagepoint : " + '{:.1f}'.format(imagepoint), (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(output_image, "Yaw : " + str(int(yaw)), (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(output_image, "Pitch : " + str(int(pitch)), (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

        # 描画
        cv2.imshow(window_name, output_image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    if video_capture is not None:
        video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
