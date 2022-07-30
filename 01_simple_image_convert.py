# coding: UTF-8
#!/usr/bin/python3
import time
import argparse

import cv2
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--roll", type=int, default=0)
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--yaw", type=int, default=0)

    parser.add_argument("--viewpoint", type=float, default=-1.0)
    parser.add_argument("--imagepoint", type=float, default=1.0)

    parser.add_argument("--width", type=float, default=640)
    parser.add_argument("--height", type=float, default=360)

    parser.add_argument("--sensor_size", type=float, default=0.561)

    parser.add_argument("--image", type=str, default='sample.png')
    parser.add_argument("--output", type=str, default='output.png')

    args = parser.parse_args()

    return args


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
    # コマンドライン引数
    args = get_args()

    roll_degree = float(args.roll)
    pitch_degree = float(args.pitch)
    yaw_degree = float(args.yaw)

    viewpoint = args.viewpoint
    imagepoint = args.imagepoint + viewpoint

    output_width = args.width
    output_height = args.height

    sensor_width = args.sensor_size
    sensor_height = args.sensor_size * output_height / output_width

    image_path = args.image
    output_path = args.output

    # 処理時間
    time_list = []

    # 画像読み込み
    start_time = time.time()
    image = cv2.imread(image_path)
    time_list.append(['imread()', time.time() - start_time])

    # 回転行列生成
    start_time = time.time()
    rotation_matrix = create_rotation_matrix(
        roll_degree,
        pitch_degree,
        yaw_degree,
    )
    time_list.append(['create_rotation_matrix()', time.time() - start_time])

    # 角度座標φ, θ算出
    start_time = time.time()
    phi, theta = calculate_phi_and_theta(
        viewpoint,
        imagepoint,
        sensor_width,
        sensor_height,
        output_width,
        output_height,
        rotation_matrix,
    )
    time_list.append(['calculate_phi_and_theta()', time.time() - start_time])

    # 画像変換
    start_time = time.time()
    output_image = remap_image(image, phi, theta)
    time_list.append(['remap_image()', time.time() - start_time])

    # 処理時間表示
    total_elapsed_time = 0.0
    for time_info in time_list:
        elapsed_time = time_info[1] * 1000
        total_elapsed_time += elapsed_time
        print(time_info[0] + ':', '{:.1f}'.format(elapsed_time) + 'ms')
    print('\nTotal:', '{:.1f}'.format(total_elapsed_time) + 'ms')

    # 表示
    cv2.imwrite(output_path, output_image)


if __name__ == '__main__':
    main()
