import numpy as np
import cv2
import os
from pylibdmtx.pylibdmtx import decode, encode
import matplotlib.pyplot as plt
import glob
import json
import time
import pandas as pd
import re
from PIL import Image, ImageEnhance

def visualize(img, title=''):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def refine_dm_bbox(binary_image, expand, rotated=False):
    """
    根据max_contour 精定位DM码的外接矩形
    :param binary_image: DM码二值图
    :return: 外接矩形的x, y, w, h
    """
    # tighten the bounding box: assume the finder pattern is darker!
    contours, _ = cv2.findContours(255 - binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = np.concatenate(list(contours))
    max_cnt = None
    angle = 0
    max_area = -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_cnt = cnt
            max_area = area


    if rotated:
        rrect = cv2.minAreaRect(all_contours)
        ((x, y), (w, h), angle) = rrect
    else:
        x, y, w, h = cv2.boundingRect(all_contours)
    #
    # if w / dm_image.shape[1] < 0.875:
    #
    #     w = dm_image.shape[1]
    #     x = 0
    #
    # if h / dm_image.shape[0] < 0.875:
    #     h = dm_image.shape[0]
    #     y = 0

    return x, y, w, h, angle



def count_transitions(binary_arr):
    """
    计算二值化矩阵某一方向内的跳变次数
    :param binary_arr: 二值化（0/1）的矩阵
    :return: 值跳变次数
    """

    if binary_arr.shape[0] < binary_arr.shape[1]:
        axis = 1
    else:
        axis = 0

    if np.min(binary_arr.shape) == 1:
        # 矩阵的宽或高=1
        return int(np.sum(np.abs(np.diff(binary_arr, axis=axis)) > 0))
    else:
        return int(np.median(np.sum(np.abs(np.diff(binary_arr, axis=axis)) > 0, axis=axis)))

def count_transitions2(binary_arr):
    """
    计算二值化矩阵某一方向内的跳变次数
    :param binary_arr: 二值化（0/1）的矩阵
    :return: 值跳变次数
    """

    valid_transition_list = []
    for array in binary_arr:
        diff_array = np.abs(np.diff(array))/255
        l_sum = np.sum(diff_array[:len(diff_array)//2] > 0)
        r_sum = np.sum(diff_array[len(diff_array)//2:] > 0)
        sum = l_sum + r_sum
        if np.sum(array)/len(array) > 0.5 and sum < 5:
            continue

        if np.abs(l_sum - r_sum) > 4:
            continue

        valid_transition_list.append(sum)
        if len(valid_transition_list) > 3:
            break

    if len(valid_transition_list) == 0:
        return 0
    else:
        return int(np.median(valid_transition_list))

def locate_timing_pattern(binary_dm_image, band_offset, band_width, method_flag=1):
    """
    定位timing pattern的位置
    :param binary_dm_image: DM码二值图
    :param band_offset: offset
    :param band_width: 边宽
    :return: 横向和纵向的timing pattern的位置
    """
    band_offset = 1
    band_width = 15
    if method_flag == 1:
        # 提取DM码二值图的四个边
        if np.max(binary_dm_image) == 255:
            # 转成0/1的值
            binary_dm_image = (binary_dm_image / 255).astype(np.uint8)

        top_row = binary_dm_image[band_offset: band_offset + band_width, :]
        bottom_row = binary_dm_image[-band_offset - 1 - band_width:-band_offset - 1, :]
        left_col = binary_dm_image[:, band_offset: band_offset + band_width]
        right_col = binary_dm_image[:, -band_offset - 1 - band_width: -band_offset - 1]

        # 计算每个边的跳变数量
        transitions = {
            'top': count_transitions2(top_row),
            'bottom': count_transitions2(bottom_row),
            'left': count_transitions2(left_col.T),
            'right': count_transitions2(right_col.T)
        }

    # 跳变数量最多的二两边是Timing pattern
    sorted_edges = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    timing_edges = sorted_edges[:2]

    # 检查Timing pattern的位置是否合理(例如2个timing patterns都是横的或都是竖的）
    horizontal_timing = max([e for e in timing_edges if e[0] in ['top', 'bottom']], key=lambda x: x[1])[0]
    vertical_timing = max([e for e in timing_edges if e[0] in ['left', 'right']], key=lambda x: x[1])[0]

    return horizontal_timing, vertical_timing

def search_dm_dimensions(dm_image, possible_dimensions, timing_pattern_pos):
    """
    搜索最优的DM码中cells的行列数
    :param dm_image: DM码的灰度图片
    :param possible_dimensions: 一个可能的cells行列组合的列表
    :param timing_pattern_pos: 一个包含timing pattern的2个位置的列表
    :return: DM码cells的行数和列数
    """

    height, width = dm_image.shape[:2]
    max_diff = -1
    best_dim = None
    for dim in possible_dimensions:
        # 循环所有可能的cells行列组合
        rows, cols = dim
        cell_width = width / cols
        cell_height = height / rows

        if cell_width < 2 or cell_height < 2:
            # 如果单个cell的宽高小于2个像素，直接跳过这个尺寸。
            continue

        #取timing pattern时从图片边缘增加一点offset
        offset_width = np.max([int(cell_width*1/4), 1])
        offset_height = np.max([int(cell_width*1/4), 1])

        # 预设DM码图片的四个条边的图片，对应的K个cell数，每个cell的宽 width
        # 后面根据timing pattern的位置来取对应的信息
        pattern_info_map = {
            'top': [dm_image[offset_height: int(cell_height-offset_height), :], cols, cell_width],
            'bottom': [dm_image[int(- cell_height + offset_height): int(height-offset_height), :], cols, cell_width],
            'left': [dm_image[:, offset_width: int(cell_width - offset_width)].T, rows, cell_height],
            'right': [dm_image[:, int(- cell_width + offset_width): int(width - offset_width)].T, rows, cell_height]
        }

        # 计算出2个timing pattern这边奇与偶数cells之间的最小绝对差只和
        diff = 0
        for pos in timing_pattern_pos:
            pattern, k, w = pattern_info_map[pos]
            # 计算所有奇数和偶数cell的中指，然后算所有奇与偶数cells之间的最小绝对差
            even_cell_values = np.array([np.median(pattern[:, int(i*w+1): int((i + 1) * w+1) if i < k - 1 else pattern.shape[1]]) for i in range(0, k, 2)])
            odd_cell_values = np.array([np.median(pattern[:, int(i*w+1): int((i + 1) * w+1) if i < k - 1 else pattern.shape[1]]) for i in range(1, k+1, 2)])
            pos_mean_diff = np.min(np.abs(odd_cell_values - even_cell_values))
            diff += pos_mean_diff

        if diff > max_diff:
            # 奇与偶数cells之间的最小绝对差最大的cells行列组合，即是最优组合
            max_diff = diff
            best_dim = dim

    rows, cols = best_dim
    return rows, cols

def compute_dm_array(binary_dm_image, rows, cols, kernel_size, ratio_threshold, save_image=False):
    """
    根据DM码的二值图，判断出每个cell的颜色（0/1），生成一个DM码0/1矩阵
    :param binary_dm_image: DM码二值图
    :param rows: 多少行cells
    :param cols: 多少列cells
    :return: DM码矩阵
    """

    binary_image_show = cv2.cvtColor((binary_dm_image).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    h, w = binary_dm_image.shape[0], binary_dm_image.shape[1]
    cell_width = w / cols
    cell_height = h / rows
    sync_dm_array = np.zeros([rows, cols])

    for i in range(rows):  # 行
        for j in range(cols):  # 列
            # 计算子块坐标（最后一行/列可能略大，确保覆盖全图）
            y1 = int(i * cell_height + 1)
            y2 = int((i + 1) * cell_height + 1) if i < rows - 1 else h  # 最后一行覆盖剩余高度
            x1 = int(j * cell_width + 1)
            x2 = int((j + 1) * cell_width + 1) if j < cols - 1 else w  # 最后一列覆盖剩余宽度

            if save_image:
                # 绘制子块边框
                cv2.rectangle(
                    binary_image_show,
                    (x1, y1),  # 左上角
                    (x2, y2),  # 右下角
                    (0, 255, 0),  # 绿色边框
                    1  # 边框厚度

                )

            # 计算子块中心坐标
            center_x = round((x1 + x2) // 2)
            center_y = round((y1 + y2) // 2)

            # 获取中心白色像素比例并判断是否修改子块
            center_roi = binary_dm_image[center_y - kernel_size:center_y + kernel_size, \
                         center_x - kernel_size:center_x + kernel_size]

            white_ratio = np.count_nonzero(center_roi) / ((2*kernel_size) ** 2)
            # white_ratio = np.sum(center_roi) / ((2 * kernel_size) ** 2)
            if white_ratio > ratio_threshold:
                sync_dm_array[i, j] = 1

    return sync_dm_array, binary_image_show

def correct_timing_finder_pattern(sync_dm_array, horizontal_timing_pos, vertical_timing_pos):
    """
    纠正timing和finder pattern
    :param sync_dm_array: DM的二值化（0/1）矩阵，每一个element代表DM码的一个cell，即如果DM码的cell数量是rows x cols，这个矩阵尺寸是rows x cols
    :param horizontal_timing_pos: 横向的timing pattern的位置：top 或 bottom
    :param vertical_timing_pos: 纵向的timing pattern的位置：left 或 right
    :return: timing和pattern被更正后的sync_dm_array
    """

    if horizontal_timing_pos == 'top':
        # bottom is Finder pattern
        sync_dm_array[-1, :] = 0
        if vertical_timing_pos == 'left':
            timing_col_idx = 0
            # right is Finder pattern
            sync_dm_array[:, -1] = 0

        if vertical_timing_pos == 'right':
            timing_col_idx = -1
            # left is Finder pattern
            sync_dm_array[:, 0] = 0

        for i in range(rows):
            value = (i + 1) % 2
            # fill the vertical timing
            sync_dm_array[i, timing_col_idx] = value

        for j in range(cols):
            if timing_col_idx == 0:
                value = (j + 1) % 2
            else:
                value = j % 2
            # fill the horizontal timing
            sync_dm_array[0, j] = value

    else:
        # top is Finder pattern
        sync_dm_array[0, :] = 0
        if vertical_timing_pos == 'left':
            timing_col_idx = 0
            # right is Finder pattern
            sync_dm_array[:, -1] = 0

        if vertical_timing_pos == 'right':
            timing_col_idx = -1
            # left is Finder pattern
            sync_dm_array[:, 0] = 0

        for i in range(rows):
            value = i % 2
            # fill the vertical timing
            sync_dm_array[i, timing_col_idx] = value

        for j in range(cols):
            if timing_col_idx == -1:
                value = j % 2
            else:
                value = (j + 1) % 2
            # fill the horizontal timing
            sync_dm_array[-1, j] = value
    return sync_dm_array

def sync_dm_code(sync_dm_array, sync_cell_width, border_width):
    """
    根据DM码矩阵（一个cell一个像素）生成用于解码的DM码
    :param sync_dm_array: DM码矩阵（一个cell一个像素）
    :param sync_cell_width: 每个cell的宽
    :param border_width: 外圈白边的宽
    :return: 最终用于解码的DM码
    """

    # 生成最终解码使用的合成DM码
    rows, cols = sync_dm_array.shape[0], sync_dm_array.shape[1]
    sync_dm_code = np.zeros([rows * sync_cell_width, cols * sync_cell_width])
    for i in range(rows):
        for j in range(cols):
            if sync_dm_array[i, j] == 1:
                sync_dm_code[i * sync_cell_width:(i + 1) * sync_cell_width,
                j * sync_cell_width:(j + 1) * sync_cell_width] = 255

    # 添加白边
    sync_dm_code_padded = cv2.copyMakeBorder(
        src=sync_dm_code,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_CONSTANT,
        value=255  # 边框颜色设为白色（与二值化图像的白色保持一致）
    )
    return sync_dm_code_padded

def rotate_image(img, angle, return_H = False):
    """
    旋转图片：symv里有这个算子
    :param img: 图片
    :param angle: 旋转角度（degree）
    :param return_H: 旋转矩阵
    :return: 旋转后的图片
    """
    if angle != 0:

        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        H = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        vertices = np.array([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]])
        rotated_vertices = np.hstack([vertices, np.ones([vertices.shape[0], 1])]) @ H.T
        min_x, min_y = np.min(rotated_vertices, axis=0)
        if abs(angle) == 90 or abs(angle) == 270:
            h, w = w, h
        H[:, 2] -= [min_x, min_y]

        rotated_img = cv2.warpAffine(img, H, (int(w), int(h)))
    else:
        rotated_img = img
    if return_H:
        return rotated_img, H
    else:
        return rotated_img

def crop_rotated_rect(image, x, y , w, h, angle):
    """
    从图片中扣出任意旋转矩形框：symv里有这个算子
    :param img: 图片
    :param x, y , w, h, angle: 旋转框的中心点，长款和角度（degree）
    :return: 抠出后的图片
    """
    # generate rotated rectangle patch with w, h centered at instance's center (x,y) in CAD

    src_pts = cv2.boxPoints(((x, y), (w, h), angle)).astype("float32")

    # coordinate of the points in box points after the rectangle has been flatten
    dst_pts = np.array([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]], dtype="float32")

    # the perspective transformation matrix
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the flatten rectangle
    # crop out the image patch of size w x h
    warped_image_patch = cv2.warpPerspective(image, H, (int(w), int(h)), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return warped_image_patch

if __name__ == '__main__':
    print(">>> DEBUG START <<<")

    # ==== 数据文件夹路径 ======
    task_folder = '/home/cat/workspace/DMCODE/SNcode/'
    # data_folder = os.path.join(task_folder, 'DM_code_data_part_*')
    data_folder = os.path.join(task_folder, 'DM_code_data_part_3_*')
    image_format = '.png'
    # 所有DM码图片路径
    image_path_list = glob.glob(f'{data_folder}/*/*{image_format}')
    # 用于存bad cases图片的路径
    bad_cases_folder = os.path.join(task_folder, 'badcases')
    os.makedirs(bad_cases_folder, exist_ok=True)
    # 读取预处理参数
    image_meta_info_df = pd.read_csv(os.path.join(task_folder, 'combined_DM_code_data3.csv'))

    # ==== 参数预设 ======
    # 所有可能的DM码尺寸组合
    valid_dm_dimensions_square = [[r, r] for r in range(10, 146, 2)]
    valid_dm_dimensions_rectangle = [[r, c] for r, c in zip([8, 8, 12, 12, 16, 16],
                                                            [18, 32, 26, 36, 36, 48])]
    valid_dm_dimensions = valid_dm_dimensions_square + valid_dm_dimensions_rectangle

    # 单个cell内用于判断cell是白还是黑的区域及阈值
    kernel_size = 2
    ratio_threshold = 0.0
    # 生成DM码的cell宽度和外边padding尺寸
    sync_cell_width = 5
    border_width = 10
    # action: nopreprocess不做后处理，直接用libdmtx检测 或 sync_dm_code_search 我们合成的方法
    action = 'sync_dm_code_search'
    # 阶段stage: programming 编程阶段 或 inspection 检测阶段
    stage = 'programming'

    # ==== 结果统计信息 =====
    counter = 0
    save_image = True
    t1 = time.time()
    failed_list = []
    image_meta_info_aug_list = []
    # failed_list_tmpt = [0, 20, 36, 37, 38, 42, 43, 46, 53, 75, 76, 120, 121, 145, 147, 168]

    for id, image_path in enumerate(image_path_list):
        # if id not in forailed_list_tmpt:
        #     continue
        # 读取图片
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        image_name = image_path.split('/')[-1]
        # if image_name != '3c7cca23-fa96-4dc2-8589-13c0a45f76eb_white_BARCODE_101_BARCODE_101.png':
        #     continue
        # 读取标注
        anno_path = image_path.replace(image_format, '.json')
        annotation_json = json.load(open(anno_path))
        bbox = np.array(annotation_json['annotations']['bbox']).astype(np.int32)
        angle = annotation_json['annotations']['attributes']['rotation']
        cx, cy, w1, h1 = bbox

        # 读取前处理参数
        # 编程阶段Programming: 参数由默认config设定
        # 检测阶段Inspection: 参数读取金版编程阶段设定的参数
        image_meta_info = image_meta_info_df[image_meta_info_df['image_name'] == image_name].squeeze()
        dm_color = image_meta_info['finder_color']
        threshold = image_meta_info['threshold']
        # threshold = 50
        # 编程阶段，根据检测框外扩做精定位的外扩尺寸
        expand =  image_meta_info['expand']
        # 前处理用的形态学参数
        morph_op = image_meta_info['morph_op']
        morph_iters = image_meta_info['morph_iters']
        # 用于判断哪两边是timing pattern的区域设定
        band_width = 3
        band_offset = expand + 3

        if action == 'nopreprocess':
            # 原始的DM码解码链路
            expand = 10
            # 抠出DM码
            sync_dm_code_with_padding = crop_rotated_rect(image, int(cx), int(cy), int(w1 + expand),
                                         int(h1 + expand), angle)

        elif action.startswith('sync_dm_code'):

            # == 抠出DM码 ===
            if data_folder == '/mnt/c/Shiyuan/data/NonDL/SNcode/DM_code_data_part_1_crop_data':
                dm_image = crop_rotated_rect(image,int(cx), int(cy), int(w1 + expand),
                                           int(h1 + expand), angle)

            else:
                dm_image = crop_rotated_rect(image, int(cx), int(cy), int(w1 + expand),
                                           int(h1 + expand), angle)
            # === 预处理DM码图片 ===
            # dm_color, resize_ratio, threshold, morph_op, morph_iter都是可以人为设定的参数
            # 确保长方形DM码，高是短边
            h1, w1 = dm_image.shape[:2]
            if h1/w1 > 1.5:
                dm_image = rotate_image(dm_image, 90)
            # 对太大的DM码图片，resize
            if np.min([h1, w1]) > 500:
                resize_ratio = 2
                dm_image = cv2.resize(dm_image, (int(w1 / resize_ratio), int(h1 / resize_ratio)),
                                      interpolation=cv2.INTER_LINEAR)
            # 确保DM码的FinderPattern是深色的，对于白色的DM码，要反转图片像素值
            if dm_color == 'white':
                dm_image = 255 - dm_image
            if threshold == -1 and stage == 'programming':
                # 真实使用时就是金版编程阶段
                threshold, binary_dm_image = cv2.threshold(dm_image, 0, 255, cv2.THRESH_OTSU)
                binary_dm_image = cv2.morphologyEx(binary_dm_image, morph_op, np.ones((3, 3)), iterations=morph_iters)


                # 根据检测框，做二次精定位
                x2, y2, w2, h2, angle = refine_dm_bbox(binary_dm_image, expand, rotated=False)
                # binary_dm_image = crop_rotated_rect(binary_dm_image, x2, y2, w2, h2, angle )
                # dm_image = crop_rotated_rect(dm_image, x2, y2, w2, h2, angle )

                binary_dm_image = binary_dm_image[y2:y2 + h2, x2: x2 + w2]
                dm_image = dm_image[y2:y2 + h2, x2: x2 + w2]
            else:
                # 真实使用时就是检测阶段
                # dm_image = cv2.Gaussian
                # dm_image = cv2.GaussianBlur(dm_image, (3, 3), 0)
                threshold, binary_dm_image = cv2.threshold(dm_image, threshold, 255, cv2.THRESH_BINARY)
                binary_dm_image = cv2.morphologyEx(binary_dm_image, morph_op, np.ones((3, 3)), iterations=morph_iters)
                x2, y2, w2, h2, angle = refine_dm_bbox(binary_dm_image, expand, rotated=False)
                binary_dm_image = binary_dm_image[y2:y2 + h2, x2: x2 + w2]
                dm_image = dm_image[y2:y2 + h2, x2: x2 + w2]

            # === 判断Timing pattern的位置及DM码cells的行列数 ===
            # 判断Timing pattern的位置
            try:
                horizontal_timing, vertical_timing = locate_timing_pattern(binary_dm_image, band_offset, band_width)
            except:
                counter += 1
                failed_list.append(id)
                print(id, 'locate timing pattern failed')
                continue
            # 搜索DM码有多少行和多少列cells
            try:
                height, width = binary_dm_image.shape[:2]
                if 0.5 < height/width < 2:
                    # 正方形DM码
                    candidate_dm_dimensions = valid_dm_dimensions_square
                else:
                    # 长方形DM码，cells的行列数都超过2
                    candidate_dm_dimensions = valid_dm_dimensions_rectangle
                rows, cols= search_dm_dimensions(dm_image, candidate_dm_dimensions,
                                                 timing_pattern_pos=[horizontal_timing, vertical_timing])
            except:
                counter += 1
                failed_list.append(id)
                print(id, 'dm_search_failed')
                continue

            # === 确定DM码每个cell的颜色, 生成最终解码的DM码  ===
            # 确定DM码每个cell的颜色, 输出一个DM码矩阵
            sync_dm_array, binary_dm_image_show = compute_dm_array(binary_dm_image, rows, cols, kernel_size, ratio_threshold, save_image)
            # 纠正最外圈的timing pattern and finder pattern
            sync_dm_array = correct_timing_finder_pattern(sync_dm_array, horizontal_timing, vertical_timing)
            # 生成最终使用的DM码图片
            sync_dm_code_with_padding = sync_dm_code(sync_dm_array, sync_cell_width, border_width)

                    # 统一一个文件名前缀，方便把三张图对应起来
            base_name = os.path.splitext(image_name)[0]  # 去掉.png
            prefix = f"{id}_{base_name}"

            meta = {
                "rows": int(rows),
                "cols": int(cols),
                "sync_cell_width": int(sync_cell_width),
                "border_width": int(border_width),
            }
            with open(os.path.join(bad_cases_folder, f"{prefix}_meta.json"), "w") as f:
                json.dump(meta, f)

            if save_image:
                np.save(os.path.join(bad_cases_folder, f"{prefix}_sync_dm_array.npy"), sync_dm_array)

                
                     # ① 预处理后的灰度 DM 图（裁剪+旋转后）
                # cv2.imwrite(os.path.join(bad_cases_folder, f'{id}_image_w{image.shape[1]}h{image.shape[0]}.jpg'), image)
                cv2.imwrite(os.path.join(bad_cases_folder, f'{prefix}_dm_image_w{dm_image.shape[1]}h{dm_image.shape[0]}.jpg'), dm_image)
                     # ② 像素级二值图（阈值 + 形态学之后的 0/255）
                cv2.imwrite(os.path.join(bad_cases_folder, f'{prefix}_binary_dm_image_w{binary_dm_image.shape[1]}h{binary_dm_image.shape[0]}.jpg'), binary_dm_image)
                    # ③ 带绿色网格的二值图（方便肉眼看 cell 划分）
                cv2.imwrite(os.path.join(bad_cases_folder, f'{prefix}_binary_dm_image_rect_w{binary_dm_image.shape[1]}h{binary_dm_image.shape[0]}.jpg'), binary_dm_image_show)
                    
                    # ④ 用 sync_dm_array 重建的“标准 DM 图”
                cv2.imwrite(os.path.join(bad_cases_folder, f'{prefix}_sync_dm_code_w{sync_dm_code_with_padding.shape[1]}h{sync_dm_code_with_padding.shape[0]}.jpg'), sync_dm_code_with_padding)

        # 用libdmtx解码
        decoded_res = decode(sync_dm_code_with_padding, max_count=1)

        # 打印及统计解码结果
        if len(decoded_res) == 0:
            # 如果解码结果为空
            counter += 1
            failed_list.append(id)
            print(image_name)
            continue

        else:
            obj = decoded_res[0]
            content = obj.data.decode('utf-8')
            clean_content = re.sub(r'[\x00-\x1f]', '', content)
            encoded = encode(clean_content.encode('utf-8'))  # Encode to Data Matrix
            encoded_img = Image.frombytes('RGB', (encoded.width, encoded.height), encoded.pixels)
            
            if save_image:
                encoded_img = encoded_img.rotate(90)
                    # 保存libdmtx编码生成的DM码图片，方便对比
                encoded_img.save(
                    os.path.join(bad_cases_folder, f"{prefix}_encoder_dm.jpg")
                )
                                # === 把 encoder 图也转成 cell 级矩阵 ===
                # 1) PIL 转 numpy 灰度图
                enc_gray = np.array(encoded_img.convert("L"))

                # 2) 简单二值化（阈值可以用 OTSU）
                _, enc_binary = cv2.threshold(enc_gray, 0, 255, cv2.THRESH_OTSU)

                # 3) 去掉外围全白边（找最大连通区域）
                x_e, y_e, w_e, h_e, _ = refine_dm_bbox(enc_binary, expand=0, rotated=False)
                enc_symbol = enc_binary[y_e:y_e + h_e, x_e:x_e + w_e]

                # 4) 用 *同样的* rows / cols 切格子，得到 encoder 的 cell 矩阵
                encoder_dm_array, _ = compute_dm_array(
                    enc_symbol,
                    rows,              # 和 sync_dm_array 一样的行数
                    cols,              # 和 sync_dm_array 一样的列数
                    kernel_size=2,     # 你可以用同一个 kernel_size
                    ratio_threshold=0.5,  # 这里可以设 0.5，让“白像素比例 > 0.5”认为是 1
                    save_image=False
                )

                # 5) 保存 encoder 的 cell 级矩阵
                np.save(os.path.join(bad_cases_folder, f"{prefix}_encoder_dm_array.npy"), encoder_dm_array)

            # visualize(binary_dm_image)
            # visualize(sync_dm_code_with_padding)
            # visualize(encoded_img.rotate(90))

        # print(id,decoded_res)
        image_meta_info['detection_res'] = True
        image_meta_info_aug_list.append(image_meta_info)

    t2 = time.time()

    # if not save_image:
    #     all_image_df = pd.DataFrame(image_meta_info_aug_list)
    #     all_image_df.to_csv(os.path.join(task_folder, data_folder.split('/')[-1]+'.csv'))
    print(f'{action}-{kernel_size}-{ratio_threshold}: success rate = {1 - counter/len(image_path_list)}, time cost = {(t2-t1)/len(image_path_list)}')

