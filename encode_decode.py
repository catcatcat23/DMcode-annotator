from pylibdmtx.pylibdmtx import decode, encode
from PIL import Image, ImageEnhance
import cv2
from PIL import ImageEnhance
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import zxing

def visualize(img, title=''):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


data_folder = '/mnt/c/Shiyuan/data/DMcode/'
# img_path = os.path.join(data_folder, 'encoded_datamatrix1.png')
img_path = os.path.join(data_folder, 'ms_dm2.png')
img = Image.open(img_path)
img = img.convert('L')
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
reader = zxing.BarCodeReader()
# img = img.point(mode='imageops')
for angle in [0, 90, 180, 270]:
    print(f'angle={angle}')
    image_np = np.array(img.rotate(angle))
    threshold, dm_img = cv2.threshold(image_np, 0, 255, cv2.THRESH_OTSU)
    # dm_img = image_np
    # visualize(dm_img)

    # zxing_res = reader.decode(img_path)
    # if zxing_res:
    #     print("zxing data:", zxing_res[0].text)
    decoded_objects = decode(dm_img)
    # visualize(255 - img)
    if len(decoded_objects) == 0:
        continue

    for obj in decoded_objects:
        print("Decoded data:", obj.data)
        visualize(dm_img)
        content = obj.data.decode('utf-8')
        clean_content = re.sub(r'[\x00-\x1f]', '', content)
        encoded = encode(content.encode('utf-8'))  # Encode to Data Matrix
        encoded_img = Image.frombytes('RGB', (encoded.width, encoded.height), encoded.pixels)
        encoded_img.save( os.path.join(data_folder, 'encoded_datamatrix0.png'))

        decoded_objects2 = decode(encoded_img)
        print("Decoded data:", decoded_objects2[0].data)
        print(decoded_objects2[0].data == obj.data)
        visualize(encoded_img)
