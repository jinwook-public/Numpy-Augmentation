import random
import numpy as np
from PIL import Image

'''
alpha = [1, 2, 3, 4 ,5]
alpha[::-1] = [5, 4, 3, 2, 1]
'''

def VerticalFlip(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)  # (187, 270, 3)
    img_vertical = img_array[::-1] # np.flip(img_array, axis=0)
    img_vertical = Image.fromarray(img_vertical)
    img_vertical.show()
# VerticalFlip('./puppy.jpg')

def HorizontalFlip(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)  # (187, 270, 3)
    img_horizon = img_array[:,::-1] # np.flip(img_array, axis=1)
    img_horizon = Image.fromarray(img_horizon)
    img_horizon.show()
# HorizontalFlip('./puppy.jpg')

def GrayScale(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)  # (187, 270, 3)
    img_gray = img_array[:,:,0]
    img_gray = Image.fromarray(img_gray)
    img_gray.show()
# GrayScale('./puppy.jpg')

def RandomShif(img_path, offset, fill):
    img = Image.open(img_path)
    img_array = np.array(img)  # (187, 270, 3)
    img_x = img_array.shape[1]
    img_y = img_array.shape[0]

    x = random.randrange(-2, 2) # [-2, -1, 0, 1]
    y = random.randrange(-2, 2)

    print('x: ',x, 'y: ', y)
    r_img = img_array[:, :, 0]
    g_img = img_array[:, :, 1]
    b_img = img_array[:, :, 2]

    row_r_add = np.full(shape=(offset, r_img.shape[:2][1]), fill_value=fill)
    row_g_add = np.full(shape=(offset, g_img.shape[:2][1]), fill_value=fill)
    row_b_add = np.full(shape=(offset, b_img.shape[:2][1]), fill_value=fill)

    if y >= 0:
        r_img = np.vstack((r_img, row_r_add))  # (187, offset+270)
        g_img = np.vstack((g_img, row_g_add))
        b_img = np.vstack((b_img, row_b_add))

    else:
        r_img = np.vstack((row_r_add, r_img))  # (187, 270+offset)
        g_img = np.vstack((row_g_add, g_img))
        b_img = np.vstack((row_b_add, b_img))

    col_r_add = np.full(shape=(r_img.shape[:2][0], offset), fill_value=fill)
    col_g_add = np.full(shape=(g_img.shape[:2][0], offset), fill_value=fill)
    col_b_add = np.full(shape=(b_img.shape[:2][0], offset), fill_value=fill)

    if x >= 0:
        r_img = np.hstack((col_r_add, r_img))  # (offset+187, 270)
        g_img = np.hstack((col_g_add, g_img))
        b_img = np.hstack((col_b_add, b_img))
    else:
        r_img = np.hstack((r_img, col_r_add))  # (187+offset, 270+depth)
        g_img = np.hstack((g_img, col_g_add))
        b_img = np.hstack((b_img, col_b_add))

    img_aug = np.dstack((r_img, g_img, b_img))  # (187+offset, 270+offset, 3)

    if x>=0:
        if y>=0: # (+,+) (offset+187, offset+270)
            img_shfit = img_aug[offset:, :img_x]  # (187, 270, 3)
        else: # (+, -) (offset+187, 270+offset)
            img_shfit = img_aug[:img_y, :img_x]
    else:
        if y>=0: # (-,+) (187+offset, offset+270)
            img_shfit = img_aug[offset:, offset:]
        else: # (-, -) (187+offset, 270+offset)
            img_shfit = img_aug[:img_y, offset:]

    img_shfit = Image.fromarray(img_shfit.astype(np.uint8))
    img_shfit.show()
# RandomShif(img_path='./puppy.jpg', offset= 75, fill =255)

def RandomCrop(img_path, crop_size, fill=None, padding=None, padding_mode='constant'):
    img = Image.open(img_path)
    img_array = np.array(img)  # (187, 270, 3)
    img_x = img_array.shape[1]
    img_y = img_array.shape[0]



    if padding == None:
        rand_x = random.randrange(0, img_x - crop_size[0])
        rand_y = random.randrange(0, img_y - crop_size[1])


        crop_area = img_array[rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[0]]
        crop_area = Image.fromarray(crop_area)
        crop_area.show()

    else:
        rand_x = random.randrange(0, img_x+2*padding-crop_size[0])
        rand_y = random.randrange(0, img_y+2*padding-crop_size[1])

        assert (rand_x+crop_size[0]) <= (img_x + (2*padding)), 'Crop_size is Out of X range'
        assert (rand_y+crop_size[0]) <= (img_y + (2*padding)), 'Crop_size is Out of Y range'


        if padding_mode=='constant':
            r_img = img_array[:, :, 0]
            g_img = img_array[:, :, 1]
            b_img = img_array[:, :, 2]

            row_r_add = np.full(shape=(padding, r_img.shape[:2][1]), fill_value=fill)
            row_g_add = np.full(shape=(padding, g_img.shape[:2][1]), fill_value=fill)
            row_b_add = np.full(shape=(padding, b_img.shape[:2][1]), fill_value=fill)

            r_img = np.vstack((r_img, row_r_add))
            g_img = np.vstack((g_img, row_g_add))
            b_img = np.vstack((b_img, row_b_add))

            r_img = np.vstack((row_r_add, r_img))
            g_img = np.vstack((row_g_add, g_img))
            b_img = np.vstack((row_b_add, b_img))

            col_r_add = np.full(shape=(r_img.shape[:2][0], padding), fill_value=fill)
            col_g_add = np.full(shape=(g_img.shape[:2][0], padding), fill_value=fill)
            col_b_add = np.full(shape=(b_img.shape[:2][0], padding), fill_value=fill)

            r_img = np.hstack((col_r_add, r_img))
            g_img = np.hstack((col_g_add, g_img))
            b_img = np.hstack((col_b_add, b_img))

            r_img = np.hstack((r_img, col_r_add))
            g_img = np.hstack((g_img, col_g_add))
            b_img = np.hstack((b_img, col_b_add))

            img_aug = np.dstack((r_img, g_img, b_img)).astype(np.uint8)

            print(img_aug.shape)

            crop_area = img_aug[rand_y:rand_y + crop_size[1], rand_x:rand_x + crop_size[0]]
            crop_area = Image.fromarray(crop_area)
            crop_area.show()

        elif padding_mode =='edge':
            r_img = img_array[:, :, 0]
            g_img = img_array[:, :, 1]
            b_img = img_array[:, :, 2]

            row_r_add = np.full(shape=(padding, r_img.shape[:2][1]), fill_value=r_img[-1,:])
            row_g_add = np.full(shape=(padding, g_img.shape[:2][1]), fill_value=g_img[-1,:])
            row_b_add = np.full(shape=(padding, b_img.shape[:2][1]), fill_value=b_img[-1,:])

            r_img = np.vstack((r_img, row_r_add))
            g_img = np.vstack((g_img, row_g_add))
            b_img = np.vstack((b_img, row_b_add))

            row_r_add = np.full(shape=(padding, r_img.shape[:2][1]), fill_value=r_img[0,:])
            row_g_add = np.full(shape=(padding, g_img.shape[:2][1]), fill_value=g_img[0,:])
            row_b_add = np.full(shape=(padding, b_img.shape[:2][1]), fill_value=b_img[0,:])

            r_img = np.vstack((row_r_add, r_img))
            g_img = np.vstack((row_g_add, g_img))
            b_img = np.vstack((row_b_add, b_img))

            r_left = r_img[:, 0]
            g_left = g_img[:,0]
            b_left = b_img[:, 0]

            r_right = r_img[:, -1]
            g_right = g_img[:,-1]
            b_right = b_img[:, -1]

            r_left = np.array(r_left)[:, np.newaxis]
            g_left = np.array(g_left)[:, np.newaxis]
            b_left = np.array(b_left)[:, np.newaxis]

            r_right = np.array(r_right)[:, np.newaxis]
            g_right = np.array(g_right)[:, np.newaxis]
            b_right = np.array(b_right)[:, np.newaxis]

            col_r_add = np.full(shape=(r_img.shape[:2][0], padding), fill_value=r_left)
            col_g_add = np.full(shape=(g_img.shape[:2][0], padding), fill_value=g_left)
            col_b_add = np.full(shape=(b_img.shape[:2][0], padding), fill_value=b_left)

            r_img = np.hstack((col_r_add, r_img))
            g_img = np.hstack((col_g_add, g_img))
            b_img = np.hstack((col_b_add, b_img))

            col_r_add = np.full(shape=(r_img.shape[:2][0], padding), fill_value=r_right)
            col_g_add = np.full(shape=(g_img.shape[:2][0], padding), fill_value=g_right)
            col_b_add = np.full(shape=(b_img.shape[:2][0], padding), fill_value=b_right)

            r_img = np.hstack((r_img, col_r_add))
            g_img = np.hstack((g_img, col_g_add))
            b_img = np.hstack((b_img, col_b_add))

            img_aug = np.dstack((r_img, g_img, b_img))
            crop_area = img_aug[rand_y:rand_y + crop_size[1], rand_x:rand_x + crop_size[0]]
            crop_area = Image.fromarray(crop_area)
            crop_area.show()

# RandomCrop(img_path='./puppy.jpg', crop_size=(100,100),fill=125, padding=30, padding_mode='constant')

def RandomBlur(img_path, filter_size, mode='Average'):
    img = Image.open(img_path)
    img_array = np.array(img)  # (187, 270, 3)
    img_x = img_array.shape[1]
    img_y = img_array.shape[0]

    r_img = img_array[:, :, 0]
    g_img = img_array[:, :, 1]
    b_img = img_array[:, :, 2]

    copyImg_r = img_array.copy()[:, :, 0]
    copyImg_g = img_array.copy()[:, :, 1]
    copyImg_b = img_array.copy()[:, :, 2]

    half_filter = int(filter_size / 2)
    startRow = half_filter
    startCol = half_filter

    if mode == 'Average':
        for row in range(startRow, img_y - half_filter):
            for col in range(startCol, img_x - half_filter):
                localPixels_r = r_img[row - half_filter:row + half_filter + 1, col - half_filter:col + half_filter + 1]
                localPixels_g = g_img[row - half_filter:row + half_filter + 1, col - half_filter:col + half_filter + 1]
                localPixels_b = b_img[row - half_filter:row + half_filter + 1, col - half_filter:col + half_filter + 1]

                blurredValue_r = np.mean(localPixels_r)
                blurredValue_g = np.mean(localPixels_g)
                blurredValue_b = np.mean(localPixels_b)

                copyImg_r[row, col] = blurredValue_r
                copyImg_g[row, col] = blurredValue_g
                copyImg_b[row, col] = blurredValue_b

    elif mode == 'Median':
        for row in range(startRow, img_y - half_filter):
            for col in range(startCol, img_x - half_filter):
                localPixels_r = r_img[row - half_filter:row + half_filter + 1, col - half_filter:col + half_filter + 1]
                localPixels_g = g_img[row - half_filter:row + half_filter + 1, col - half_filter:col + half_filter + 1]
                localPixels_b = b_img[row - half_filter:row + half_filter + 1, col - half_filter:col + half_filter + 1]

                blurredValue_r = np.median(localPixels_r)
                blurredValue_g = np.median(localPixels_g)
                blurredValue_b = np.median(localPixels_b)

                copyImg_r[row, col] = blurredValue_r
                copyImg_g[row, col] = blurredValue_g
                copyImg_b[row, col] = blurredValue_b

    img_aug = np.dstack((copyImg_r, copyImg_g, copyImg_b))
    img_aug = Image.fromarray(img_aug.astype(np.uint8))
    img_aug.show()
# RandomBlur('./puppy.jpg', filter_size=7, mode='Median')

