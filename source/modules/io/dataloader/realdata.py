import os
import re
import cv2
import glob
import logging
import numpy as np

logger = logging.getLogger(__name__)

class dataloader():
    def __init__(self, numberOfImages = None, outdir = '.'):
        self.numberOfImages = numberOfImages
        self.outdir = outdir
        self.data_workspace = outdir  # 初始化默认值
        self.objname = ''

    def img_tile(self, imgs, rows, cols, outdir): # [N, h, w, c]
        n, h, w, c = np.shape(imgs)
        if rows * cols <= n:
            img_tiled = []
            for i in range(cols):
                temp = np.reshape(imgs[rows*i:rows*i+rows,:,:,:], (-1, w, 3))
                img_tiled.append(temp)
            img_tiled = np.concatenate(img_tiled, axis = 1)
            os.makedirs(outdir, exist_ok=True)
            cv2.imwrite(f'{outdir}/tiled.png', (255 * img_tiled[:,:,::-1]).astype(np.uint8))

    def merge_img(self, imgs, merge_num): # [N, h*w, 3]
        imgs_merged = np.zeros(imgs.shape, np.float32)
        for k in range(imgs.shape[0]):
            ids = np.random.permutation(imgs.shape[0])[:merge_num]
            img = imgs[ids, :, :]
            img = np.sum(img, axis=0)
            imgs_merged[k, :, :] = img
        return imgs_merged

    def psfcn_normalize(self, imgs): # [NLight, H, W ,C]
        h, w, c = imgs[0].shape
        imgs = [img.reshape(-1, 1) for img in imgs]
        img = np.hstack(imgs)
        norm = np.sqrt((img * img).clip(0.0).sum(1))
        img = img / (norm.reshape(-1,1) + 1e-10)
        imgs = np.split(img, img.shape[1], axis=1)
        imgs = [img.reshape(h, w, -1) for img in imgs]
        print('PSFCN_NORMALIZED')
        return imgs

    def _normalize_image(self, img):
        if img.dtype == np.uint8:
            bit_depth = 255.0
        elif img.dtype == np.uint16:
            bit_depth = 65535.0
        else:
            bit_depth = 1.0
        return np.float32(img) / bit_depth

    def _compute_crop_bounds(self, mask, image_shape, margin):
        rows, cols = np.nonzero(mask)
        if len(rows) == 0 or len(cols) == 0:
            return None

        rowmin = np.min(rows)
        rowmax = np.max(rows)
        colmin = np.min(cols)
        colmax = np.max(cols)
        row = rowmax - rowmin
        col = colmax - colmin

        if rowmin - margin <= 0 or rowmax + margin > image_shape[0] or colmin - margin <= 0 or colmax + margin > image_shape[1]:
            return None

        if row > col:
            row_start = rowmin - margin
            row_end = rowmax + margin
            col_start = np.max([colmin - int(0.5 * (row - col)) - margin, 0])
            col_end = np.min([colmax + int(0.5 * (row - col)) + margin, image_shape[1]])
        else:
            row_start = np.max([rowmin - int(0.5 * (col - row)) - margin, 0])
            row_end = np.min([rowmax + int(0.5 * (col - row)) + margin, image_shape[0]])
            col_start = colmin - margin
            col_end = colmax + margin

        return row_start, row_end, col_start, col_end

    def _crop_and_resize(self, arr, crop_bounds, size, interpolation):
        if crop_bounds is not None:
            row_start, row_end, col_start, col_end = crop_bounds
            if arr.ndim == 2:
                arr = arr[row_start:row_end, col_start:col_end]
            else:
                arr = arr[row_start:row_end, col_start:col_end, :]
        return cv2.resize(arr, dsize=size, interpolation=interpolation)

    def _load_normal_map(self, normal_path, crop_bounds, size, mask):
        if not os.path.isfile(normal_path):
            logger.warning(f'Normal map not found at {normal_path}. Using zero normal map.')
            return np.zeros((size[1], size[0], 3), np.float32)

        normal_raw = cv2.imread(normal_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if normal_raw is None:
            raise RuntimeError(f'Failed to read normal map at {normal_path}')

        normal = cv2.cvtColor(normal_raw, cv2.COLOR_BGR2RGB)
        normal = self._crop_and_resize(normal, crop_bounds, size, cv2.INTER_NEAREST)
        normal = self._normalize_image(normal) * 2.0 - 1.0
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / np.clip(norm, 1e-6, None)
        normal *= mask[:, :, None]
        return normal.astype(np.float32)

    def load(self, objlist, objid,  prefix,  margin = 0, loader_imgsize = 256):

        self.objname = re.split(r'\\|/',objlist[objid])[-1]
        self.data_workspace = f'{self.outdir}/{self.objname}'
        os.makedirs(self.data_workspace, exist_ok=True)

        directlist = []
        [directlist.append(p) for p in glob.glob(objlist[objid] + '/%s' % prefix,recursive=True) if os.path.isfile(p)]
        directlist = sorted(directlist)

        if len(directlist) == 0:
            return False
        if os.name == 'posix':
            temp = directlist[0].split("/")
        if os.name == 'nt':
            temp = directlist[0].split("\\")
        img_dir = "/".join(temp[:-1])

        if self.numberOfImages is not None:
            indexset = np.random.permutation(len(directlist))[:self.numberOfImages]
        else:
            indexset = range(len(directlist))
        numberOfImages = len(indexset)

        imgsize = loader_imgsize
        output_size = (imgsize, imgsize)
        margin = 4
        crop_bounds = None

        mask_path = img_dir + '/mask.png'
        first_img_raw = cv2.imread(directlist[indexset[0]], flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if first_img_raw is None:
            raise RuntimeError(f'Failed to read image at {directlist[indexset[0]]}')
        first_img = cv2.cvtColor(first_img_raw, cv2.COLOR_BGR2RGB)
        h, w = imgsize, imgsize

        if os.path.isfile(mask_path):
            mask_raw = cv2.imread(mask_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if mask_raw is None:
                raise RuntimeError(f'Failed to read mask at {mask_path}')
            mask = (mask_raw > 0).astype(np.float32)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            crop_bounds = self._compute_crop_bounds(mask, first_img.shape[:2], margin)
            mask = self._crop_and_resize(mask, crop_bounds, output_size, cv2.INTER_NEAREST).astype(np.float32)
        else:
            mask = np.ones((h, w), np.float32)

        I = np.zeros((numberOfImages, h, w, 3), np.float32)
        for i, indexofimage in enumerate(indexset):
            img_path = directlist[indexofimage]
            img_raw = cv2.imread(img_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img_raw is None:
                raise RuntimeError(f'Failed to read image at {img_path}')
            img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            img = self._crop_and_resize(img, crop_bounds, output_size, cv2.INTER_NEAREST)
            img = self._normalize_image(img)
            I[i, :, :, :] = img

        normal_path = img_dir + '/Normal_gt.png'
        N = self._load_normal_map(normal_path, crop_bounds, output_size, mask)

        self.img_tile(I, 3, 3, self.data_workspace)

        I = np.reshape(I, (-1, h * w, 3))
        temp = np.mean(I[:, mask.flatten()==1,:], axis=2)
        mean = np.mean(temp, axis=1)
        I /= mean.reshape(-1, 1, 1)

        I = np.transpose(I, (1, 2, 0))
        I = I.reshape(h, w, 3, numberOfImages)
        mask = (mask.reshape(h, w, 1)).astype(np.float32)
        I = I * mask[:, :, :, np.newaxis]

        self.h = mask.shape[0]
        self.w = mask.shape[1]
        self.I = I
        self.N = N
        self.mask = mask
        logger.debug(f'Loaded data for object "{self.objname}": {numberOfImages} images of size ({self.h}, {self.w})')
