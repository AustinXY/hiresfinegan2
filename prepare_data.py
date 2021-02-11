import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
import numpy as np
import pickle


def resize_and_convert(img, mask, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    mask = trans_fn.resize(mask, size)
    mask = trans_fn.center_crop(mask, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()
    return [val, mask]


def resize_multiple(
    img, mask, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100
    ):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, mask, size, resample, quality))

    return imgs


class IMIM:
    def __init__(self, image, mask):
        '''
        image: byte
        '''
        self.image = image
        self.mask = mask
    def get_image_buffer(self):
        return BytesIO(self.image)


def get_mask(imshape, bbox):
    c, r = imshape
    bbox = [int(float(x)) for x in bbox.split(' ')]
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    x1 = x1
    y1 = y1
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(c-1, x2)
    y2 = min(r-1, y2)
    mask = np.zeros((r, c))
    mask[y1:y2+1, x1:x2+1] = 1
    return Image.fromarray(mask)


def resize_worker(img_file, sizes, resample, ib_dict):
    i, file = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    bbox = ib_dict[file]
    mask = get_mask(img.size, bbox)
    out = resize_multiple(img, mask, sizes=sizes, resample=resample)
    return i, out


def prepare(
    env, dataset, ib_dict, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS,
    ):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample, ib_dict=ib_dict)

    files = sorted(dataset.imgs, key=lambda x: x[0])

    files = [(i, file) for i, (file, _) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imims in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, imim in zip(sizes, imims):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")
                value = IMIM(imim[0], imim[1])
                with env.begin(write=True) as txn:
                    txn.put(key, pickle.dumps(value))

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))

def get_img_bbox_dict(dir_path):
    img_path = dir_path + '/images.txt'
    bbox_path = dir_path + '/bounding_boxes.txt'
    img_dir = dir_path + '/images/'
    ib_dict = {}
    with open(img_path, 'r') as imf:
        with open(bbox_path, 'r') as bbf:
            for l0 in imf:
                img = img_dir + l0.split(' ')[1].split('\n')[0]
                l1 = bbf.readline()
                bbox = ' '.join(l1.split(' ')[1:5]).split('\n')[0]
                ib_dict[img] = bbox
    return ib_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for model training")
    parser.add_argument("--out", type=str, help="filename of the result lmdb dataset")
    parser.add_argument(
        "--size",
        type=str,
        default="128,256,512,1024",
        help="resolutions of images for the dataset",
    )
    parser.add_argument(
        "--n_worker",
        type=int,
        default=8,
        help="number of workers for preparing dataset",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="lanczos",
        help="resampling methods for resizing images",
    )
    parser.add_argument("path", type=str, help="path to the image dataset")

    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)
    dir_path = args.path.split('/images')[0]
    ib_dict = get_img_bbox_dict(dir_path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, ib_dict, args.n_worker, sizes=sizes, resample=resample)
