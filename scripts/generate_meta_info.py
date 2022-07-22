import argparse
import cv2
import glob
import os
from tqdm import tqdm


def main(args):
    txt_file = open(args.meta_info, 'w')
    for folder, root in zip(args.input, args.root):
        # for d in data_list:
        #     d_path = os.path.join(data_dir, d)

        #     train_list = os.listdir(d_path)
        #     train_list = [t for t in train_list if os.path.isdir(os.path.join(d_path, t))]
        f_path=sorted(glob.glob(os.path.join(folder,'*')))
        f_path = [d for d in f_path if os.path.isdir(d)]
        for l_path in f_path:
            print(l_path)
            name_paths = sorted(glob.glob(os.path.join(l_path, '*')))
            for i_path in tqdm(name_paths):
                if "_multiscale" in i_path:
                    continue
                img_paths = sorted(glob.glob(os.path.join(i_path, '*')))
                img_paths = [file for file in img_paths if ".jpg" in file or ".png" in file]
                for i, img_path in enumerate(img_paths):
                    if i % 4 != 0: continue
                    status = True
                    # if args.check:
                    #     # read the image once for check, as some images may have errors
                    #     try:
                    #         img = cv2.imread(img_path)
                    #     except (IOError, OSError) as error:
                    #         print(f'Read {img_path} error: {error}')
                    #         status = False
                    #     if img is None:
                    #         status = False
                    #         print(f'Img is None: {img_path}')
                    if status:
                        # get the relative path
                        img_name = os.path.relpath(img_path, root)
                        # print(img_name)
                        txt_file.write(f'{img_name}\n')


if __name__ == '__main__':
    """Generate meta info (txt file) for only Ground-Truth images.

    It can also generate meta info from several folders into one txt file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['/vtca/datasets/Vtc'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['/vtca/datasets/Vtc'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/vtca/datasets/Vtc/meta_info.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)

    main(args)
