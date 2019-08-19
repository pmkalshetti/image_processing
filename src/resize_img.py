import cv2 as cv
import numpy as np
import argparse
from lib.interpolation.bilinear import resize


def read_args():

    """Parses arguments from cmd.

    Returns
    -------
    args : argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-pi", "--path_img",
                        default="meta_data/white_rose.jpg",
                        help="path to input image.")
    parser.add_argument("-s", "--scale",
                        type=float, default=2,
                        help="scale factor for resizing.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # read command arguments
    args = read_args()

    # load image
    img = cv.imread(args.path_img, cv.IMREAD_GRAYSCALE)

    # resize using bilinear interpolation
    img_resized = resize(img, args.scale)

    # visualize
    cv.namedWindow("original image", cv.WINDOW_AUTOSIZE)
    cv.moveWindow("original image", 10, 10)
    cv.imshow("original image", img)
    cv.imwrite("img_original.png", img)

    cv.namedWindow("resized image", cv.WINDOW_AUTOSIZE)
    cv.moveWindow("resized image", 200, 10)
    cv.imshow("resized image", img_resized)
    cv.imwrite("img_resized.png", img_resized)

    cv.waitKey(0)
    cv.destroyAllWindows()
