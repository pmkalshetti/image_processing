import numpy as np


def resize(img, scale):
    """Resize image using scale.

    Arguments
    ---------
    img : np.ndarray; (height_img, width_img); np.uint8
        Grayscale image to resize.
    scale : float
        Scale factor.

    Returns
    -------
    img_resized : np.ndarray; (scale*height_img, scale*width_img); np.uint8
        Resized grayscale image.
    """

    def get_bilinear_intensity(img, y, x):
        """Returns intensity of img at [y, x] using bilinear interpolation.

        Arguments
        ---------
        img : np.ndarray; (height_img, width_img); np.uint8
            Grayscale image to resize.
        y, x : float
            Coordinates in image where intensity is to be computed.
        
        Returns
        -------
        intensity_interpolated : int
            computed interpolated intensity value at [y,x].
        """
        # extract corner coordinates
        y_top = int(y)  # top in image representation is low in y value
        x_left = int(x)
        y_bottom = min(y_top+1, img.shape[0]-1)
        x_right = min(x_left+1, img.shape[1]-1)

        # extract fractional component
        y_frac = y - y_top
        x_frac = x - x_left

        # pixel values at corner
        intensity_top_left = img[y_top, x_left]
        intensity_top_right = img[y_top, x_right]
        intensity_bottom_left = img[y_bottom, x_left]
        intensity_bottom_right = img[y_bottom, x_right]

        # bilinear interpolation of intensity values based on fractional components
        intensity_interpolated_top = intensity_top_left * (1-x_frac) \
            + intensity_top_right * x_frac
        intensity_interpolated_bottom = intensity_bottom_left * (1-x_frac) \
            + intensity_bottom_right * x_frac
        intensity_interpolated = intensity_interpolated_top * (1-y_frac) \
            + intensity_interpolated_bottom * y_frac

        return intensity_interpolated

    # create empty resized image (shape clipped to int)
    height_img_resized = int(scale*img.shape[0])
    width_img_resized = int(scale*img.shape[1])
    img_resized = np.empty([height_img_resized, width_img_resized],
                           dtype=np.uint8)

    # scaling factor across height and width
    scale_height = img.shape[0] / img_resized.shape[0]
    scale_width = img.shape[1] / img_resized.shape[1]

    # compute intensity at each pixel
    for idx_row_resized in range(height_img_resized):
        for idx_col_resized in range(width_img_resized):

            # compute position of resized pixel in original image
            y = idx_row_resized * scale_height
            x = idx_col_resized * scale_width

            # fill value using bilinear interpolation
            img_resized[idx_row_resized, idx_col_resized] = get_bilinear_intensity(
                img, y, x)

    return img_resized
