import albumentations as A
import numpy as np


def brightness(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Alters the image brightness level by a factor defined by the user.

    Parameters
    ----------
    img : the source image to alter
    
    factor : values between 0 and 1 will decrease the brightness, whereas
    values above 1 will increase the brightness

    Returns
    -------
    the altered image
    """
    transform = A.ColorJitter(
        brightness=[factor, factor],
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
        always_apply=True,
    )

    return transform(image=img)["image"]


def downscale(img: np.ndarray, factor: float = 0.25) -> np.ndarray:
    """
    Downscale and upscale the image back to its original size, simulating low
    resolution images.

    Parameters
    ----------
    img : the source image to alter

    factor : values must be between 0 and 1, low values will make the altered image
    more pixelated

    Returns
    -------
    the altered image
    """
    transform = A.Downscale(
        scale_min=factor, 
        scale_max=factor, 
        always_apply=True
    )

    return transform(image=img)["image"]


def motion_blur(img: np.ndarray, factor: float = 13) -> np.ndarray:
    """
    Simulates motion blur on the image.

    Parameters
    ----------
    img : the source image to alter

    factor : the kernel size to use, needs to be greater than 3 and an odd number

    Returns
    -------
    the altered image
    """
    transform = A.MotionBlur(
        blur_limit=[factor, factor], 
        allow_shifted=True, 
        always_apply=True
    )

    return transform(image=img)["image"]


def rotation(img: np.ndarray, factor: float = 20) -> np.ndarray:
    """
    Rotates the image by (factor) degrees.

    Parameters
    ----------
    img : the source image to alter

    factor : the angle in degrees to rotate the image. Negative values will rotate
    in clockwise direction

    Returns
    -------
    the altered image
    """
    transform = A.Affine(
        rotate=[factor, factor], 
        always_apply=True, 
        cval=[int(img.mean())] * 3
    )

    return transform(image=img)["image"]


def translate(img: np.ndarray, factor: float = 0.5, axis: str = "x") -> np.ndarray:
    """
    Translate the image by (factor) in percent of pixels. A factor of 0.5 will translate
    the image to its halfpoint.

    Parameters
    ----------
    img : the source image to alter

    factor : the percent of pixels to translate, negative values will move the pixels to the left

    axis : the axis to perform the translation

    Returns
    -------
    the altered image
    """
    if axis == "x":
        translate_percent = {"x": [factor, factor], "y": [0, 0]}
    elif axis == "y":
        translate_percent = {"x": [0, 0], "y": [factor, factor]}

    transform = A.Affine(
        translate_percent=translate_percent,
        always_apply=True,
        cval=[int(img.mean())] * 3,
    )

    return transform(image=img)["image"]


def patches(img: np.ndarray, coordinates: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Apply patches to the image cutting out information. The patches are filled with the image
    mean pixel value. The patch is applied based on coordinates.

    Parameters
    ----------
    img : the source image to alter
    coordinates : a list containing the coordinates [x,y] to apply the patches

    width : the width of the patches

    height : the height of the patches

    Returns
    -------
    the altered image
    """
    img = img.copy()

    for x, y in coordinates:
        x1 = np.clip(x - width  // 2, 0, img.shape[1])
        y1 = np.clip(y - height // 2, 0, img.shape[0])
        x2 = np.clip(x + width  // 2, 0, img.shape[1])
        y2 = np.clip(y + height // 2, 0, img.shape[0])

        img[y1:y2, x1:x2] = int(img.mean())

    return img
