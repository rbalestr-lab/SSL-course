import torch
import torch.nn.functional as F
import numbers
import torchvision
from collections.abc import Sequence
from torchvision.transforms.functional import (
    _interpolation_modes_from_int,
    InterpolationMode,
)
from typing import List, Optional, Tuple, Union
import numpy as np

class AddParams(torch.nn.Module):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
            empty list to stack DA parameters
        """
        return img, []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.p})"


class RandomGrayscale(torch.nn.Module):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, inputs):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        img, params = inputs
        num_output_channels, _, _ = torchvision.transforms.functional.get_dimensions(
            img
        )
        if torch.rand(1) < self.p:
            return torchvision.transforms.functional.rgb_to_grayscale(
                img, num_output_channels=num_output_channels
            ), params + [1]
        return img, params + [0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


def _check_sequence_input(x, name, req_sizes):
    msg = (
        req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    )
    if not isinstance(x, Sequence):
        raise TypeError(f"{name} should be a sequence of length {msg}.")
    if len(x) not in req_sizes:
        raise ValueError(f"{name} should be a sequence of length {msg}.")


def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(f"If {name} is a single number, it must be positive.")
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


class RandomRotation(torch.nn.Module):
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number): Pixel fill value for the area outside the rotated
            image. Default is ``0``. If given a number, the value is used for all bands respectively.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(
        self,
        degrees,
        interpolation=InterpolationMode.NEAREST,
        expand=False,
        center=None,
        fill=0,
    ):
        super().__init__()

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        self.center = center

        self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(
            torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item()
        )
        return angle

    def forward(self, inputs):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        img, params = inputs
        fill = self.fill
        channels, _, _ = torchvision.transforms.functional.get_dimensions(img)
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)

        return torchvision.transforms.functional.rotate(
            img, angle, self.interpolation, self.expand, self.center, fill
        ), params + [angle]

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(degrees={self.degrees}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", expand={self.expand}"
        if self.center is not None:
            format_string += f", center={self.center}"
        if self.fill is not None:
            format_string += f", fill={self.fill}"
        format_string += ")"
        return format_string


class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, inputs):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        img, params = inputs
        if torch.rand(1) < self.p:
            return torchvision.transforms.functional.hflip(img), params + [1.0]
        return img, params + [0.0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class ColorJitter(torch.nn.Module):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non-negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0) -> None:
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )

    @torch.jit.unused
    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f"If {name} is a single number, it must be non negative."
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        else:
            raise TypeError(
                f"{name} should be a single number or a list/tuple with length 2."
            )

        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(
                f"{name} values should be between {bound}, but got {value}."
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            return None
        else:
            return tuple(value)

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = (
            None
            if brightness is None
            else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        )
        c = (
            None
            if contrast is None
            else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        )
        s = (
            None
            if saturation is None
            else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        )
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def forward(self, inputs):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        img, params = inputs

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = (
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = torchvision.transforms.functional.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = torchvision.transforms.functional.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = torchvision.transforms.functional.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = torchvision.transforms.functional.adjust_hue(img, hue_factor)

        return img, params + [
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ]

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue})"
        )
        return s


class ToTensor:
    """Convert a PIL Image or ndarray to tensor and scale the values accordingly.

    This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.

    .. _references: https://github.com/pytorch/vision/tree/main/references/segmentation
    """

    def __call__(self, inputs):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """

        if not isinstance(inputs, tuple):
            return torchvision.transforms.functional.to_tensor(inputs)
        return torchvision.transforms.functional.to_tensor(inputs[0]), np.asarray(inputs[1])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
