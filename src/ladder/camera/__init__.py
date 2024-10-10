from abc import abstractmethod, ABC
from typing import Tuple
from enum import Enum

import math
import taichi as ti
import taichi.math as tm
import numpy as np


class CameraMode(Enum):
    PERSPECTIVE = 1, 80.0 * (math.pi / 180.0)
    ORTHOGRAPHIC = 2, 80.0
    PHYSICAL = 3, 80.0

    def __new__(cls, value, data):
        obj = object.__new__(cls)
        obj._value_ = value
        # Add additional fields, with different name
        if value == 1:
            obj.fov_rad = data
        elif value == 2:
            obj.box_height = data
        elif value == 3:
            obj.focal_length = data
        else:
            raise ValueError("Invalid camera mode")

        return obj

    def set_fov_rad(self, fov_rad):
        if self.value == 1:
            self.fov_rad = fov_rad
        else:
            raise ValueError("Invalid camera mode")
        return self

    def set_box_height(self, box_height):
        if self.value == 2:
            self.box_height = box_height
        else:
            raise ValueError("Invalid camera mode")
        return self

    def set_focal_length(self, focal_length):
        if self.value == 3:
            self.focal_length = focal_length
        else:
            raise ValueError("Invalid camera mode")
        return self


class CameraSettings:
    def __init__(
        self,
        mode: CameraMode = CameraMode.PERSPECTIVE,
        near_plane: float = 0.1,
        far_plane: float = 10000,
        shutter_speed: float = 10.0,
        aperture: float = 4.0,
        iso: float = 100,
        sensor_size: Tuple[float, float] = (36.0, 24.0),
        focus_distance: float = 4.0,
        aspect_ratio: float = 1.0,
    ):
        """
        :param mode: Camera mode
        :param near_plane: Near plane distance
        :param far_plane: Far plane distance
        :param shutter_speed: Shutter speed in seconds
        :param aperture: Aperture in f-stops, should be in [1, 32] range
        :param iso: ISO
        :param sensor_size: Sensor size in mm
        :param focus_distance: Focus distance in meters, this should not be lower than the focal length
        :param aspect_ratio: Aspect ratio
        """
        self.mode = mode
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.shutter_speed = shutter_speed
        self.aperture = aperture
        self.iso = iso
        self.sensor_size = sensor_size
        self.focus_distance = focus_distance
        self.aspect_ratio = aspect_ratio


@ti.data_oriented
class Camera:
    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self.projection_matrix = ti.Matrix.field(4, 4, float, shape=())
        self.view_matrix = ti.Matrix.field(4, 4, float, shape=())

    @ti.kernel
    def compute_projection_matrix(self):
        if ti.static(self.settings.mode == CameraMode.PERSPECTIVE):
            t = self.settings.near_plane * ti.tan(self.settings.mode.fov_rad / 2)
            r = t * self.settings.aspect_ratio
            self.projection_matrix[None] = projection_from_frustum(
                -r, r, -t, t, self.settings.near_plane, self.settings.far_plane
            )


@ti.func
def projection_from_frustum(left, right, bottom, top, near, far) -> ti.Matrix:
    depth_range = far - near
    n2 = 2 * near
    mxx = n2 / (right - left)
    myy = n2 / (top - bottom)
    mzx = (right + left) / (right - left)
    mzy = (top + bottom) / (top - bottom)
    mzz = -(far) / depth_range
    mwz = near * mzz
    mzw = -1
    # use reversed z
    return ti.Matrix(
        [
            [mxx, 0, 0, 0],
            [0, myy, 0, 0],
            [mzx, mzy, -mzz - 1, mzw],
            [0, 0, -mwz, 0],
        ]
    )
