class CameraDistortion:
    def __init__(self, k1: float, k2: float, p1: float, p2: float, distortion_model: str):
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2

class CameraIntrinsics:
    def __init__(self, fx: float, fy: float, u: float, v: float):
        self.fx = fx
        self.fy = fy
        self.u = u
        self.v = v

class CameraResolution:
    def __init__(self, width: int, height: int):
        self.width = width
        self.heigth = height

class CameraCalibration:
    def __init__(self, intrinsics: CameraIntrinsics, distortion: CameraDistortion, resolution: CameraResolution, camera_model:str='pinhole', flip:bool=True):
        self.intrinsics = intrinsics
        self.distortion = distortion
        self.resolution = resolution
        self.camera_model = camera_model
        self.flip = flip
