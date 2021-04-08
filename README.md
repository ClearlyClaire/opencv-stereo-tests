This is just a collection of scripts I wrote trying to make sense of the Valve Index cameras.

Although they are not usable for any of the SteamVR features on linux at the moment I wrote this, they are available as a dual-fisheye V4L2 device.

These scripts are provided as is, as examples or starting point, there is no plan to build any working application around them.

So far the results are somewhat disappointing, the resulting videos still seem off, mainly on close objects, but that might be inherent to the large distance between the Valve Index cameras.

## Dependencies

All the scripts depend on the Python OpenCV bindings as well as NumPy.
The `v4l2loopback.py` script also depends on [V4L2 bindings for Python](https://pypi.org/project/v4l2/) and the [v4l2loopback module](https://github.com/umlaeute/v4l2loopback) (available as `v4l2loopback-dkms` in Debian).

## Scripts

- `calibrate.py`: creates a [ChAruco board](https://docs.opencv.org/3.4/da/d13/tutorial_aruco_calibration.html) image (`charuco_board.png` by default), opens a camera device or a video file (edit the file as needed), and tries calibrating the pair of cameras from the ChAruco board. It outputs the camera intrisics, distortion parameters and stereo rotation and translation matrixes on exit.
- `remap.py`: computes maps from camera intrinsics, distortion parameters and stereo rotation, and remap a video file or camera input to a side-by-side stereo video file.
- `v4l2loopback.py`: similar to `remap.py` but instead of outputing to a side-by-side stereo video file, output each camera output to a loopback v4l2 device.
