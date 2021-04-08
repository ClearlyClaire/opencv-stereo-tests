import cv2
from numpy import array, matrix
import v4l2
import fcntl

# Camera width and height (for a single eye)
WIDTH = 960
HEIGHT = 960

# Camera device index or video file path
# Should be a dual fisheye camera. Left and Right eye views
# are assumed to be side-by-side and of the same pixel size.
VIDEO_INPUT = 0

# Output V4L2 devices (requires the v4l2-loopback module)
LEFT_OUT_V4L2_DEVICE  = '/dev/video2'
RIGHT_OUT_V4L2_DEVICE = '/dev/video3'

# Camera intrinsics, distortion parameters and stereo parameters computed with calibration.py
leftCameraMatrix = matrix([[418.12009366,   0.        , 474.70168291],
        [  0.        , 418.26107355, 475.82296102],
        [  0.        ,   0.        ,   1.        ]])
leftDistCoeffs = array([[-1.41472770e-01, -5.00821052e-03, -1.36898095e-04,
        -5.31109897e-04,  5.92505609e-03]])
rightCameraMatrix = matrix([[420.70164293,   0.        , 494.0718344 ],
        [  0.        , 421.00224821, 485.6593556 ],
        [  0.        ,   0.        ,   1.        ]])
rightDistCoeffs = array([[-1.47821076e-01,  2.58986389e-03, -5.24606308e-04,
         1.42863352e-04,  3.41432318e-03]])
R = array([[ 0.98734558,  0.02063431, -0.15723527],
       [-0.02137971,  0.99976677, -0.00305063],
       [ 0.15713565,  0.00637367,  0.98755646]])
T = array([[-0.0902457 ],
       [ 0.00017903],
       [-0.00636208]])


# Compute reprojection and correction matrices from above data
imgSize = (WIDTH, HEIGHT)
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(leftCameraMatrix, leftDistCoeffs, rightCameraMatrix, rightDistCoeffs, imgSize, R, T, flags=cv2.CALIB_ZERO_DISPARITY)
leftMapX, leftMapY = cv2.initUndistortRectifyMap(leftCameraMatrix, leftDistCoeffs, R1, P1, imgSize, cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(rightCameraMatrix, rightDistCoeffs, R2, P2, imgSize, cv2.CV_32FC1)

cap = cv2.VideoCapture(VIDEO_INPUT)

def setup_cam_out(path, width, height):
  f = open(path, 'wb')

  if not f:
    raise Exception

  v = v4l2.v4l2_format()
  v.type = v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT

  if fcntl.ioctl(f, v4l2.VIDIOC_G_FMT, v) < 0:
    raise Exception

  v.fmt.pix.width = width
  v.fmt.pix.height = height
  v.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_RGB24
  v.fmt.pix.sizeImage = width * height * 3

  if fcntl.ioctl(f, v4l2.VIDIOC_S_FMT, v) < 0:
    raise Exception

  return f

left_dummy_cam = setup_cam_out(LEFT_OUT_V4L2_DEVICE, WIDTH, HEIGHT)
right_dummy_cam = setup_cam_out(RIGHT_OUT_V4L2_DEVICE, WIDTH, HEIGHT)

while cap.isOpened():
  ret, frame = cap.read()
  if ret == True:
    left  = frame[0:HEIGHT, 0:WIDTH]
    right = frame[0:HEIGHT, WIDTH:WIDTH*2]

    left = cv2.remap(left, leftMapX, leftMapY, cv2.INTER_LINEAR)
    right = cv2.remap(right, rightMapX, rightMapY, cv2.INTER_LINEAR)

    frame = cv2.hconcat([left, right])
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    left_dummy_cam.write(left.data)

    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    right_dummy_cam.write(right.data)

  else:
    break

left_dummy_cam.close()
right_dummy_cam.close()
cap.release()
out.release()

cv2.destroyAllWindows()
