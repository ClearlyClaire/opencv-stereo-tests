import numpy
import cv2

# Camera width and height (for a single eye)
WIDTH = 960
HEIGHT = 960

# Camera device index or video file path
# Should be a dual fisheye camera
VIDEO_INPUT = 0

# ChAruco board parameters
# The default parameters are the same as the ones used by opencv_interactive-calibration
CHARUCO_OUT_PATH = 'charuco_board.png'
CHARUCO_SQUARES = (6, 8)
CHARUCO_PIXEL_SIZE = (1080, 1920)
ARUCO_DICT = 0 # Default dict
CHARUCO_SQUARE_LENGTH = 0.065
CHARUCO_MARKER_LENGTH = CHARUCO_SQUARE_LENGTH / 2.

# Generate a chAruco board and output a file that can be used for calibration
aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT)
board = cv2.aruco.CharucoBoard_create(CHARUCO_SQUARES[0], CHARUCO_SQUARES[1], CHARUCO_SQUARE_LENGTH, CHARUCO_MARKER_LENGTH, aruco_dict)
board_img = board.draw(CHARUCO_PIXEL_SIZE)
cv2.imwrite(CHARUCO_OUT_PATH, board_img)

# Open input video
cap = cv2.VideoCapture(VIDEO_INPUT)

# We basically need to reimplement part of the charuco methods since the checkboard isn't exposed
# and we need to give object points to stereoCalibrate
def make_chess_board(squares_x, squares_y, square_length):
  res = []
  for y in range(squares_y - 1):
    for x in range(squares_x - 1):
      res.append(numpy.array([[(x + 1) * square_length, (y + 1) * square_length, 0]], numpy.float32))
  return res

chessboard = make_chess_board(CHARUCO_SQUARES[0], CHARUCO_SQUARES[1], CHARUCO_SQUARE_LENGTH)

def detect_charucos_stereo(left, right, allObjPoints, allLeftImgPoints, allRightImgPoints):
  """Detect common points from left and right images using chAruco boards.

  This detects aruco markers on both left and right images, interpolates the associated checkboard corners,
  then keep only those which were detected in both images, reorders them as needed and extracts the object
  points."""
  left_gray  = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
  right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

  leftCorners,  leftIds,  rejected = cv2.aruco.detectMarkers(left_gray, aruco_dict)
  rightCorners, rightIds, rejected = cv2.aruco.detectMarkers(right_gray, aruco_dict)
  if len(leftCorners) == 0 or len(rightCorners) == 0:
    return

  leftRet,  leftCorners,  leftIds  = cv2.aruco.interpolateCornersCharuco(leftCorners,  leftIds,  left_gray,  board)
  rightRet, rightCorners, rightIds = cv2.aruco.interpolateCornersCharuco(rightCorners, rightIds, right_gray, board)
  if not leftRet or not rightRet:
    return

  commonIds = list(set(x[0] for x in leftIds).intersection(x[0] for x in rightIds))
  if len(commonIds) < 6:
    return

  commonIds.sort()

  # Slice left and right corners/ids with common stuff
  # Care has to be given to reorder objects so they match across left and right
  leftIdxs = [[x[0] for x in leftIds].index(y) for y in commonIds]
  rightIdxs = [[x[0] for x in rightIds].index(y) for y in commonIds]
  leftCorners = numpy.array([leftCorners[x] for x in leftIdxs])
  leftIds = numpy.array([leftIds[x] for x in leftIdxs])
  rightCorners = numpy.array([rightCorners[x] for x in rightIdxs])
  rightIds = numpy.array([rightIds[x] for x in rightIdxs])

  objPoints = numpy.array([chessboard[x[0]] for x in leftIds])

  allObjPoints.append(objPoints)
  allLeftImgPoints.append(leftCorners)
  allRightImgPoints.append(rightCorners)

    
leftCameraMatrix = numpy.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=numpy.double)
leftDistCoeffs = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=numpy.double)

rightCameraMatrix = numpy.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=numpy.double)
rightDistCoeffs = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=numpy.double)

allObjPoints = []
allLeftImgPoints = []
allRightImgPoints = []

flags = 0

leftMapX, leftMapY = None, None
rightMapX, rightMapY = None, None

imgSize = (WIDTH, HEIGHT)

frame_number = 0
while cap.isOpened():
  ret, frame = cap.read()
  frame_number += 1
  if ret == True:
    left  = frame[0:HEIGHT, 0:WIDTH]
    right = frame[0:HEIGHT, WIDTH:WIDTH*2]

    # do not pick up all frames because it's just too much data otherwise
    if frame_number % 25 == 0:
      detect_charucos_stereo(left, right, allObjPoints, allLeftImgPoints, allRightImgPoints)

    if len(allObjPoints) > 40:
      retval, leftCameraMatrix, leftDistCoeffs, rightCameraMatrix, rightDistCoeffs, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=allObjPoints,
        imagePoints1=allLeftImgPoints,
        imagePoints2=allRightImgPoints,
        imageSize=imgSize,
        cameraMatrix1=leftCameraMatrix,
        distCoeffs1=leftDistCoeffs,
        cameraMatrix2=rightCameraMatrix,
        distCoeffs2=rightDistCoeffs,
        flags=flags)
      flags |= cv2.CALIB_USE_INTRINSIC_GUESS
      allObjPoints = []
      allLeftImgPoints = []
      allRightImgPoints = []
      R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(leftCameraMatrix, leftDistCoeffs, rightCameraMatrix, rightDistCoeffs, imgSize, R, T, flags=cv2.CALIB_ZERO_DISPARITY)
      leftMapX, leftMapY = cv2.initUndistortRectifyMap(leftCameraMatrix, leftDistCoeffs, R1, P1, imgSize, cv2.CV_32FC1)
      rightMapX, rightMapY = cv2.initUndistortRectifyMap(rightCameraMatrix, rightDistCoeffs, R2, P2, imgSize, cv2.CV_32FC1)

    if leftMapX is not None:
      left = cv2.remap(left, leftMapX, leftMapY, cv2.INTER_LINEAR)
      right = cv2.remap(right, rightMapX, rightMapY, cv2.INTER_LINEAR)

    frame = cv2.hconcat([left, right])
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    break

print('leftCameraMatrix = %r' % leftCameraMatrix)
print('leftDistCoeffs = %r' % leftDistCoeffs)
print('rightCameraMatrix = %r' % rightCameraMatrix)
print('rightDistCoeffs = %r' % rightDistCoeffs)
print('R = %r' % R)
print('T = %r' % T)

cap.release()

cv2.destroyAllWindows()
