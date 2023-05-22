import cv2
import numpy as np
import math
def preprocess(map,map_size=100,robot_size=0.2,scale=10):
    # Convert the image to grayscale
    # gray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    # Threshold the image, let's consider values close to 0 as black (adjust according to your case)
    map = map.astype('uint8')
    _, thresh = cv2.threshold(map, 1, 255, cv2.THRESH_BINARY_INV)
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    robot_range=max(1, int(math.floor(map_size * robot_size / scale / 2)))
    occupancy_map = (
            np.ones((map_size + 2 * robot_range, map_size + 2 * robot_range)) * 255
    )
    robot_range=1
    # Iterate over each contour
    # print(contours)
    for contour in contours:
        # Calculate moments for each contour
        M = cv2.moments(contour)
        # Calculate x,y coordinate of the center of the contour
        if M["m00"]==0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        for m in range(-robot_range, robot_range, 1):
            for n in range(-robot_range, robot_range, 1):
                occupancy_map[cX + m][cY + n] = 0
    occupancy_map = occupancy_map[
                    robot_range:-robot_range, robot_range:-robot_range
                    ]
    return occupancy_map
if __name__=="__main__":
    map=cv2.imread("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/4.jpg")
    reference=cv2.imread("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/4s.jpg")
    map = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    map=preprocess(map)
    # cv2.imwrite("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/4preprocess.jpg",map)
    print(map)
    print(reference)
    processed = cv2.imread("/home/xinchi/catkin_ws/src/multi_robot_formation/src/multi_robot_formation/4preprocess.jpg")
    processed = cv2.cvtColor( processed, cv2.COLOR_BGR2GRAY)

    cv2.imshow("map",map)
    cv2.imshow("reference",reference)
    cv2.imshow("processed", processed)

    cv2.waitKey(0)