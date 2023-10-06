#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that listens to std_msgs/Strings published
## to the 'chatter' topic

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from cmvision.msg import Blobs
from cmvision_3d.msg import Blobs3d, Blob3d


from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import sys
import cv2
import pyrealsense2 as rs

class ImageListener:
    def __init__(self, topic):
        self.topic = topic
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(topic, Blobs3d, self.imageDepthCallback)

        self.map_size=1000
        self.range=2
        self.height=0.5
        self.color_map={"red":(255,0,0),"yellow":(255,255,0),"green":(0,255,0),"blue":(0,0,255)}
    def imageDepthCallback(self, data):

        try:

            occupancy_map=np.ones((self.map_size,self.map_size,3))

            blob_list=[]
            for blob in data.blobs:
                blob_list.append([blob.name,[blob.center.x,blob.center.y,blob.center.z],
                                  [blob.top_left.x,blob.top_left.y,blob.top_left.z],
                                  [blob.bottom_right.x,blob.bottom_right.y,blob.bottom_right.z]])

            for p in blob_list:
                if -self.height<p[1][1]<self.height and -self.range<p[1][0]<self.range and -self.range<p[1][2]<self.range:
                    front_w=min(p[1][2],p[2][2],p[3][2])
                    back_w=max(p[1][2],p[2][2],p[3][2])
                    left_w=min(p[1][0],p[2][0],p[3][0])
                    right_w = max(p[1][0],p[2][0],p[3][0])
                    front_m= int(front_w * self.map_size / self.range / 2 + self.map_size / 2)
                    back_m=int(back_w * self.map_size / self.range / 2 + self.map_size / 2)
                    left_m=int(left_w * self.map_size / self.range / 2 + self.map_size / 2)
                    right_m=int(right_w * self.map_size / self.range / 2 + self.map_size / 2)
                    print(p)
                    for i in range(front_m,front_m+int(20000/self.map_size)):
                        for j in range(left_m,left_m+int(20000/self.map_size)):
                            occupancy_map[self.map_size - i][j][2] = self.color_map[p[0]][0]/255.0
                            occupancy_map[self.map_size - i][j][1] = self.color_map[p[0]][1]/255.0
                            occupancy_map[self.map_size - i][j][0] = self.color_map[p[0]][2]/255.0
            # r, g, b = cv2.split(occupancy_map)
            # occupancy_map = cv2.merge([b, g, r])
            # points = point_cloud2.read_points_list(data)
            #
            # for p in points:
            #     if -self.height<p[1]<self.height and -self.range<p[0]<self.range and -self.range<p[2]<self.range:
            #         x = int(p[2] / self.range * self.map_size / 2) + self.map_size / 2
            #         y = int(p[0] / self.range * self.map_size / 2) + self.map_size / 2
            #         occupancy_map[self.map_size-x][y]=0
            # cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # pix = (data.width/2, data.height/2)
            # print(type(cv_image))
            # points=rs.points()

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', occupancy_map)
            cv2.waitKey(1)
            # sys.stdout.write('%s: Depth at center(%d, %d): %f(mm)\r' % (self.topic, pix[0], pix[1], cv_image[pix[1], pix[0]]))

            sys.stdout.flush()
        except:
            return


if __name__ == '__main__':
    rospy.init_node("occupancy_map_visualize")
    topic = '/blobs_3d'
    listener = ImageListener(topic)
    rospy.spin()
