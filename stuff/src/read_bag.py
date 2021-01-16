#!/usr/bin/env python2.7

import matplotlib
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Int32
import numpy as np
from matplotlib.animation import FuncAnimation
import random
from mpl_toolkits.mplot3d import Axes3D


class Visualiser:
    def __init__(self):
        # self.fig, self.ax = plt.subplots()

        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig, self.ax = plt.subplots()

        self.ln, = plt.plot([], [], 'ro')
        self.x_data, self.y_data = [], []

    def plot_init(self):
        self.ax.set_xlim(0, 10000)
        self.ax.set_ylim(-7, 7)
        return self.ln

    def odom_callback(self, msg):
        yaw_angle = random.randint(1, 5)  # self.getYaw(msg.pose.pose)
        self.y_data.append(yaw_angle)
        x_index = len(self.x_data)
        self.x_data.append(random.randint(3, 8000))

        #self.ln.set_data(self.x_data, self.y_data)

    def update_plot(self, frame):
        self.ln.set_data(self.x_data, self.y_data)
        return self.ln


rospy.init_node('Camera_Lidar')
vis = Visualiser()
sub = rospy.Subscriber('/counter', Int32, vis.odom_callback)

ani = FuncAnimation(vis.fig, vis.update_plot, init_func=vis.plot_init)
plt.show()




#
rate = rospy.Rate(5)
pub = rospy.Publisher('/counter', Int32, queue_size=1)
counter = 0

# vis = Visualiser()
# sub = rospy.Subscriber('/counter', Int32, vis.scanner_callback, queue_size=1)
# plt.ion()
# plt.show()
# rospy.spin()
# plt.show(block=True)
# plt.show()
# try:
#    rospy.spin()
# except rospy.ROSInterruptException:
#    rospy.loginfo('Shutting down')
# plt.show()
rospy.loginfo("Hello ROS!")
while not rospy.is_shutdown():
    counter += 1
    msg = Int32()
    msg.data = counter
    pub.publish(msg)
    rate.sleep()
