import rospy
import time
from geometry_msgs.msg import PoseStamped

rospy.init_node('pose_publisher', anonymous=True)
pose_pub = rospy.Publisher('/sync_nav', PoseStamped, queue_size=10)

# **Wait until at least one subscriber is connected**
while pose_pub.get_num_connections() == 0:
    rospy.loginfo("Waiting for subscribers...")
    time.sleep(0.1)

pose_msg = PoseStamped()
pose_msg.header.stamp = rospy.Time.now()
pose_msg.header.frame_id = "map"
pose_msg.pose.position.x = 1.0
pose_msg.pose.position.y = 2.0
pose_msg.pose.position.z = 3.0
pose_msg.pose.orientation.w = 1.0  # Quaternion (1,0,0,0) = No rotation

pose_pub.publish(pose_msg)
rospy.loginfo("Published a single camera pose message.")
