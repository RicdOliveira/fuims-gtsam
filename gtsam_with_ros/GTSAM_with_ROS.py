import gtsam
import rospy
from gtsam.symbol_shorthand import X
from geometry_msgs.msg import PoseStamped
from threading import Lock

DATA_PATH = "../data/"

class TrajectoryEstimator:
    def __init__(self):
        rospy.init_node('trajectory_estimator', anonymous=True)

        # ROS Subscriber for camera pose
        self.cam_pose_sub = rospy.Subscriber('/camera_pose', PoseStamped, self.cam_pose_callback)
        self.sync_nav_sub = rospy.Subscriber('/sync_nav', PoseStamped, self.sync_nav_callback)
        self.sync_nav_sub = None
        self.__prev_nave_gtsam_pose = None
        self.latest_cam_pose = None
        self.lock = Lock()  # Ensure thread safety

        # ROS Publisher for estimated trajectory
        self.pose_pub = rospy.Publisher('/estimated_trajectory', PoseStamped, queue_size=10)
    
    def cam_pose_callback(self, msg):
        """Callback to update the latest camera pose from ROS."""
        with self.lock:
            self.latest_cam_pose = msg
    
    def sync_nav_callback(self, msg):
        """Callback to update the latest navegation pose from ROS."""
        with self.lock:
            self.latest_cam_pose = msg

    def get_camera_pose(self):
        """
        Blocks until a new camera pose message is received.
        Ensures synchronization with ROS.
        """
        rospy.loginfo("Waiting for camera pose...")
        cam_pose_msg = rospy.wait_for_message('/camera_pose', PoseStamped)
        with self.lock:
            self.latest_cam_pose = cam_pose_msg
        return cam_pose_msg
    
    def get_nav_pose(self):
        """
        Blocks until a new navegation pose message is received.
        Ensures synchronization with ROS.
        """
        rospy.loginfo("Waiting for sync nav pose...")
        nav_pose_msg = rospy.wait_for_message('/sync_nav', PoseStamped)
        with self.lock:
            self.latest_sync_nav = nav_pose_msg
        return nav_pose_msg

    def estimate_trajectory_gtsam(self):
        # Initialize GTSAM factor graph and values
        graph = gtsam.NonlinearFactorGraph()
        initial_estimates = gtsam.Values()

        # Noise models
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1] * 6)
        measurement_noise = gtsam.noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.2, 0.5, 0.5, 0.5])

        nav_pose_msg1 = self.get_nav_pose()

        nav_pos1 = [nav_pose_msg1.pose.position.x, nav_pose_msg1.pose.position.y, nav_pose_msg1.pose.position.z]
        nav_rot1 = gtsam.Rot3.Quaternion(
            nav_pose_msg1.pose.orientation.w,
            nav_pose_msg1.pose.orientation.x,
            nav_pose_msg1.pose.orientation.y,
            nav_pose_msg1.pose.orientation.z
        )
        initial_pose = gtsam.Pose3(nav_rot1, nav_pos1)
        self.__prev_nave_gtsam_pose = initial_pose

        graph.add(gtsam.PriorFactorPose3(X(0), initial_pose, prior_noise))
        initial_estimates.insert(X(0), initial_pose)

        i = 1
        while 1:
            nav_pose_msg = self.get_nav_pose()

            nav_pos = [nav_pose_msg.pose.position.x, nav_pose_msg.pose.position.y, nav_pose_msg.pose.position.z]
            nav_rot = gtsam.Rot3.Quaternion(
                nav_pose_msg.pose.orientation.w,
                nav_pose_msg.pose.orientation.x,
                nav_pose_msg.pose.orientation.y,
                nav_pose_msg.pose.orientation.z
            )
            absolute_nav_measurement = gtsam.Pose3(nav_rot, nav_pos)

            cam_pose_msg = self.get_camera_pose()

            cam_pos = [cam_pose_msg.pose.position.x, cam_pose_msg.pose.position.y, cam_pose_msg.pose.position.z]
            cam_rot = gtsam.Rot3.Quaternion(
                cam_pose_msg.pose.orientation.w,
                cam_pose_msg.pose.orientation.x,
                cam_pose_msg.pose.orientation.y,
                cam_pose_msg.pose.orientation.z
            )
            absolute_cam_measurement = gtsam.Pose3(cam_rot, cam_pos)

            # Add absolute factors every 5 steps
            if i % 5 == 0:
                graph.add(gtsam.PriorFactorPose3(X(i), absolute_nav_measurement, measurement_noise))
                graph.add(gtsam.PriorFactorPose3(X(i), absolute_cam_measurement, measurement_noise))

            # Compute relative transformation
            relative_measurement = self.__prev_nave_gtsam_pose.between(absolute_nav_measurement)
            graph.add(gtsam.BetweenFactorPose3(X(i - 1), X(i), relative_measurement, measurement_noise))

            self.__prev_nave_gtsam_pose = absolute_cam_measurement
            # Insert initial estimate
            initial_estimates.insert(X(i), absolute_nav_measurement)

            # Run partial optimization
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
            temp_result = optimizer.optimize()

            # Extract optimized pose
            optimized_pose = temp_result.atPose3(X(i))

            # Publish the estimated pose
            self.publish_estimated_pose(optimized_pose)

            rospy.loginfo(f"Estimativa otimizada parcial para X({i}): {optimized_pose}")
            i = i + 1

    def publish_estimated_pose(self, pose):
        """Publishes the optimized pose as a ROS PoseStamped message."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = pose.x()
        pose_msg.pose.position.y = pose.y()
        pose_msg.pose.position.z = pose.z()

        q = pose.rotation().toQuaternion()
        pose_msg.pose.orientation.w = q.w()
        pose_msg.pose.orientation.x = q.x()
        pose_msg.pose.orientation.y = q.y()
        pose_msg.pose.orientation.z = q.z()

        self.pose_pub.publish(pose_msg)

if __name__ == '__main__':
    estimator = TrajectoryEstimator()    
    estimator.estimate_trajectory_gtsam()
