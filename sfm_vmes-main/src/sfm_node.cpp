#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <rosgraph_msgs/Clock.h>
#include <rosgraph_msgs/Log.h>
#include <lsa_auv_msgs/Imu.h>
#include <lsa_auv_msgs/Pressure.h>
#include <lsa_auv_msgs/water_params.h>
#include <nav_msgs/Odometry.h>
#include <tf2_msgs/TFMessage.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <sfm_node/message_queue.hpp>
#include <thread>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

class sfm_node
{
    public:
        ros::Subscriber image_sub;
        ros::Subscriber nav_sub;
        image_transport::Publisher image_pub;
        image_transport::Publisher feature_pub;
        ros::Publisher pointcloud_pub;
        ros::Publisher trajectory_pub;
        nav_msgs::Path path_msg;
        std::string image_topic_name;
        std::string nav_topic_name;

        sfm_node()
            :image(2)
        {    
            _img_queue = new message_queue<cv::Mat>();
            _thread = new std::thread(&sfm_node::image_thread, this);
        }

        bool getparam(ros::NodeHandle nh);
        void cam_callback(const sensor_msgs::ImageConstPtr& msg);
        void nav_callback(const nav_msgs::OdometryConstPtr& msg);
        void image_thread();
        void gray_convertion(cv::Mat image);
        void publish_image(cv::Mat image, image_transport::Publisher pub);
        void undirstor_im(cv::Mat gray_image);
        void feature_tracking(cv::Mat image_1);
        void estimate_camera_motion(std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2);
        void triangulate_points(cv::Mat R, cv::Mat t, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2);
        void publish_pointcloud();
        void publish_trajectory(cv::Mat R, cv::Mat t);
        void close_prg();

    private:
        /* data */
        std::thread *_thread;
        message_queue<cv::Mat> *_img_queue;
        std::vector<cv::Mat> image;
        std::vector<cv::Point3f> point_cloud;
        std::vector<cv::KeyPoint> accumulated_keypoints;
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
        cv::Mat global_R = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat global_t = cv::Mat::zeros(3, 1, CV_64F); 

};

bool sfm_node::getparam(ros::NodeHandle nh){
    std::vector<double> camera_matrix_vector, dist_coeffs_vector;
    if (!nh.getParam("image_topic_name", image_topic_name))
    {
        ROS_ERROR("ERRO image_topic_name");
        return false;
    }
    if (!nh.getParam("nav_topic_name", nav_topic_name))
    {
        ROS_ERROR("ERRO nav_topic_name");
        return false;
    }
    if(!nh.getParam("camera_matrix/data", camera_matrix_vector)){
        ROS_ERROR("ERRO camera_matrix");
        return false;
    }
    if(!nh.getParam("distortion_coefficients/data", dist_coeffs_vector)){
        ROS_ERROR("ERRO distortion_coefficients");
        return false;
    }

    cameraMatrix = cv::Mat(3, 3, CV_64F, camera_matrix_vector.data()).clone();
    distCoeffs = cv::Mat(dist_coeffs_vector).clone();  
    return true;
}

void sfm_node::cam_callback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        // Converte mensagem ROS para imagem OpenCV
        cv::Mat cv_image = cv_bridge::toCvCopy(msg, "bgr8")->image;
        
        _img_queue->push(cv_image);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Erro ao converter imagem: %s", e.what());
    }
}

void sfm_node::nav_callback(const nav_msgs::OdometryConstPtr& msg){

}

void sfm_node::image_thread(){
    cv::Mat image;
    while(ros::ok()){
        _img_queue->read_message_block(image);
        publish_image( image, image_pub);
        gray_convertion(image);
    }
}

void sfm_node::gray_convertion(cv::Mat image){
    cv::Mat gray_image, adjusted;
    uint8_t brightness = 30;
    uint8_t contrast = 1.2;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    gray_image.convertTo(adjusted,-1,contrast, brightness);
    std::string title = " Gray image";
    undirstor_im(adjusted);
}

void sfm_node::publish_image(cv::Mat image, image_transport::Publisher pub){
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    cv_ptr->header.stamp = ros::Time::now();
    cv_ptr->encoding = "bgr8";
    cv_ptr->image = image;
    //image publishing 
    pub.publish(cv_ptr->toImageMsg());
}

void sfm_node::undirstor_im(cv::Mat gray_image){
    cv::Mat dst, map1, map2,new_camera_matrix;
    cv::Rect validROI;
    cv::Size imageSize(cv::Size(gray_image.cols,gray_image.rows));
    
    // Refining the camera matrix using parameters obtained by calibration
    new_camera_matrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, &validROI);
    
    //remover a distorção
    cv::undistort( gray_image, dst, new_camera_matrix, distCoeffs, new_camera_matrix );
    
    std::string title = "Undistorted Image";
    feature_tracking(dst);
}

void sfm_node::feature_tracking(cv::Mat image_1){
    if (image[0].empty()){
        image[0]= image_1;
    } 
    else {
        image[1]= image_1;
    }
    if(!image[1].empty()){
        ROS_INFO_STREAM("not empty");

        int minHessian = 400;
        cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        detector->detectAndCompute( image[0], cv::noArray(), keypoints1, descriptors1 );
        detector->detectAndCompute( image[1], cv::noArray(), keypoints2, descriptors2 );


        // IMPROVED: Use better matching and filtering
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);
        
        // Filter matches using Lowe's ratio test
        const float ratio_thresh = 0.7f;
        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() >= 2 && 
                knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        
        // Only proceed if we have enough good matches
        if (good_matches.size() < 10) {
            ROS_WARN("Not enough good matches found (%zu)", good_matches.size());
            image[0] = image[1]; // Still update frame
            return;
        }
        
        // Extract matched points - FIXED: use correct indices
        std::vector<cv::Point2f> points1, points2;
        for (const auto &match : good_matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt); // FIXED: trainIdx for second image
        }
        
        // Optional: Draw and publish matched features
        cv::Mat img_matches;
        cv::drawMatches(image[0], keypoints1, image[1], keypoints2, good_matches, img_matches);
        publish_image(img_matches, feature_pub);
        
        // Estimate camera motion
        estimate_camera_motion(points1, points2);
        
        // Store keypoints for visualization if needed
        accumulated_keypoints.insert(accumulated_keypoints.end(), keypoints2.begin(), keypoints2.end());
        
        // Update reference frame
        image[0] = image[1];
    }
}


// void sfm_node::estimate_camera_motion(std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2) {
//     // Camera intrinsics (already set in getparam())
//     cv::Mat K = cameraMatrix;

//     // Compute Essential Matrix
//     cv::Mat E = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.999, 1.0);

//     if (E.empty()) {
//         ROS_WARN("Essential Matrix computation failed.");
//         return;
//     }

//     // Recover R (rotation) and t (translation)
//     cv::Mat R, t;
//     int num_inliers = cv::recoverPose(E, points1, points2, K, R, t);
//     if (num_inliers < 10) {
//         ROS_WARN("Insufficient inliers for motion estimation.");
//         return;
//     }

//     // Update the global camera pose
//     cv::Mat global_R = cv::Mat::eye(3, 3, CV_64F);
//     cv::Mat global_t = cv::Mat::zeros(3, 1, CV_64F);
//     global_t = global_t + (global_R * t);  // Move camera forward
//     global_R = global_R * R;  // Rotate camera

//     // Call triangulation
//     triangulate_points(global_R, global_t, points1, points2);
// }

void sfm_node::estimate_camera_motion(std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2) {
    // Camera intrinsics
    cv::Mat K = cameraMatrix;
    
    // Make sure we have enough points to proceed
    if (points1.size() < 8 || points2.size() < 8) {
        ROS_WARN("Not enough points for essential matrix: %zu", points1.size());
        return;
    }
    
    // Step 1: Find essential matrix with RANSAC
    cv::Mat mask; // Output mask of inliers
    cv::Mat E = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.999, 1.0, mask);
    
    if (E.empty()) {
        ROS_WARN("Essential Matrix computation failed.");
        return;
    }
    
    // Step 2: Call recoverPose with the SAME points and mask
    cv::Mat R, t;
    // IMPORTANT: Pass points1, points2, K, E, and mask directly to recoverPose
    int num_inliers = cv::recoverPose(E, points1, points2, K, R, t, mask);
    
    // Check if we have enough inliers
    if (num_inliers < 8) {
        ROS_WARN("Insufficient inliers for motion estimation: %d", num_inliers);
        return;
    }
    
    // Print some debugging info
    ROS_INFO("Camera motion estimated with %d inliers out of %zu points", 
             num_inliers, points1.size());
    
    // Step 3: Update global pose correctly
    // Update rotation first
    cv::Mat R_new = R * global_R;
    // Then update translation
    cv::Mat t_new = global_t + global_R * t;
    
    // Update global variables
    global_R = R_new;
    global_t = t_new;
    
    // Publish trajectory
    publish_trajectory(global_R, global_t);
    
    // Triangulate points using only inlier pairs
    std::vector<cv::Point2f> inlier_points1, inlier_points2;
    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<uchar>(i)) {
            inlier_points1.push_back(points1[i]);
            inlier_points2.push_back(points2[i]);
        }
    }
    
    // Call triangulation with the relative pose
    triangulate_points(R, t, inlier_points1, inlier_points2);
}

// void sfm_node::triangulate_points(cv::Mat R, cv::Mat t, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2) {
//     //1. Check R and t before using them
//     if (R.empty() || t.empty()) {
//         ROS_WARN("Invalid R or t for triangulation.");
//         return;
//     }

//     //2. Create projection matrices
//     cv::Mat proj1 = cameraMatrix * cv::Mat::eye(3, 4, CV_64F);
//     cv::Mat proj2 = cameraMatrix * (cv::Mat_<double>(3, 4) << 
//                         R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
//                         R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
//                         R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2));

//     //3. Triangulate points
//     cv::Mat points4D;
//     cv::triangulatePoints(proj1, proj2, points1, points2, points4D);

//     // Convert 4D homogeneous points to 3D
//     points4D.convertTo(points4D, CV_64F);  // Ensure consistency
//     for (int i = 0; i < points4D.cols; i++) {
//         cv::Point3f new_point(
//             points4D.at<double>(0, i) / points4D.at<double>(3, i),
//             points4D.at<double>(1, i) / points4D.at<double>(3, i),
//             points4D.at<double>(2, i) / points4D.at<double>(3, i));

//         //5. Convert to global coordinates
//         cv::Mat R_t = cv::Mat::eye(4, 4, CV_64F);
//         R.copyTo(R_t(cv::Rect(0, 0, 3, 3)));
//         t.copyTo(R_t(cv::Rect(3, 0, 1, 3)));

//         cv::Mat new_point_h = (cv::Mat_<double>(4, 1) << new_point.x, new_point.y, new_point.z, 1.0);
//         cv::Mat global_point_h = R_t * new_point_h;

//         cv::Point3f transformed_point(
//             global_point_h.at<double>(0, 0),
//             global_point_h.at<double>(1, 0),
//             global_point_h.at<double>(2, 0));

//         // Ignore outliers
//         if (cv::norm(transformed_point) < 200) {  
//             point_cloud.push_back(transformed_point);
//         }
//     }

//     // Publish updated point cloud
//     publish_pointcloud();
// }


void sfm_node::triangulate_points(cv::Mat R, cv::Mat t, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2) {
    if (R.empty() || t.empty() || points1.empty() || points2.empty()) {
        ROS_WARN("Invalid inputs for triangulation.");
        return;
    }
    
    // FIXED: Use relative camera poses for triangulation
    // First camera is at origin with identity rotation
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
    P1 = cameraMatrix * P1;
    
    // Second camera has relative pose [R|t] from first camera
    cv::Mat P2(3, 4, CV_64F);
    R.copyTo(P2.colRange(0, 3));
    t.copyTo(P2.col(3));
    P2 = cameraMatrix * P2;
    
    // Triangulate points
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, points1, points2, points4D);
    
    // Convert homogeneous coordinates to 3D points
    std::vector<cv::Point3f> new_points;
    for (int i = 0; i < points4D.cols; i++) {
        if (points4D.at<float>(3, i) == 0) continue; // Skip points at infinity
        
        // Convert to 3D point
        cv::Point3f p(
            points4D.at<float>(0, i) / points4D.at<float>(3, i),
            points4D.at<float>(1, i) / points4D.at<float>(3, i),
            points4D.at<float>(2, i) / points4D.at<float>(3, i)
        );
        
        // IMPROVED: Better filtering of outliers
        // Check if point is in front of both cameras
        cv::Mat p1 = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
        cv::Mat p2 = R * p1 + t;
        
        if (p1.at<double>(2) > 0 && p2.at<double>(2) > 0) {
            // Check distance (reject points too far or too close)
            double dist = cv::norm(p);
            if (dist > 0.1 && dist < 100.0) {
                // Transform to global coordinate system
                cv::Mat p_local = (cv::Mat_<double>(4, 1) << p.x, p.y, p.z, 1.0);
                
                // Create global transformation matrix
                cv::Mat Rt_global = cv::Mat::eye(4, 4, CV_64F);
                global_R.copyTo(Rt_global(cv::Rect(0, 0, 3, 3)));
                global_t.copyTo(Rt_global(cv::Rect(3, 0, 1, 3)));
                
                // Transform point to global space
                cv::Mat p_global = Rt_global * p_local;
                
                cv::Point3f global_point(
                    p_global.at<double>(0, 0),
                    p_global.at<double>(1, 0),
                    p_global.at<double>(2, 0)
                );
                
                point_cloud.push_back(global_point);
            }
        }
    }
    
    // IMPROVED: Limit point cloud size to prevent memory issues
    if (point_cloud.size() > 10000) {
        point_cloud.erase(point_cloud.begin(), point_cloud.begin() + (point_cloud.size() - 10000));
    }
    
    // Publish updated point cloud
    publish_pointcloud();
}

void sfm_node::publish_pointcloud() {
    sensor_msgs::PointCloud2 cloud_msg;
    cloud_msg.header.stamp = ros::Time::now();
    cloud_msg.header.frame_id = "map";
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(point_cloud.size());

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
    for (const auto &pt : point_cloud) {
        *iter_x = pt.x;
        *iter_y = pt.y;
        *iter_z = pt.z;
        ++iter_x;
        ++iter_y; 
        ++iter_z;
    }
    pointcloud_pub.publish(cloud_msg);
}

void sfm_node::publish_trajectory(cv::Mat R, cv::Mat t) {
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = "map";

    // Convert OpenCV matrix to ROS Pose
    pose.pose.position.x = t.at<double>(0, 0);
    pose.pose.position.y = t.at<double>(1, 0);
    pose.pose.position.z = t.at<double>(2, 0);

    // Convert rotation matrix to quaternion
    cv::Mat R_vec;
    cv::Rodrigues(R, R_vec);  // Convert 3x3 rotation matrix to vector
    tf2::Matrix3x3 tf_R(
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2));

    tf2::Quaternion q;
    tf_R.getRotation(q);
    pose.pose.orientation.x = q.x();
    pose.pose.orientation.y = q.y();
    pose.pose.orientation.z = q.z();
    pose.pose.orientation.w = q.w();

    path_msg.header.stamp = ros::Time::now();
    path_msg.header.frame_id = "map";
    path_msg.poses.push_back(pose);
    // Publish trajectory
    trajectory_pub.publish(path_msg);
}


void sfm_node::close_prg(){
    _img_queue->close();
    if (_img_queue != nullptr){
        delete _img_queue;
    }
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "sfm_node");
    ros::NodeHandle nh("~");
    
    sfm_node node;
     //Ler parametros 
    if(!node.getparam(nh)){
        ROS_ERROR("ERROR getting params");
        return -1;
    }
    

    // Subscrevendo aos outros tópicos
    node.image_sub = nh.subscribe(node.image_topic_name, 10, &sfm_node::cam_callback, &node);
    node.nav_sub = nh.subscribe(node.nav_topic_name, 10, &sfm_node::nav_callback, &node);
   
    image_transport::ImageTransport it(nh);
    node.image_pub = it.advertise("image", 10);
    node.feature_pub = it.advertise("image_feature", 10);
    node.pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>("sfm_pointcloud", 10);
    node.trajectory_pub = nh.advertise<nav_msgs::Path>("sfm_trajectory",10);


    ROS_INFO("Nó combinado iniciado. Aguardando mensagens...");
    ros::spin();
    node.close_prg();

    return 0;
}