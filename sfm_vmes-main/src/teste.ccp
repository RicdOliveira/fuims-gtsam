#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <thread>
#include <vector>

class sfm_node {
public:
    ros::Subscriber image_sub, nav_sub;
    image_transport::Publisher image_pub, feature_pub;
    ros::Publisher pointcloud_pub;
    std::string image_topic_name, nav_topic_name;

    sfm_node() : image(2) {
        _img_queue = new message_queue<cv::Mat>();
        _thread = new std::thread(&sfm_node::image_thread, this);
    }

    ~sfm_node() { close_prg(); }

    bool getparam(ros::NodeHandle nh);
    void cam_callback(const sensor_msgs::ImageConstPtr& msg);
    void nav_callback(const nav_msgs::OdometryConstPtr& msg);
    void image_thread();
    void gray_convertion(cv::Mat image);
    void publish_image(cv::Mat image, image_transport::Publisher pub);
    void undistort_im(cv::Mat gray_image);
    void feature_tracking(cv::Mat image_1);
    void triangulate_points(cv::Mat R, cv::Mat t, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2);
    void publish_pointcloud();
    void close_prg();

private:
    std::thread* _thread;
    message_queue<cv::Mat>* _img_queue;
    std::vector<cv::Mat> image;
    std::vector<cv::Point3f> point_cloud;
    std::vector<cv::KeyPoint> accumulated_keypoints;
    cv::Mat cameraMatrix, distCoeffs;
};

void sfm_node::cam_callback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv::Mat cv_image = cv_bridge::toCvCopy(msg, "bgr8")->image;
        _img_queue->push(cv_image);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Error converting image: %s", e.what());
    }
}

void sfm_node::nav_callback(const nav_msgs::OdometryConstPtr& msg) {
    // Handle odometry updates if needed
}

void sfm_node::image_thread() {
    cv::Mat image;
    while (ros::ok()) {
        _img_queue->read_message_block(image);
        publish_image(image, image_pub);
        gray_convertion(image);
    }
}

void sfm_node::gray_convertion(cv::Mat image) {
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    undistort_im(gray_image);
}

void sfm_node::publish_image(cv::Mat image, image_transport::Publisher pub) {
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    cv_ptr->header.stamp = ros::Time::now();
    cv_ptr->encoding = "bgr8";
    cv_ptr->image = image;
    pub.publish(cv_ptr->toImageMsg());
}

void sfm_node::undistort_im(cv::Mat gray_image) {
    cv::Mat dst;
    cv::undistort(gray_image, dst, cameraMatrix, distCoeffs);
    feature_tracking(dst);
}

void sfm_node::feature_tracking(cv::Mat image_1) {
    if (image[0].empty()) image[0] = image_1;
    else image[1] = image_1;

    if (!image[1].empty()) {
        cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        detector->detectAndCompute(image[0], cv::noArray(), keypoints1, descriptors1);
        detector->detectAndCompute(image[1], cv::noArray(), keypoints2, descriptors2);

        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);

        std::vector<cv::Point2f> points1, points2;
        for (auto& match : matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }
        accumulated_keypoints.insert(accumulated_keypoints.end(), keypoints2.begin(), keypoints2.end());
        image[0] = image[1];
    }
}

void sfm_node::triangulate_points(cv::Mat R, cv::Mat t, std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2) {
    cv::Mat proj1 = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat proj2 = cv::Mat::zeros(3, 4, CV_64F);
    R.copyTo(proj2(cv::Rect(0, 0, 3, 3))); // Fixed incorrect syntax
    t.copyTo(proj2.col(3));

    cv::Mat points4D;
    cv::triangulatePoints(proj1, proj2, points1, points2, points4D);

    point_cloud.clear();
    for (int i = 0; i < points4D.cols; i++) {
        point_cloud.emplace_back(
            points4D.at<float>(0, i) / points4D.at<float>(3, i),
            points4D.at<float>(1, i) / points4D.at<float>(3, i),
            points4D.at<float>(2, i) / points4D.at<float>(3, i));
    }
    publish_pointcloud();
}

void sfm_node::publish_pointcloud() {
    sensor_msgs::PointCloud2 cloud_msg;
    cloud_msg.header.stamp = ros::Time::now();
    cloud_msg.header.frame_id = "map";
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(point_cloud.size());
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x"), iter_y(cloud_msg, "y"), iter_z(cloud_msg, "z");
    for (const auto& pt : point_cloud) {
        *iter_x = pt.x; *iter_y = pt.y; *iter_z = pt.z;
        ++iter_x; ++iter_y; ++iter_z;
    }
    pointcloud_pub.publish(cloud_msg);
}

void sfm_node::close_prg() {
    _img_queue->close();
    if (_thread) { _thread->join(); delete _thread; }
    delete _img_queue;
}
