#include <ros/ros.h>

#include "hand_lio/HandLioNode.h"

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    
    ros::init(argc, argv, "hand_lio_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    try {
        hand_lio::HandLioNode node(nh, pnh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL("[hand_lio] %s", e.what());
        return 1;
    }
    return 0;
}
