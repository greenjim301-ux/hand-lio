#include <ros/ros.h>

#include "hand_lio/HandLioNode.h"

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    
    ros::init(argc, argv, "hand_lio_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    hand_lio::HandLioNode node(nh, pnh);
    ros::spin();
    return 0;
}
