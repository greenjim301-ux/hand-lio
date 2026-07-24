# Livox custom pointcloud format. 与 /livox/lidar (livox_ros_driver::CustomMsg) 逐字段一致。

uint32 offset_time      # offset time relative to the base time, ns
float32 x               # X axis, unit:m
float32 y               # Y axis, unit:m
float32 z               # Z axis, unit:m
uint8 reflectivity      # reflectivity, 0~255
uint8 tag               # livox tag
uint8 line              # laser number in lidar
