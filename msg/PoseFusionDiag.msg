# pose_fusion_shadow_node 每收到一条 /lidar_pose 发一条，供离线对比用。
# header.stamp 是这条 /lidar_pose 的到达时刻。
Header header
time lidar_stamp            # 这条 /lidar_pose 的消息头时刻
float64 age                 # 到达时刻 - 消息头时刻 [s]，即这次要前推多久
# 新锚点 - 前推预测（世界系）。这就是“前推了 age 秒之后差多少”。
float64 innovation_x
float64 innovation_y
float64 innovation_z
float64 innovation_yaw      # [rad]
float64 slope               # 当前用于 z 前推的坡度 dz/ds
bool first                  # 第一条锚点，innovation 无意义
bool jump                   # 判为 /lidar_pose 自身跳变（输出直接跟过去）。判据见 PoseFusion.h
bool age_exceeds_buffer     # 延迟超过速度缓冲，前推不完整
bool velocity_stale         # 到达时速度源已超时（按零速前推）
string velocity_source      # cmd_vel / motion_status
