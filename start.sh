source /opt/ros/melodic/setup.bash
source ~/robot_ws/devel/setup.bash
gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 5; roslaunch zoo_robot robot_lidar.launch; exec bash"' \
--tab -e 'bash -c "sleep 5; roslaunch zoo_control keyboard_teleop.launch; exec bash"' \
--tab -e 'bash -c "sleep 5; roslaunch robot_slam gmapping.launch; exec bash"' \
--tab -e 'bash -c "sleep 5; roslaunch robot_slam view_mapping.launch; exec bash"' \


