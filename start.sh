source ~/ros1_ws/devel/setup.bash
gnome-terminal --window -- bash -c "roslaunch aubo_i5_moveit_config moveit_planning_execution.launch robot_ip:=127.0.0.1; exec bash"  \
--tab -- -- bash -c "sleep 5; source ~/anaconda3/etc/profile.d/conda.sh; conda activate foundationpose; /home/hh/anaconda3/envs/foundationpose/bin/python /home/hh/ros1_ws/src/FoundationPose/grasp_final.py; exec bash"  \
# gnome-terminal --window -- bash -c "sleep 5; source ~/anaconda3/etc/profile.d/conda.sh; conda activate foundationpose; /home/hh/anaconda3/envs/foundationpose/bin/python /home/hh/ros1_ws/src/FoundationPose/grasp_final.py; exec bash"  \
# --tab -- bash -c "sleep 5; roslaunch aubo_i5_moveit_config moveit_planning_execution.launch robot_ip:=127.0.0.1; exec bash" \
# --tab -e 'bash -c "sleep 5; roslaunch zoo_control keyboard_teleop.launch; exec bash"' \
# --tab -e 'bash -c "sleep 5; roslaunch robot_slam gmapping.launch; exec bash"' \
# --tab -e 'bash -c "sleep 5; roslaunch robot_slam view_mapping.launch; exec bash"' \


