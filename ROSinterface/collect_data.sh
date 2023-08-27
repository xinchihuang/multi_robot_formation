num_runs=1000
# Loop to run the command multiple times
for ((i = 1; i <= num_runs; i++)); do
    echo "Run $i: Starting ROS Node..."
    rosrun multi_robot_formation data_collector.py
done