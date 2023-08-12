# simple-fiducial-autodock
Very simple auto docking that tracks a fiducial marker using OpenCV python. It feeds distance and yaw into PID controllers to align with the marker. Can be used for very simple auto docking purposes.

Required Packages:
``pip install opencv-python flask simple_pid pupil_apriltags``

Spawns a webserver at localhost:5000 that is used to control it. Connects to video0 on a linux device.
