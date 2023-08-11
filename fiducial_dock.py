import cv2
from pupil_apriltags import Detector
from flask import Flask, Response, jsonify
import numpy as np
from simple_pid import PID
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class ControlPublisher(Node):
    def __init__(self):
        super().__init__('control_publisher')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

    def send_control(self, z_c, yaw_c):
        msg = Twist()
        
        # Assuming z_c controls linear velocity in x direction 
        # and yaw_c controls angular velocity in z direction.
        msg.linear.x = z_c
        msg.angular.z = yaw_c

        self.publisher_.publish(msg)
        self.get_logger().info(f'Sending z_c: {z_c}, yaw_c: {yaw_c}')


app = Flask(__name__)

# Create a ROS node and a publisher
rclpy.init(args=None)
control_publisher = ControlPublisher()


# Load your camera calibration parameters
fx, fy, cx, cy = 864.43937962, 863.81651931, 324.25856102, 275.43077411# Load these from your calibration data
dist = 0# Load these from your calibration data

camera_params = (fx, fy, cx, cy)
tag_size = 0.085# Physical size of the tag, e.g., 0.06 for 6x6 cm

z_setpoint = -0.19
x_setpoint = 0.0
yaw_setpoint = 0.0

x_pid = PID(0.5, 0, 0, x_setpoint)
z_pid = PID(0.6, 0, 0, z_setpoint)
yaw_pid = PID(0.15, 0, 0, yaw_setpoint)

z_pid.output_limits = (-0.4, 0.4)
yaw_pid.output_limits = (-1.0, 1.0)

isDocking = False

detector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)

def generate_frames():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera.")
        exit(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray, True, camera_params, tag_size)

        for det in detections:
            for i in range(4):
                p0 = tuple(det.corners[i].astype(int))
                p1 = tuple(det.corners[(i+1) % 4].astype(int))
                cv2.line(frame, p0, p1, (0, 255, 0), 2)

            tag_center = tuple(det.center.astype(int))
            cv2.putText(frame, str(det.tag_id), tag_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            pose_r = det.pose_R
            pose_t = det.pose_t
            x, y, z = pose_t
            theta_rad = np.arctan(x[0] / z[0])
            theta_deg = np.degrees(theta_rad)

            R_inv = np.linalg.inv(pose_r)

            t_inv = -np.dot(R_inv, pose_t)
            x, y, z = t_inv


            #print(f"X: {x[0]}, Y: {y[0]}, Z: {z[0]}")

    	    # Display these values on the frame
            info_str = f"X: {x[0]:.2f}m, Z: {z[0]:.2f}m, Angle: {theta_deg:.2f}"
            cv2.putText(frame, info_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            z_c, yaw_c = z_pid(z[0]), yaw_pid(theta_deg)
            if (isDocking):
                runControl(z_c, yaw_c, theta_deg, z[0])
                info_str = f"Z-PID: {z_c:.2f}m/s, Yaw-PID: {yaw_c:.2f}rad/s, Docking: Yes"
            else:
                info_str = f"Z-PID: {z_c:.2f}m/s, Yaw-PID: {yaw_c:.2f}rad/s, Docking: No"
            cv2.putText(frame, info_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
        <body>
            <img src="/video_feed" width="640" height="480">
            <br>
            <button onclick="startDocking()">Start Docking</button>
            <button onclick="stopDocking()">Stop Docking</button>
            <script>
                function startDocking() {
                    fetch('/start_docking', { method: 'POST' });
                }
                function stopDocking() {
                    fetch('/stop_docking', { method: 'POST' });
                }
            </script>
        </body>
    </html>
    """

@app.route('/start_docking', methods=['POST'])
def start_docking():
    global isDocking
    isDocking = True
    return jsonify(status="Docking started")

@app.route('/stop_docking', methods=['POST'])
def stop_docking():
    global isDocking
    isDocking = False
    return jsonify(status="Docking stopped")


def runControl(z_c, yaw_c, theta, dist):
    global isDocking
    if (abs(dist) - 0.2 > 0.02):
        z_c_m = z_c
    else:
        isDocking = False
        return
    
    if (abs(theta) > 4.0):
        yaw_c_m = yaw_c
    else:
        yaw_c_m = 0.0

    control_publisher.send_control(z_c_m, yaw_c_m)
    return


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    rclpy.spin(control_publisher)
    rclpy.shutdown()
