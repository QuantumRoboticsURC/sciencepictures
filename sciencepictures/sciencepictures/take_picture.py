import rclpy 
import math
from rclpy.node import Node
from sensor_msgs.msg import Imu 
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Int8
from ublox_ubx_msgs.msg import UBXNavHPPosLLH
from rclpy.qos import *
from pynput import keyboard

def euler_from_quaternion(x, y, z, w):
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + y * y)
	roll_x = math.atan2(t0, t1)
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	pitch_y = math.asin(t2)
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (y * y + z * z)
	yaw_z = math.atan2(t3, t4)
	return roll_x, pitch_y, yaw_z # in radians

class PictureTaker(Node):
    def __init__(self):
        super().__init__("node_tester")
        self.subscription = self.create_subscription(UBXNavHPPosLLH,'/gps_base/ubx_nav_hp_pos_llh',self.callback,qos_profile_sensor_data)
        self.s= self.create_subscription(Imu, "/bno055/imu", self.callback2, 10)
        self.pressed = self.create_publisher(Int8,"key_pressed",1)
        self.angle = 0.0
        self.gps_coordinates = [0.0,0.0]
        self.key_code = 0
        self.timer = self.create_timer(0.0001,self.main)
            
    def callback(self, data):
        self.gps_coordinates[0]=data.lat/(10000000)
        self.gps_coordinates[1]=data.lon/(10000000)    
        
    def callback2(self, data):
        quat= Quaternion()
        quat=data.orientation
        angle_x,angle_y,angle_z = euler_from_quaternion(quat.x,quat.y,quat.z,quat.w)
        self.angle = (angle_z+2*math.pi)%(2*math.pi)
        
    def main(self):
        
        listener = keyboard.Listener(on_press=self.on_press) #on_release=self.on_key_release
        listener.start()
        listener.join()
    def on_press(self,key):
        try: 
            k = key.char   
        except:
            k = key.name 
        if k=="f":
            data = Int8()
            data.data = 0
            self.pressed.publish(data)
        elif k=="s":
            data = Int8()
            data.data = 1
            self.pressed.publish(data)

def main(args=None):
	rclpy.init(args=args)
	gps = PictureTaker()
	rclpy.spin(gps)
	gps.destroy_node()
	rclpy.shutdown()

    
if __name__=="__main__":
    main()
    