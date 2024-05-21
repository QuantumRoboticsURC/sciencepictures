import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Int8
from sensor_msgs.msg import Image,Imu
from cv_bridge import CvBridge
from ublox_ubx_msgs.msg import UBXNavHPPosLLH
from rclpy.qos import *

import pyzed.sl as sl
import cv2
import numpy as np
import math
import os

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

class Science_image(Node):
    def __init__(self):
        super().__init__("Science_image")
        timer_group = MutuallyExclusiveCallbackGroup()
        listener_group = ReentrantCallbackGroup()
        self.publisher_ = self.create_publisher(Image, 'camera/image', 10)
        self.arm_cam_publisher = self.create_publisher(Image, "/arm_cam", 10)
        self.ant_cam_publisher = self.create_publisher(Image, "/ant_cam", 10)
        self.create_subscription(Int8, "image_quality", self.quality_callback, 1)
        self.quality = 18
        self.state = -1
        self.angle = 0.0
        self.bridge = CvBridge()
        self.state_pub = self.create_publisher(Int8, "state", 1)
        self.create_subscription(Int8, "state", self.update_state, 1, callback_group=listener_group)
        self.create_subscription(Imu, "/bno055/imu", self.update_angle, 10,callback_group=listener_group)
        self.create_subscription(UBXNavHPPosLLH,'/gps_base/ubx_nav_hp_pos_llh',self.callback,qos_profile_sensor_data)
        self.create_subscription(Int8,"key_pressed",self.pressed,1)
        self.key = -1
        self.gps_coordinates = [0.0,0.0]
        #self.timer_coords = self.create_timer(0.0001,self.start_listener)

        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER

        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Camera Open: {}. Exiting program.".format(repr(status)))
            exit()

        self.runtime_parameters = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.other_condition = False

        self.timer = self.create_timer(0.0001, self.imagen, callback_group=timer_group)
        # Inicializar cámaras
        #self.arm_camera = self.initialize_camera()
        #self.antenna_camera = self.initialize_camera()
        
        # Inicializar variables para el cálculo de la escala
        self.camera_to_rover_distance = 0.5  # Distancia de la cámara al borde central del robot (en metros)
        self.width = 0.84  # Ancho del robot (en metros)
        self.start_angle = math.atan2((self.width/2), self.camera_to_rover_distance)
        self.pixels = 680 # Ancho de la imagen en píxeles

        self.coordinate_1 = None
        self.coordinate_2 = None

    def pressed(self,msg):
        self.key = msg.data
        if self.key == 0:
            print("------------------------------------------------------")
            self.coordinate_1 = self.gps_coordinates
            print(self.coordinate_1)
        elif self.key == 1:
            print("------------------------------------------------------")
            self.coordinate_2 = self.gps_coordinates
            print(self.coordinate_2) 
    
    def quality_callback(self, msg):
        self.quality = msg.data

    def callback(self, data):
        self.gps_coordinates[0]=data.lat/(10000000)
        self.gps_coordinates[1]=data.lon/(10000000)
        #print("En el callback")    
        

    def initialize_camera(self):
        num_cameras = 5  # Rango de posibles indices generados
        for index in range(num_cameras):
            camera = cv2.VideoCapture(index)
            if camera.isOpened():
                self.get_logger().info(f"Cámara encontrada en el índice {index}")
                return camera
            camera.release()
        self.get_logger().error("No se encontraron cámaras disponibles.")
        return None
    
    def cv2_to_imgmsg(self, image):
        msg = self.bridge.cv2_to_imgmsg(image, encoding = "bgra8")
        return msg

    def cv2_to_imgmsg_resized(self, image, scale_percent):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        msg = self.bridge.cv2_to_imgmsg(resized_image, encoding = "bgra8")
        return msg

    def update_state(self, msg):
        self.state = msg.data
        #print(msg.data)
    
    def update_angle(self,msg):
        quat = Quaternion()
        quat = msg.orientation
        angle_x,angle_y,angle_z = euler_from_quaternion(quat.x,quat.y,quat.z,quat.w)
        self.angle = ((angle_z+2*math.pi)%(2*math.pi))*(180/math.pi)
        #print(f"Update angle", self.angle)

    def distanceBetweenCoords(self,current_lat,current_long,target_lat,target_long):
        earth_radius = 6371000
        dLat = np.deg2rad(target_lat-current_lat)
        dLon = np.deg2rad(target_long-current_long)
        current_lat=np.deg2rad(current_lat)
        target_lat=np.deg2rad(target_lat)
        
        a = np.sin(dLat/2)*np.sin(dLat/2)+np.sin(dLon/2)*np.sin(dLon/2)*np.cos(current_lat)*np.cos(target_lat)

        c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
        return earth_radius*c
        
    def calculate_scale(self):
        # Obtener la nueva distancia
        if(self.coordinate_1 and self.coordinate_2):
            new_distance = self.distanceBetweenCoords(self.coordinate_1[0],self.coordinate_1[1],self.coordinate_2[0],self.coordinate_2[1])
            print("Distancia ",new_distance)
            if new_distance <1:
                return 1
            scale = (((self.pixels / 2) * self.camera_to_rover_distance) / ((self.width / 2) * new_distance))
            return scale
        return None

    def imagen(self):
        if self.state == 8:
            if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                image_ocv = self.image.get_data()

                # Convertir la imagen a escala de grises
                image_gray = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2GRAY)

                # Aplicar Sobel edge detection
                sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
                edges = cv2.Canny(np.uint8(np.absolute(sobel_x)), threshold1=30, threshold2=100)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
                min_length = 200
                angle_threshold = 5
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2 - x1)*2 + (y2 - y1)*2)
                    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                    if length > min_length and (angle < angle_threshold or angle > 180 - angle_threshold):
                        cv2.line(image_ocv, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)

                # Combinar los dos filtros Sobel
                #sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

                imu=self.angle
                print(f"Timer", self.angle)
                img = image_ocv
                brujula = cv2.imread("/home/shikur_orin/ros2_ws/src/sciencepictures/sciencepictures/sciencepictures/Brujula.jpeg")
                escala = cv2.imread("/home/shikur_orin/ros2_ws/src/sciencepictures/sciencepictures/sciencepictures/escala.jpeg")
                filas, columnas,canales = img.shape
                tam=int(filas/3)
                brujula2 = cv2.resize(brujula, (tam,tam))
                escala2= cv2.resize(escala,(int(columnas/4),int(filas/32)))
                filas, columnas,canales = brujula2.shape
                roi = img[0:filas,0:columnas]
                brujula_gris = cv2.cvtColor(brujula2,cv2.COLOR_BGR2GRAY)
                ret, brujula_byn =cv2.threshold(brujula_gris,190,255,cv2.THRESH_BINARY)
                brujula3 = cv2.bitwise_not(brujula_byn)
                M = cv2.getRotationMatrix2D((columnas//2,filas//2),imu-90,1)
                brujula4 = cv2.warpAffine(brujula3,M,(columnas,filas))
                brujulafin=cv2.bitwise_not(brujula4)

                img_fondo=cv2.bitwise_and(roi, roi, mask = brujulafin)
                img[0:filas,0:columnas] = img_fondo

                filas, columnas,canales = img.shape
                filas2, columnas2,canales2 = escala2.shape
                escala3 = cv2.cvtColor(escala2,cv2.COLOR_BGR2GRAY)
                ret, escala4 =cv2.threshold(escala3,190,255,cv2.THRESH_BINARY)
                roi2 = img[filas-filas2-20:filas-20,columnas-columnas2-20:columnas-20]
                img_fondo2=cv2.bitwise_and(roi2, roi2, mask = escala4)
                img[filas-filas2-20:filas-20,columnas-columnas2-20:columnas-20]=img_fondo2

                # Mostrar la escala en la imagen
                scale = self.calculate_scale()
                scale_text = f"Scale: {scale} m/px"
                print(scale)
                cv2.putText(img, scale_text, (20, filas - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imwrite('Grises.png',img)
                cv2.imwrite('copia_sobel.png', image_ocv)

                self.publisher_.publish(self.cv2_to_imgmsg(img))
                if(self.pressed==2):
                    cv2.imwrite('nueva.png', img)
                print("se supone xd")
                filas, columnas,canales = brujula2.shape			

        elif self.state == 9:
            if self.arm_camera:
                ret_arm, frame_arm = self.arm_camera.read()
                if ret_arm:
                    self.arm_cam_publisher.publish(self.cv2_to_imgmsg_resized(frame_arm, self.quality))
            if self.antenna_camera:
                ret_ant, frame_ant = self.antenna_camera.read()
                if ret_ant:
                    self.ant_cam_publisher.publish(self.cv2_to_imgmsg_resized(frame_ant, self.quality))


def main(args=None):
    rclpy.init(args=args)
    det = Science_image()
    executor = MultiThreadedExecutor()
    executor.add_node(det)
    executor.spin()
    det.zed.close()
    det.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
