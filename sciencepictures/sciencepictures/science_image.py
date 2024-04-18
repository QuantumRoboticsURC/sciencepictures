import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Int8
from sensor_msgs.msg import Image,Imu
from cv_bridge import CvBridge
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

		self.timer = self.create_timer(0.0001, self.imagen, callback_group=timer_group)
		# Inicializar cámaras
		self.arm_camera = self.initialize_camera()
		self.antenna_camera = self.initialize_camera()

	def quality_callback(self, msg):
		self.quality = msg.data

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
		print(msg.data)
	
	def update_angle(self,msg):
		quat = Quaternion()
		quat = msg.orientation
		angle_x,angle_y,angle_z = euler_from_quaternion(quat.x,quat.y,quat.z,quat.w)
		self.angle = ((angle_z+2*math.pi)%(2*math.pi))*(180/math.pi)
		print(f"Update angle", self.angle)
		
	def imagen(self):
		if self.state == 8:
			if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
				self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
				image_ocv = self.image.get_data()

				# Convertir la imagen a escala de grises
				image_gray = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2GRAY)

				# Aplicar Sobel edge detection
				sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
				sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)

				# Combinar los dos filtros Sobel
				sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

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

				cv2.imwrite('Grises.png',img)
				cv2.imwrite('copia_sobel.png', sobel_combined)

				self.publisher_.publish(self.cv2_to_imgmsg(img))
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
