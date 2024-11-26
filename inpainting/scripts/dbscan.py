#!/usr/bin/env python3

import time
import rospy
import open3d as o3d
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
from sensor_msgs import point_cloud2
import std_msgs.msg
import struct  # Para empaquetar colores en formato RGB

class DBSCAN:
    def __init__(self):
        # Publicadores
        self.pub_inpainted = rospy.Publisher("/camera/depth_inpainted/points", PointCloud2, queue_size=3)
        self.pub_camera_info = rospy.Publisher("/camera/depth_inpainted/camera_info", CameraInfo, queue_size=3)

        # Suscribirse al tópico de la nube de puntos y la información de la cámara
        self.pointcloud_sub = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pointcloud_callback)
        self.info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.infocb)

        self.pointcloud_msg = None
        self.infomsg = None

    def filter_by_distance(self, points, max_distance=5.0):
        """
        Filtra los puntos que están más allá de una cierta distancia máxima.
        """
        filtered_points = []
        for point in points:
            distance = np.linalg.norm(point[:3])  # Calcular la distancia al origen
            if distance <= max_distance:
                filtered_points.append(point)
        return filtered_points

    def pointcloud_callback(self, msg):
        rospy.loginfo("Recibiendo nube de puntos")
        point_list = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            point_list.append([p[0], p[1], p[2]])

        # Aplicar el filtro basado en la distancia
        max_distance = 1.5  # Ajusta este valor según tus necesidades
        point_list = self.filter_by_distance(point_list, max_distance=max_distance)
        
        rospy.loginfo(f"Después de filtrar, quedan {len(point_list)} puntos dentro de {max_distance} metros.")

        # Crear la nube de puntos Open3D
        if point_list:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(point_list))

            # Reducir los puntos con un filtro de voxelización
            voxel_size = 0.01  # Ajustar según el entorno
            pcd = pcd.voxel_down_sample(voxel_size)

            # Aplicar DBSCAN para detectar clusters (blobs)
            labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=True))
            max_label = labels.max()
            rospy.loginfo(f"Detected {max_label + 1} clusters")

            # Asignar colores a los clusters
            colors = np.zeros((len(labels), 3))  # Inicializar la matriz de colores
            for cluster_label in range(max_label + 1):
                cluster_color = np.random.rand(3)  # Color aleatorio para cada cluster
                colors[labels == cluster_label] = cluster_color

            # Para los puntos que no pertenecen a ningún cluster, los pintamos de gris
            colors[labels == -1] = [0.5, 0.5, 0.5]  # Color gris para ruido o puntos no clasificados

            # Asignar los colores a la nube de puntos
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Convertir a PointCloud2 de ROS, incluyendo los colores empaquetados en formato RGB
            points_with_colors = []
            for i, point in enumerate(np.asarray(pcd.points)):
                color = colors[i]
                r = int(color[0] * 255)
                g = int(color[1] * 255)
                b = int(color[2] * 255)
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]  # Empaquetar RGB en un solo campo
                points_with_colors.append([point[0], point[1], point[2], rgb])

            # Definir los campos para XYZRGB en el mensaje de ROS
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1),  # RGB empaquetado en un solo campo
            ]

            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = msg.header.frame_id

            cloud_msg = point_cloud2.create_cloud(header, fields, points_with_colors)

            # Publicar la nube de puntos procesada con colores
            self.pub_inpainted.publish(cloud_msg)
        else:
            rospy.loginfo("No se recibieron puntos para procesar.")

    def infocb(self, msg):
        self.infomsg = msg


if __name__ == "__main__":
    try:
        rospy.init_node('blob_detection')
        dbscan = DBSCAN()
        time.sleep(3)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
