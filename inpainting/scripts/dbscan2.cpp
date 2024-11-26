#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Header.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include <cmath>
#include <random>

class DBSCAN {
public:
    DBSCAN() : nh_("~"), gen_(rd_()), dis_(0, 255) {
        pub_inpainted_ = nh_.advertise<sensor_msgs::PointCloud2>("/camera/depth_inpainted/points", 1);
        pub_camera_info_ = nh_.advertise<sensor_msgs::CameraInfo>("/camera/depth_inpainted/camera_info", 1);

        pointcloud_sub_ = nh_.subscribe("/camera/depth/color/points", 1, &DBSCAN::pointcloudCallback, this);
        info_sub_ = nh_.subscribe("/camera/depth/camera_info", 1, &DBSCAN::infocb, this);
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher pub_inpainted_;
    ros::Publisher pub_camera_info_;
    ros::Subscriber pointcloud_sub_;
    ros::Subscriber info_sub_;

    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_int_distribution<> dis_;

    void infocb(const sensor_msgs::CameraInfo::ConstPtr& msg) {}

    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        cloud = filterByDistance(cloud, 2.0);  // Ajusta según la necesidad

        // Aplicar filtro de voxelización
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud);
        voxel.setLeafSize(0.02f, 0.02f, 0.02f);  // Tamaño ajustado de voxel
        voxel.filter(*cloud);

        // Filtro estadístico para eliminar ruido
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*cloud);

        // Clustering
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud);
        
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.05);
        ec.setMinClusterSize(20);     // Ajusta el tamaño mínimo para capturar clusters más pequeños
        ec.setMaxClusterSize(5000);   // Asegura capturar clusters grandes de obstáculos
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        // Asignar colores a los clusters
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (const auto& indices : cluster_indices) {
            uint8_t r = dis_(gen_);
            uint8_t g = dis_(gen_);
            uint8_t b = dis_(gen_);

            for (int idx : indices.indices) {
                pcl::PointXYZRGB point;
                point.x = cloud->points[idx].x;
                point.y = cloud->points[idx].y;
                point.z = cloud->points[idx].z;
                point.r = r;
                point.g = g;
                point.b = b;
                colored_cloud->points.push_back(point);
            }
        }
        
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*colored_cloud, output);
        output.header = msg->header;
        pub_inpainted_.publish(output);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filterByDistance(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double max_distance) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        std::copy_if(cloud->points.begin(), cloud->points.end(), std::back_inserter(filtered_cloud->points),
                     [max_distance](const pcl::PointXYZ& point) {
                         return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z) <= max_distance;
                     });
        return filtered_cloud;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "dbscan_node");
    DBSCAN dbscan;
    ros::spin();
    return 0;
}
