#include "ros/ros.h"
#include <ros/node_handle.h>
#include <queue>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float32MultiArray.h>
#include <nav_msgs/Odometry.h>
#include <mutex>
#include "balloon_estimation/ekf.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <autoware_msgs/DroneSyn.h>

using namespace std;
using namespace ros;

using namespace balloon_est;

mutex m_bb;
mutex m_uav;
queue<Eigen::Vector4d> bb_buf;
queue<double> time_buf;
queue<int> id_buf;
map<int,Eigen::Vector2d> id_bb;
map<int,double> id_time;
map<int,Eigen::Matrix4d> uav_id_pose;
map<int,double> uav_id_time;

vector<ros::Publisher> pubs;
vector<ros::Publisher> pubs_imu;

Eigen::Matrix<double,8,1> cam_calib;
Eigen::Matrix4d T_CtoI;
Eigen::Matrix4d T_GPStoI;

EKF* ekf;



void UAV_callback(const autoware_msgs::DroneSyn::ConstPtr &msg, int id)
{
  
    // std::cout<<"in uav callback id "<<id<<std::endl;
    Eigen::Vector3d position_GPS = Eigen::Vector3d(msg->gps[0],msg->gps[1],msg->gps[2]);
    Eigen::Quaterniond q_IMU = Eigen::Quaterniond(msg->imu[3],msg->imu[0],msg->imu[1],msg->imu[2]);
    Eigen::Vector3d position_IMU = position_GPS - q_IMU.toRotationMatrix() * T_GPStoI.block(0,3,3,1);
    Eigen::Matrix4d Pose_IMU = Eigen::Matrix4d::Identity();
    Pose_IMU.block(0,0,3,3) = q_IMU.toRotationMatrix();
    Pose_IMU.block(0,3,3,1) = position_IMU;
    Eigen::Matrix4d Pose_cam = Eigen::Matrix4d::Identity();
    Pose_cam = Pose_IMU * T_CtoI;
    Eigen::Matrix3d R_cam = Pose_cam.block(0,0,3,3);
    Eigen::Quaterniond q_cam(R_cam);
    Eigen::Vector3d position_cam = Pose_cam.block(0,3,3,1);
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "world";
    pose.header.stamp = msg->header.stamp;
    pose.pose.orientation.x = q_cam.x();
    pose.pose.orientation.y = q_cam.y();
    pose.pose.orientation.z = q_cam.z();
    pose.pose.orientation.w = q_cam.w();
    pose.pose.position.x = position_cam.x();
    pose.pose.position.y = position_cam.y();
    pose.pose.position.z = position_cam.z();
    pubs[id].publish(pose);

    pose.pose.orientation.x = q_IMU.x();
    pose.pose.orientation.y = q_IMU.y();
    pose.pose.orientation.z = q_IMU.z();
    pose.pose.orientation.w = q_IMU.w();
    pose.pose.position.x = position_IMU.x();
    pose.pose.position.y = position_IMU.y();
    pose.pose.position.z = position_IMU.z();
    pubs_imu[id].publish(pose);


    m_uav.lock();
    if(uav_id_pose.find(id)==uav_id_pose.end())
    {
      uav_id_pose.insert({id,Pose_cam});
      uav_id_time.insert({id,msg->header.stamp.toSec()});
    }
    else
    {
      uav_id_pose[id]=Pose_cam;
      uav_id_time[id]=msg->header.stamp.toSec();
    }
    m_uav.unlock();

  
}



void bounding_box_callback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    // cout<<"in boundboxcallback"<<endl;
    int id = stoi(msg->header.frame_id);
    if(id == 4)
      cout<<"id: "<<id<<endl;
    double t = msg->header.stamp.toSec();
    double left_up_x = msg->pose.orientation.x;
    double left_up_y = msg->pose.orientation.y;
    double right_down_x = msg->pose.orientation.z;
    double right_down_y = msg->pose.orientation.w;
    double center_x = (left_up_x + right_down_x) * 0.5;
    double center_y = (left_up_y + right_down_y) * 0.5;
    m_bb.lock();
    if(!ekf->is_init())
    {
      Eigen::Vector2d bb;
      bb<<center_x,center_y;
      // cout<<"init center x y :"<<center_x<<" "<<center_y<<endl;
      if(id_time.find(id)==id_time.end())
      {
        id_time.insert({id,t});
        id_bb.insert({id,bb});
      }
      else{
        id_time[id]=t;
        id_bb[id]=bb;
      }

    }
    else
    {
      Eigen::Vector4d bb;
      bb<<left_up_x,left_up_y,right_down_x,right_down_y;
      id_buf.push(id);
      bb_buf.push(bb);
      time_buf.push(t);
    }
     m_bb.unlock();
    


}

void get_measurements(Eigen::Vector4d& bb, Meas_pair& meas)
{
  //compute the center of the bounding box;
  double left_up_x = bb(0);
  double left_up_y = bb(1);
  double right_down_x = bb(2);
  double right_down_y = bb(3);
  double center_x = (left_up_x + right_down_x) * 0.5;
  double center_y = (left_up_y + right_down_y) * 0.5;

  vector<double> x;
  vector<double> y;
  x.push_back(left_up_x);
  y.push_back(left_up_y);

  x.push_back(left_up_x);
  y.push_back(right_down_y);

  x.push_back(right_down_x);
  y.push_back(left_up_y);

  x.push_back(right_down_x);
  y.push_back(right_down_y);

  x.push_back(center_x);
  y.push_back(center_y);
  vector<double> as;
  vector<double> es;
  Vector2d uv;
  uv<<center_x,center_y;
  meas.uv = uv;

  cv::Matx33d camK;
  camK(0, 0) = cam_calib(0,0);
  camK(0,1)=0;
  camK(0,2)=cam_calib(2,0);
  camK(1,0)=0;
  camK(1,1)=cam_calib(1,0);
  camK(1,2)=cam_calib(3,0);
  camK(2,0)=0;
  camK(2,1)=0;
  camK(2,2)=1;
  cv::Vec4d camD;
  camD(0) = cam_calib(4,0);
  camD(1) = cam_calib(5,0);
  camD(2) = cam_calib(6,0);
  camD(3) = cam_calib(7,0);

  for(int i=0;i<x.size();i++)
  {
    cv::Point2f pt;
    pt.x = float(x[i]);
    pt.y = float(y[i]);
    cv::Mat mat(1,2,CV_32F);
    mat.at<float>(0, 0) = pt.x;
    mat.at<float>(0, 1) = pt.y;
        
    mat = mat.reshape(2); // Nx1, 2-channel

    cv::undistortPoints(mat, mat, camK, camD);

    cv::Point2f pt_out; //get norm coordinate
    mat = mat.reshape(1); // Nx2, 1-channel
    pt_out.x = mat.at<float>(0, 0);
    pt_out.y = mat.at<float>(0, 1);

   

    // double azi = atan2(pt_out.y,pt_out.x);
    double azi = atan2(1,pt_out.x);

    // double ele = atan2(1,sqrt(pt_out.x * pt_out.x + pt_out.y * pt_out.y));
    double ele = atan2(pt_out.y,sqrt(1+pt_out.x *pt_out.x));
    if(i == 4)
    {
      meas.uv_norm = Vector2d(pt_out.x,pt_out.y);
      cout<<"uv: "<<x[i]<<" "<<y[i]<<endl;
      cout<<"pt x: "<<pt_out.x<<" "<<"pt y: "<<pt_out.y<<endl;
      cout<<"azi: "<<azi<<" ele: "<<ele<<endl;
    }
    // cout<<"i: "<<i<<"pt x: "<<pt_out.x<<" "<<"pt y: "<<pt_out.y<<endl;
    // cout<<"azi: "<<azi<<" ele: "<<ele<<endl;

    as.push_back(azi);
    es.push_back(ele);
  }
  

  meas.azimuth = as[4];
  meas.elevation = es[4];
  
  double azimuth_std = 0;
  double elevation_std = 0;
  for(int i=0;i<as.size()-1;i++)
  {
    azimuth_std = std::max(abs(as[4]-as[i]),azimuth_std);
    elevation_std = std::max(abs(es[4]-es[i]),elevation_std);
  }
  // cout<<"a std: "<<azimuth_std<<" e std: "<<elevation_std<<endl;
  meas.azimuth_std = azimuth_std;
  meas.elevation_std = elevation_std;

}

int main(int argc, char** argv)
{
  ros::init(argc,argv, "balloon_estimation_node");
  ros::NodeHandle nh;

  string bb_topic;
  double vel_noise;
  double noise_multi;
  int uav_number;
  double meas_noise;
  nh.param<string>("/balloon_estimation/bounding_box_topic",bb_topic,"/bounding_box");
  nh.param<double>("/balloon_estimation/vel_noise",vel_noise,0.1); //std;
  nh.param<double>("/balloon_estimation/meas_noise_multiplyer",noise_multi,1.0);
  nh.param<int>("/balloon_estimation/uav_number", uav_number, 10);
  nh.param<double>("/balloon_estimation/meas_noise",meas_noise,1.0);


  // Camera intrinsic properties
  std::vector<double> matrix_k, matrix_d;
  std::vector<double> matrix_k_default = {458.654,457.296,367.215,248.375};
  std::vector<double> matrix_d_default = {-0.28340811,0.07395907,0.00019359,1.76187114e-05};
  nh.param<std::vector<double>>("/balloon_estimation/cam_k", matrix_k, matrix_k_default);
  nh.param<std::vector<double>>("/balloon_estimation/cam_d", matrix_d, matrix_d_default);
  cam_calib << matrix_k.at(0),matrix_k.at(1),matrix_k.at(2),matrix_k.at(3),matrix_d.at(0),matrix_d.at(1),matrix_d.at(2),matrix_d.at(3);

  // Our camera extrinsics transform
  std::vector<double> matrix_TCtoI, matrix_TGPStoI;
  std::vector<double> matrix_TCtoI_default = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  std::vector<double> matrix_TGPStoI_default = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
  // // Read in from ROS, and save into our eigen mat
  nh.param<std::vector<double>>("/balloon_estimation/T_CtoI", matrix_TCtoI, matrix_TCtoI_default);
  T_CtoI << matrix_TCtoI.at(0),matrix_TCtoI.at(1),matrix_TCtoI.at(2),matrix_TCtoI.at(3),
          matrix_TCtoI.at(4),matrix_TCtoI.at(5),matrix_TCtoI.at(6),matrix_TCtoI.at(7),
          matrix_TCtoI.at(8),matrix_TCtoI.at(9),matrix_TCtoI.at(10),matrix_TCtoI.at(11),
          matrix_TCtoI.at(12),matrix_TCtoI.at(13),matrix_TCtoI.at(14),matrix_TCtoI.at(15);

  nh.param<std::vector<double>>("/balloon_estimation/T_GPStoI", matrix_TGPStoI, matrix_TGPStoI_default);

  T_GPStoI <<matrix_TGPStoI.at(0),matrix_TGPStoI.at(1),matrix_TGPStoI.at(2),matrix_TGPStoI.at(3),
          matrix_TGPStoI.at(4),matrix_TGPStoI.at(5),matrix_TGPStoI.at(6),matrix_TGPStoI.at(7),
          matrix_TGPStoI.at(8),matrix_TGPStoI.at(9),matrix_TGPStoI.at(10),matrix_TGPStoI.at(11),
          matrix_TGPStoI.at(12),matrix_TGPStoI.at(13),matrix_TGPStoI.at(14),matrix_TGPStoI.at(15);
  cout<<"T_GPStoI: "<<T_GPStoI<<endl;

  // // Load these into our state
  // Eigen::Matrix<double,7,1> cam_eigen;
  // cam_eigen.block(0,0,4,1) = rot_2_quat(T_CtoI.block(0,0,3,3).transpose());
  // cam_eigen.block(4,0,3,1) = -T_CtoI.block(0,0,3,3).transpose()*T_CtoI.block(0,3,3,1);

  // Insert




  ekf = new EKF(vel_noise,noise_multi,T_CtoI,cam_calib,meas_noise);



  ros::Subscriber bounding_box_sub = nh.subscribe<geometry_msgs::PoseStamped>(bb_topic, 10, &bounding_box_callback);

  ros::Publisher balloon_state_pub = nh.advertise<nav_msgs::Odometry>("/balloon_estimation/odom",2);

  ros::Publisher uav_pose_0_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_0",2);
  pubs.push_back(uav_pose_0_pub);
  ros::Publisher uav_imu_pose_0_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_0_imu",2);
  pubs_imu.push_back(uav_imu_pose_0_pub);
  ros::Publisher uav_pose_1_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_1",2);
  pubs.push_back(uav_pose_1_pub);
  ros::Publisher uav_imu_pose_1_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_1_imu",2);
  pubs_imu.push_back(uav_imu_pose_1_pub);
  ros::Publisher uav_pose_2_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_2",2);
  pubs.push_back(uav_pose_2_pub);
  ros::Publisher uav_imu_pose_2_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_2_imu",2);
  pubs_imu.push_back(uav_imu_pose_2_pub);
  ros::Publisher uav_pose_3_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_3",2);
  pubs.push_back(uav_pose_3_pub);
  ros::Publisher uav_imu_pose_3_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_3_imu",2);
  pubs_imu.push_back(uav_imu_pose_3_pub);
  ros::Publisher uav_pose_4_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_4",2);
  pubs.push_back(uav_pose_4_pub);
  ros::Publisher uav_imu_pose_4_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_4_imu",2);
  pubs_imu.push_back(uav_imu_pose_4_pub);
  ros::Publisher uav_pose_5_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_5",2);
  pubs.push_back(uav_pose_5_pub);
  ros::Publisher uav_imu_pose_5_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_5_imu",2);
  pubs_imu.push_back(uav_imu_pose_5_pub);
  ros::Publisher uav_pose_6_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_6",2);
  pubs.push_back(uav_pose_6_pub);
  ros::Publisher uav_imu_pose_6_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_6_imu",2);
  pubs_imu.push_back(uav_imu_pose_6_pub);
  ros::Publisher uav_pose_7_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_7",2);
  pubs.push_back(uav_pose_7_pub);
  ros::Publisher uav_imu_pose_7_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_7_imu",2);
  pubs_imu.push_back(uav_imu_pose_7_pub);
  ros::Publisher uav_pose_8_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_8",2);
  pubs.push_back(uav_pose_8_pub);
  ros::Publisher uav_imu_pose_8_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_8_imu",2);
  pubs_imu.push_back(uav_imu_pose_8_pub);
  ros::Publisher uav_pose_9_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_9",2);
  pubs.push_back(uav_pose_9_pub);
  ros::Publisher uav_imu_pose_9_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav_9_imu",2);
  pubs_imu.push_back(uav_imu_pose_9_pub);

  ros::Subscriber uav_0_sub = nh.subscribe<autoware_msgs::DroneSyn>("/drone_state_0", 10, boost::bind(&UAV_callback,_1,0));
  ros::Subscriber uav_1_sub = nh.subscribe<autoware_msgs::DroneSyn>("/drone_state_1", 10, boost::bind(&UAV_callback,_1,1));
  ros::Subscriber uav_2_sub = nh.subscribe<autoware_msgs::DroneSyn>("/drone_state_2", 10, boost::bind(&UAV_callback,_1,2));
  ros::Subscriber uav_3_sub = nh.subscribe<autoware_msgs::DroneSyn>("/drone_state_3", 10, boost::bind(&UAV_callback,_1,3));
  ros::Subscriber uav_4_sub = nh.subscribe<autoware_msgs::DroneSyn>("/drone_state_4", 10, boost::bind(&UAV_callback,_1,4));
  ros::Subscriber uav_5_sub = nh.subscribe<autoware_msgs::DroneSyn>("/drone_state_5", 10, boost::bind(&UAV_callback,_1,5));
  ros::Subscriber uav_6_sub = nh.subscribe<autoware_msgs::DroneSyn>("/drone_state_6", 10, boost::bind(&UAV_callback,_1,6));
  ros::Subscriber uav_7_sub = nh.subscribe<autoware_msgs::DroneSyn>("/drone_state_7", 10, boost::bind(&UAV_callback,_1,7));
  ros::Subscriber uav_8_sub = nh.subscribe<autoware_msgs::DroneSyn>("/drone_state_8", 10, boost::bind(&UAV_callback,_1,8));
  ros::Subscriber uav_9_sub = nh.subscribe<autoware_msgs::DroneSyn>("/drone_state_9", 10, boost::bind(&UAV_callback,_1,9));



  //TODO: subscribe Pose balloon; Pose UAV;

  while(ros::ok())
  {
    // cout<<"ros is ok"<<endl;
    if(!ekf->is_init())
    {
      m_bb.lock();
      m_uav.lock();
      // cout<<"id_bb size: "<<id_bb.size()<<" uav_id_pose size: "<<uav_id_pose.size()<<" uav_number: "<<uav_number<<endl;
      map<int,Meas_pair> init_meas;
      double timestamp = -1;
      bool init = false;
      if(id_bb.size()==uav_number&&uav_id_pose.size()==uav_number&&uav_number>1)
      {
        
        auto iter_uav_pose = uav_id_pose.begin();
        auto iter_bb = id_bb.begin();
        auto iter_uav_time = uav_id_time.begin();
        auto iter_time = id_time.begin();
        while(iter_uav_pose!=uav_id_pose.end())
        {

          int id = iter_uav_pose->first;
          timestamp = std::max(timestamp,id_time[id]);
          Meas_pair meas;
          meas.uav_pose = uav_id_pose[id];
          meas.uv = id_bb[id];
          init_meas.insert({id,meas});
          cout<<"before init, id: "<<id<<" bb ts: "<<to_string(id_time[id])<<" uav ts:"<<to_string(uav_id_time[id])<<std::endl;
          iter_uav_pose++;
          if(abs(id_time[id]-uav_id_time[id])>0.5)
          {
            init = false;
            break;
          }
          else
          {
            init = true;
          }

        }
      }
      m_bb.unlock();
      m_uav.unlock();
      if(!init_meas.empty()&& init)
        ekf->initialize(init_meas,timestamp);
     

    }
    
    
    if(!bb_buf.empty())
    {
      m_bb.lock();
      Eigen::Vector4d bb = bb_buf.front();
      int id = id_buf.front();
      double ts = time_buf.front();
      bb_buf.pop();
      id_buf.pop();
      time_buf.pop();
      m_bb.unlock();


      Meas_pair meas;
      bool update=false;
      m_uav.lock();
      if(uav_id_pose.find(id)!=uav_id_pose.end())
      {
        meas.uav_pose = uav_id_pose[id];
        update=true;
      }
      if(abs(uav_id_time[id]-ts)>0.5)
      {
        std::cout<<"time gap is too large! id: "<<id<<" bb ts: "<<to_string(ts)<<" uav ts: "<<to_string(uav_id_time[id])<<std::endl;
        update = false;
      }
      
      m_uav.unlock();
      meas.ts = ts;
      if(update)
      {
        std::cout<<"after init, id: "<<id<<" bb ts: "<<to_string(ts)<<" uav ts: "<<to_string(uav_id_time[id])<<std::endl;
        get_measurements(bb,meas); 
        ekf->process_msg(meas);
      }
    }

    if(ekf->is_init())
    {
      //publish stuff
      State state = ekf->get_State();
      nav_msgs::Odometry msg;
      msg.header.stamp = ros::Time(state.timestamp);
      msg.header.frame_id = "world";
      msg.pose.pose.position.x = state.position.x();
      msg.pose.pose.position.y = state.position.y();
      msg.pose.pose.position.z = state.position.z();
      msg.pose.pose.orientation.x = 0;
      msg.pose.pose.orientation.y = 0;
      msg.pose.pose.orientation.z = 0;
      msg.pose.pose.orientation.w = 1;
      msg.twist.twist.linear.x = state.vel.x();
      msg.twist.twist.linear.y = state.vel.y();
      msg.twist.twist.linear.z = state.vel.z();

      balloon_state_pub.publish(msg);

      // ekf->init = false;

      
    }

    ros::spinOnce();
  }


  delete ekf;

    // Done!
  return EXIT_SUCCESS;


}
