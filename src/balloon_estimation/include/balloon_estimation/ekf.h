#include <Eigen/Dense>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf/message_filter.h>
#include <tf_conversions/tf_eigen.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <boost/math/distributions/chi_squared.hpp>


using namespace ros;
using namespace std;
using namespace Eigen;
namespace balloon_est{
  
  struct State{
    double timestamp;
    double last_timestamp;
    Eigen::Vector3d position;
    Eigen::Vector3d vel;
    Eigen::MatrixXd Cov;

  };

  struct Meas_pair
  {
    double ts;
    Eigen::Vector2d uv;
    Eigen::Matrix4d uav_pose;
    Eigen::Vector2d uv_norm;
    double azimuth;
    double elevation;
    double azimuth_std;
    double elevation_std;
  };
  
  class EKF{

    public:
      EKF(double vel_noise, double noise_multi,Eigen::Matrix4d &T_CtoI, Eigen::Matrix<double,8,1>& cam_intrinsics, double meas_noise)
      {
        vel_cov = vel_noise * vel_noise;
        scale_factor = noise_multi;
        init = false;
        T_CtoI_ = T_CtoI;
        cam_intrinsics_ = cam_intrinsics;
        tf_listen = new tf::TransformListener();
        tf_broad = new tf::TransformBroadcaster();
        meas_noise_ = meas_noise;

        v = 10.0;
        V = Matrix2d::Identity() * meas_noise_ * meas_noise_;

        

        for (int i = 1; i < 500; i++) {
              boost::math::chi_squared chi_squared_dist(i); //DoF of chi_squared_dist. from 1 to 500
              //https://www.boost.org/doc/libs/1_35_0/libs/math/doc/sf_and_dist/html/math_toolkit/dist/dist_ref/nmp.html#math.dist.cdf
              // it returns a value x (here is chi_squared_table[i]) such that cdf(dist, x) == p(here is 0.95)
              // so that it means 95% of distribution is below x.
              // so if we have a value, which is larger than chi_square_table[i],
              // then, this value are very likely to be an outlier.  by zzq
          chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.98);
        }
      }


    
      void initialize(map<int,Meas_pair> &init_meas, double t);

      void process_msg(Meas_pair& meas, bool& flag);

      bool is_init(){return init;}

      State get_State();
     
    public:
      void propagation(Meas_pair& meas);
      

      void update(Meas_pair& meas, bool& flag);

      bool triangularize_position(vector<Eigen::Matrix3d> &R_CtoW,vector<Eigen::Vector3d> &p_CinW,vector<Eigen::Vector2d> &uvs,Eigen::Vector3d& position);

      bool single_triangulation(vector<Eigen::Matrix3d> &R_CtoW,vector<Eigen::Vector3d> &p_CinW,vector<Eigen::Vector2d> &uvs,Eigen::Vector3d& position_anchor,Eigen::Vector3d& position);

      bool single_gaussnewton(vector<Eigen::Matrix3d> &R_CtoW,vector<Eigen::Vector3d> &p_CinW,vector<Eigen::Vector2d> &uvs,Eigen::Vector3d& position_anchor,Eigen::Vector3d& position);

      double compute_error(vector<Eigen::Matrix3d> &R_CtoW,vector<Eigen::Vector3d> &p_CinW,vector<Eigen::Vector2d> &uvs,Eigen::Vector3d& position_anchor,Eigen::Vector3d& position,
                             double alpha, double beta, double rho);



      bool init;

      State state;

      tf::TransformListener* tf_listen;
      tf::TransformBroadcaster* tf_broad;

      double vel_cov;
      double scale_factor;
      double meas_noise_;
      Eigen::Matrix4d T_CtoI_;
      Eigen::Matrix<double,8,1> cam_intrinsics_;
      std::map<int, double> chi_squared_table;
      double a = 0.4;
      double d = 2;
      double v ;
      Eigen::Matrix2d V;



  };
}

