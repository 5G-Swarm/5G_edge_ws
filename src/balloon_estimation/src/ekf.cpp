#include "balloon_estimation/ekf.h"

namespace balloon_est{

  void EKF::initialize(map<int,Meas_pair> &init_meas, double t)
  {
    if(!init)
    {
      vector<Eigen::Matrix3d> R_CtoW;
      vector<Eigen::Vector3d> p_CinW;
      vector<Eigen::Vector2d> uvs;
      auto iter = init_meas.begin();
      while(iter != init_meas.end())
      {
        
        Meas_pair meas = iter->second;
        
        Eigen::Matrix4d T_CtoW= meas.uav_pose;
        // Eigen::Matrix4d T_CtoW = T_ItoW * T_CtoI_;
        R_CtoW.push_back(T_CtoW.block(0,0,3,3));
        p_CinW.push_back(T_CtoW.block(0,3,3,1));
        
        Eigen::Vector2d uv = meas.uv;
        cout<<"uv: "<<uv.transpose()<<endl;
        
        cv::Matx33d camK;
        camK(0, 0) = cam_intrinsics_(0,0);
        camK(0,1)=0;
        camK(0,2)=cam_intrinsics_(2,0);
        camK(1,0)=0;
        camK(1,1)=cam_intrinsics_(1,0);
        camK(1,2)=cam_intrinsics_(3,0);
        camK(2,0)=0;
        camK(2,1)=0;
        camK(2,2)=1;
        cv::Vec4d camD;
        camD(0) = cam_intrinsics_(4,0);
        camD(1) = cam_intrinsics_(5,0);
        camD(2) = cam_intrinsics_(6,0);
        camD(3) = cam_intrinsics_(7,0);
        cv::Point2f pt;
        pt.x = float(uv.x());
        pt.y = float(uv.y());
        cv::Mat mat(1,2,CV_32F);
        mat.at<float>(0, 0) = pt.x;
        mat.at<float>(0, 1) = pt.y;
            
        mat = mat.reshape(2); // Nx1, 2-channel

        cv::undistortPoints(mat, mat, camK, camD);

        cv::Point2f pt_out; //get norm coordinate
        mat = mat.reshape(1); // Nx2, 1-channel
        pt_out.x = mat.at<float>(0, 0);
        pt_out.y = mat.at<float>(0, 1);
        uvs.push_back(Vector2d(pt_out.x,pt_out.y));

        iter++;

        
      }

      Eigen::Vector3d position;
      if(triangularize_position(R_CtoW,p_CinW,uvs,position))
      {
        state.position = position;
        state.vel = Eigen::Vector3d::Zero();
        state.Cov = 1e-4 * Eigen::Matrix<double,6,6>::Identity();
        state.timestamp = t;
        init =true;
        cout<<"Initialize state success!!..."<<endl;
      }
      else
      {
        cout<<"waiting for intialization....."<<endl;
      }

    }
  }

  void EKF::process_msg(Meas_pair& meas, bool& flag)
  {
    if(!init)
    {
      return;
    }
    if(meas.ts < state.timestamp)
    {
      cout<<"meas ts: "<<to_string(meas.ts)<<" < "<<"current state ts: "<<to_string(state.timestamp)<<std::endl;
      if(abs(meas.ts - state.timestamp) <0.1)
      {
        meas.ts = state.timestamp;
      }
      else
      {
        cout<<"time gap too large, return"<<std::endl;
        return ;
      }
    }
    
    propagation(meas);
    
    update(meas, flag);

  }

  bool EKF::triangularize_position(vector<Eigen::Matrix3d> &R_CtoW,vector<Eigen::Vector3d> &p_CinW,vector<Eigen::Vector2d> &uvs,Eigen::Vector3d& position)
  {

    Eigen::Vector3d position_anchor;
    
    if(single_triangulation(R_CtoW,p_CinW,uvs,position_anchor,position))
    {
      bool flag = single_gaussnewton(R_CtoW,p_CinW,uvs,position_anchor,position);
      return flag;
    }
    else
    {
      return false;
    }

  }

  bool EKF::single_triangulation(vector<Eigen::Matrix3d> &R_CtoW,vector<Eigen::Vector3d> &p_CinW,vector<Eigen::Vector2d> &uvs,Eigen::Vector3d& position_anchor,Eigen::Vector3d& position)
  { 
    int total_meas = 0;
    total_meas=uvs.size();
    
    // Our linear system matrices
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2*total_meas, 3);
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(2*total_meas, 1);

    // Location in the linear system matrices
    size_t c = 0;

    // Get the position of the anchor pose
    int anchor_id = 0;

    // cout<<"feat anchor_img_id: "<<to_string(feat->anchor_img_id)<<endl;
    
    Eigen::Matrix<double,3,3> R_GtoA = R_CtoW[anchor_id].transpose();
    Eigen::Matrix<double,3,1> p_AinG = p_CinW[anchor_id];
    // cout<<"anchor pose: "<<anchor_pose->quat().transpose()<<" "<<anchor_pose->pos().transpose()<<endl; 

    // Loop through each image for this feature
    for(int i =0; i <uvs.size(); i++)
    {
        // Get the position of this image in the global
        Eigen::Matrix<double, 3, 3> R_GtoCi = R_CtoW[i].transpose();
        Eigen::Matrix<double, 3, 1> p_CiinG = p_CinW[i];

        // Convert current position relative to anchor
        Eigen::Matrix<double,3,3> R_AtoCi;
        R_AtoCi.noalias() = R_GtoCi*R_GtoA.transpose();
        Eigen::Matrix<double,3,1> p_CiinA;
        p_CiinA.noalias() = R_GtoA*(p_CiinG-p_AinG);
        // Get the UV coordinate normal
        Eigen::Matrix<double, 3, 1> b_i;
        
        b_i << uvs[i].x(),uvs[i].y(), 1;
        b_i = R_AtoCi.transpose() * b_i;
        b_i = b_i / b_i.norm();
        Eigen::Matrix<double,2,3> Bperp = Eigen::Matrix<double,2,3>::Zero();
        Bperp << -b_i(2, 0), 0, b_i(0, 0), 0, b_i(2, 0), -b_i(1, 0);

        // Append to our linear system
        A.block(2 * c, 0, 2, 3) = Bperp;
        b.block(2 * c, 0, 2, 1).noalias() = Bperp * p_CiinA;
        c++;
        
    }

    // Solve the linear system
    Eigen::MatrixXd p_f = A.colPivHouseholderQr().solve(b);

    // Check A and p_f
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd singularValues;
    singularValues.resize(svd.singularValues().rows(), 1);
    singularValues = svd.singularValues();
    //condition number: max_eigenvalue/min_eigenvalue. by zzq
    double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

    // If we have a bad condition number, or it is too close
    // Then set the flag for bad (i.e. set z-axis to nan)
    //eruoc 1000 0.25 40
    //kaist 5000 0.25 150
    if (std::abs(condA) > 5000|| p_f(2,0) < 0.25 || p_f(2,0) > 150 || std::isnan(p_f.norm())) {
        cout<<"single_triangulation: condition not satisfied, condA: "<<std::abs(condA)<<" p_f.z: "<<p_f(2,0)<<std::endl;
        return false;
    }

    // Store it in our feature object
    position_anchor= p_f;
    position = R_GtoA.transpose()*position_anchor + p_AinG;
    
    Vector3d uv_norm=Vector3d::Zero();
    uv_norm<<p_f(0)/p_f(2),p_f(1)/p_f(2),1;
    // cout<<"predict uv_norm: "<<uv_norm(0)<<" "<<uv_norm(1)<<endl;
    // cout<<"measure uv_norm: "<<uvs[0](0)<<" "<<uvs[0](1)<<endl;

    // cout<<"predict uv_norm: "<<p_A(0)/p_A(2)<<" "<<p_A(1)/p_A(2)<<endl;

    return true;
  }

  bool EKF::single_gaussnewton(vector<Eigen::Matrix3d> &R_CtoW,vector<Eigen::Vector3d> &p_CinW,vector<Eigen::Vector2d> &uvs,Eigen::Vector3d& position_anchor,Eigen::Vector3d& position)
  {
      //Get into inverse depth
    double rho = 1/position_anchor(2);
    double alpha = position_anchor(0)/position_anchor(2);
    double beta = position_anchor(1)/position_anchor(2);

    // Optimization parameters
    double lam = 1e-3;
    double eps = 10000;
    int runs = 0;

    // Variables used in the optimization
    bool recompute = true;
    Eigen::Matrix<double,3,3> Hess = Eigen::Matrix<double,3,3>::Zero();
    Eigen::Matrix<double,3,1> grad = Eigen::Matrix<double,3,1>::Zero();

    // Cost at the last iteration
    double cost_old = compute_error(R_CtoW,p_CinW,uvs,position_anchor,position,alpha,beta,rho);
    int anchor_id = 0;
    // Get the position of the anchor pose
    Eigen::Matrix<double,3,3> R_GtoA = R_CtoW[anchor_id].transpose();
    Eigen::Matrix<double,3,1> p_AinG = p_CinW[anchor_id];

    // Loop till we have either
    // 1. Reached our max iteration count
    // 2. System is unstable
    // 3. System has converged
    while (runs < 20 && lam <  1e10 && eps > 1e-6) {

        // Triggers a recomputation of jacobians/information/gradients
        if (recompute) {

            Hess.setZero();
            grad.setZero();

            double err = 0;
            for(int i =0;i<uvs.size();i++)
            {
                // Get the position of this image in the global
                Eigen::Matrix<double, 3, 3> R_GtoCi = R_CtoW[i].transpose();
                Eigen::Matrix<double, 3, 1> p_CiinG = p_CinW[i];

                // Convert current position relative to anchor
                Eigen::Matrix<double,3,3> R_AtoCi;
                R_AtoCi.noalias() = R_GtoCi*R_GtoA.transpose();
                Eigen::Matrix<double,3,1> p_CiinA;
                p_CiinA.noalias() = R_GtoA*(p_CiinG-p_AinG);
                Eigen::Matrix<double,3,1> p_AinCi;
                p_AinCi.noalias() = -R_AtoCi*p_CiinA;

                double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
                    double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
                    double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
                    // Calculate jacobian
                    double d_z1_d_alpha = (R_AtoCi(0, 0) * hi3 - hi1 * R_AtoCi(2, 0)) / (pow(hi3, 2));
                    double d_z1_d_beta = (R_AtoCi(0, 1) * hi3 - hi1 * R_AtoCi(2, 1)) / (pow(hi3, 2));
                    double d_z1_d_rho = (p_AinCi(0, 0) * hi3 - hi1 * p_AinCi(2, 0)) / (pow(hi3, 2));
                    double d_z2_d_alpha = (R_AtoCi(1, 0) * hi3 - hi2 * R_AtoCi(2, 0)) / (pow(hi3, 2));
                    double d_z2_d_beta = (R_AtoCi(1, 1) * hi3 - hi2 * R_AtoCi(2, 1)) / (pow(hi3, 2));
                    double d_z2_d_rho = (p_AinCi(1, 0) * hi3 - hi2 * p_AinCi(2, 0)) / (pow(hi3, 2));
                    Eigen::Matrix<double, 2, 3> H;
                    H << d_z1_d_alpha, d_z1_d_beta, d_z1_d_rho, d_z2_d_alpha, d_z2_d_beta, d_z2_d_rho;
                    // Calculate residual
                    Eigen::Matrix<float, 2, 1> z;
                    z << hi1 / hi3, hi2 / hi3;
                    Eigen::Matrix<float,2,1> uv_norm;
                    uv_norm<<uvs[i].x(),uvs[i].y();
                    Eigen::Matrix<float, 2, 1> res = uv_norm - z;

                    // Append to our summation variables
                    err += std::pow(res.norm(), 2);
                    grad.noalias() += H.transpose() * res.cast<double>();
                    Hess.noalias() += H.transpose() * H;
            }
            
        }

        // Solve Levenberg iteration
        Eigen::Matrix<double,3,3> Hess_l = Hess;
        for (size_t r=0; r < (size_t)Hess.rows(); r++) {
            Hess_l(r,r) *= (1.0+lam);
        }

        Eigen::Matrix<double,3,1> dx = Hess_l.colPivHouseholderQr().solve(grad);
        //Eigen::Matrix<double,3,1> dx = (Hess+lam*Eigen::MatrixXd::Identity(Hess.rows(), Hess.rows())).colPivHouseholderQr().solve(grad);

        // Check if error has gone down
        double cost = compute_error(R_CtoW,p_CinW,uvs,position_anchor,position,alpha+dx(0,0),beta+dx(1,0),rho+dx(2,0));

        // Debug print
        //cout << "run = " << runs << " | cost = " << dx.norm() << " | lamda = " << lam << " | depth = " << 1/rho << endl;

        // Check if converged
        if (cost <= cost_old && (cost_old-cost)/cost_old < 1e-6) {
            alpha += dx(0, 0);
            beta += dx(1, 0);
            rho += dx(2, 0);
            eps = 0;
            break;
        }

        // If cost is lowered, accept step，and shrink lam to make next step larger
        // Else inflate lambda to make next step smaller (try to make more stable)
        if (cost <= cost_old) {
            recompute = true;
            cost_old = cost;
            alpha += dx(0, 0);
            beta += dx(1, 0);
            rho += dx(2, 0);
            runs++;
            lam = lam/10;
            eps = dx.norm();
        } else {
            recompute = false;
            lam = lam*10;
            continue;
        }
    }

    // Revert to standard, and set to all
    position_anchor(0) = alpha/rho;
    position_anchor(1) = beta/rho;
    position_anchor(2) = 1/rho;

    // Get tangent plane to x_hat
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(position_anchor);
    Eigen::MatrixXd Q = qr.householderQ();

    // Max baseline we have between poses
    double base_line_max = 0.0;

    // Check maximum baseline
    // Loop through each camera for this feature
    //TODO: What the geometry meaning of base_line?
    for(int i=0;i<uvs.size();i++)
    {
        Eigen::Matrix<double,3,1> p_CiinG  = p_CinW[i];
            // Convert current position relative to anchor
            Eigen::Matrix<double,3,1> p_CiinA = R_GtoA*(p_CiinG-p_AinG);
            // Dot product camera pose and nullspace
            double base_line = ((Q.block(0,1,3,2)).transpose() * p_CiinA).norm();
            if (base_line > base_line_max) base_line_max = base_line;
    }

    // Check if this feature is bad or not
    // 1. If the feature is too close
    // 2. If the feature is invalid
    // 3. If the baseline ratio is large
    //euroc  0.25 40 40
    //kaist 0.25 150 500
    if(position_anchor(2) < 0.25
        || position_anchor(2) > 150
        || (position_anchor.norm() / base_line_max) > 500
        || std::isnan(position_anchor.norm())) {
          cout<<"single gaussain: condition not satisifed"<<std::endl;
        return false;
    }
    
    // cout<<"position anchor: "<<position_anchor.transpose()<<endl;
    // cout<<"predict uv_norm: "<<position_anchor(0)/position_anchor(2)<<" "<<position_anchor(1)/position_anchor(2)<<endl;
    // cout<<"measure uv_norm: "<<uvs[0](0)<<" "<<uvs[0](1)<<endl;

    // Finally get position in global frame
    position = R_GtoA.transpose()*position_anchor+ p_AinG;
    return true;
  }

  double EKF::compute_error(vector<Eigen::Matrix3d> &R_CtoW,vector<Eigen::Vector3d> &p_CinW,vector<Eigen::Vector2d> &uvs,Eigen::Vector3d& position_anchor,Eigen::Vector3d& position,
                       double alpha, double beta, double rho)
  {
    // Total error
    double err = 0;

    // Get the position of the anchor pose
    Eigen::Matrix<double,3,3> R_GtoA = R_CtoW[0].transpose();
    Eigen::Matrix<double,3,1> p_AinG = p_CinW[0];

    // Loop through each image for this feature
    for(int i=0; i<uvs.size();i++)
    {
         // Get the position of this image in the global
        Eigen::Matrix<double, 3, 3> R_GtoCi = R_CtoW[i].transpose();
        Eigen::Matrix<double, 3, 1> p_CiinG = p_CinW[i];

        // Convert current position relative to anchor
        Eigen::Matrix<double,3,3> R_AtoCi;
        R_AtoCi.noalias() = R_GtoCi*R_GtoA.transpose();
        Eigen::Matrix<double,3,1> p_CiinA;
        p_CiinA.noalias() = R_GtoA*(p_CiinG-p_AinG);
        Eigen::Matrix<double,3,1> p_AinCi;
        p_AinCi.noalias() = -R_AtoCi*p_CiinA;

        // Middle variables of the system
            //alpha: x/z in anchor ;beta:y/z in anchor; rho: 1/z in anchor
            double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
            double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
            double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);

        // Calculate residual
            Eigen::Matrix<float, 2, 1> z;
            z << hi1 / hi3, hi2 / hi3;
            Eigen::Matrix<float,2,1> uv_norm;
            uv_norm<<uvs[i].x(),uvs[i].y();
            Eigen::Matrix<float, 2, 1> res = uv_norm - z;
            // Append to our summation variables
            err += pow(res.norm(), 2);
    }

    return err;
  }

  void EKF::propagation(Meas_pair& meas)
  {

    
    double dt = meas.ts - state.timestamp;
    std::cout<<"dt: "<<to_string(dt)<<endl;
    state.position = state.position + state.vel * dt;

    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6,6);
    F.block(0,3,3,3) = dt*Eigen::Matrix3d::Identity();

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6,6);
    Q.block(3,3,3,3) = vel_cov * Eigen::Matrix3d::Identity();

    Eigen::MatrixXd Cov = F * state.Cov * F.transpose() + Q;
    state.Cov = (Cov + Cov.transpose()) * 0.5;


    Eigen::VectorXd diags = state.Cov.diagonal();
    bool found_neg = false;
    for(int i=0; i<diags.rows(); i++) {
        if(diags(i) < 0.0) {
            printf("Update - diagonal at %d is %.2f\n" ,i,diags(i));
            found_neg = true;
        }
    }
    assert(!found_neg);

    state.timestamp = meas.ts;
    cout<<"after prop, position: "<<state.position.transpose()<<std::endl;
    cout<<"after prop, vel: "<<state.vel.transpose()<<std::endl;


  }
  
  //  using azimuth as measurement
  // void EKF::update(Meas_pair& meas)
  // {

  //   Eigen::Matrix3d R_VtoW = meas.uav_pose.block(0,0,3,3) ;
  //   Eigen::Vector3d p_VinW = meas.uav_pose.block(0,3,3,1);


  //   Vector2d meas_ae;
  //   meas_ae<<meas.azimuth,meas.elevation;

  //   Vector2d meas_pred;
  //   Eigen::Vector3d p_BinV = R_VtoW.transpose() * (state.position - p_VinW);
  //   Eigen::Vector2d ob_norm;
  //   ob_norm<<p_BinV.x()/p_BinV.z(), p_BinV.y()/p_BinV.z();

  //   // meas_pred(0) = atan2(ob_norm.y(),ob_norm.x());
  //   meas_pred(0) = atan2(1,ob_norm.x());
  //   // meas_pred(1) = atan2(1,sqrt(ob_norm.x() * ob_norm.x() + ob_norm.y() * ob_norm.y()));
  //   meas_pred(1) = atan2(ob_norm.y(), sqrt(ob_norm.x() * ob_norm.x() + 1));
  //   cout<<"a e meas vs. pred: "<<meas_ae.transpose()<<" "<<meas_pred.transpose()<<endl;
  //   Vector2d r = meas_ae-meas_pred;

  //   Matrix<double,2,6> H = Matrix<double,2,6>::Zero();
  //   Matrix<double,3,6> d_vpb_dx = Matrix<double,3,6>::Zero();
  //   Matrix3d d_vpb_d_wpb = R_VtoW.transpose();
  //   d_vpb_dx.block(0,0,3,3) = d_vpb_d_wpb;
    
  //   Matrix<double,2,3> d_norm_d_vpb = Matrix<double,2,3>::Zero();
  //   d_norm_d_vpb<< 1/p_BinV.z(),0,-p_BinV.x()/(p_BinV.z()*p_BinV.z()),
  //                  0, 1/p_BinV.z(), -p_BinV.y()/(p_BinV.z()*p_BinV.z());
    
  //   Matrix<double,1,2> d_a_d_norm = Matrix<double,1,2>::Zero();

  //   double x= ob_norm.x();
  //   double y = ob_norm.y();

  //   // d_a_d_norm<<1.0/(1+pow(y/x,2))*(-y/pow(x,2)), 1.0/(1+pow(y/x,2))*(1/x);
  //   d_a_d_norm<<1.0/(1+(1/pow(x,2))) * -1.0/(pow(x,2)) , 0;
  //   Matrix<double,1,2> d_e_d_norm = Matrix<double,1,2>::Zero();

  //   // d_e_d_norm<<1.0/(1+1.0/(x*x+y*y)) * (-pow((x*x+y*y),-3.0/2.0) * x),
  //               // 1.0/(1+1.0/(x*x+y*y)) * (-pow((x*x+y*y),-3.0/2.0) * y);
  //   d_e_d_norm<<1.0/(1+y*y/(1+x*x)) * -pow((x*x+1),-3/2.0) * x * y,
  //               1.0/(1+y*y/(1+x*x)) * (1/sqrt(x*x+1));
  //   Matrix<double,2,2> d_meas_d_norm = Matrix<double,2,2>::Zero();
  //   d_meas_d_norm.block(0,0,1,2) = d_a_d_norm;
  //   d_meas_d_norm.block(1,0,1,2) = d_e_d_norm;

  //   H = d_meas_d_norm * d_norm_d_vpb * d_vpb_dx;

  //   Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
  //   R(0,0) = meas.azimuth_std * meas.azimuth_std;
  //   R(1,1) = meas.elevation_std * meas.elevation_std; 
  //   R *= scale_factor;
  //   cout<<"R: "<<endl<<R<<endl;
    
  //   Eigen::MatrixXd M = state.Cov * H.transpose();
  //   Eigen::MatrixXd S = H * M + R; 

  //   Eigen::MatrixXd K = M * S.inverse();
  //   Eigen::VectorXd dx = K * r;

  //   double chi2 = r.dot(S.llt().solve(r));
  //   double chi2_check;

  //   chi2_check = chi_squared_table[r.rows()];


  //   // Check if we should delete or not
  //   if(chi2 > chi2_check) {
  //     cout<<"outlier!...."<<endl;
  //       //cout << "featid = " << feat.featid << endl;
  //       //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
  //       //cout << "res = " << endl << res.transpose() << endl;
  //       return;
  //   }

  //   Eigen::MatrixXd Cov = state.Cov - K * M.transpose();
  //   state.Cov = (Cov + Cov.transpose())*0.5;

  //   Eigen::VectorXd diags = state.Cov.diagonal();
  //   bool found_neg = false;
  //   for(int i=0; i<diags.rows(); i++) {
  //       if(diags(i) < 0.0) {
  //           printf("Update - diagonal at %d is %.2f\n" ,i,diags(i));
  //           found_neg = true;
  //       }
  //   }
  //   assert(!found_neg);


  //   std::cout<<"dx rows: "<<dx.rows()<<std::endl;
  //   state.position = state.position + dx.head<3>();
  //   state.vel = state.vel + dx.tail<3>();

  //   cout<<"vel: "<<state.vel.transpose()<<endl;
    
  // }


  //using uv_norm / uv as measurement
  void EKF::update(Meas_pair& meas, bool& flag)
  {

    Eigen::Matrix3d R_VtoW = meas.uav_pose.block(0,0,3,3) ;
    Eigen::Vector3d p_VinW = meas.uav_pose.block(0,3,3,1);
    cout<<"uav position: "<<p_VinW.transpose()<<endl;

    // if(meas.uv(0)<40||meas.uv(0)>600||meas.uv(1)<40||meas.uv(1)>440)
    // {
    //   return;
    // }

    Vector2d uv_norm_meas = meas.uv_norm;

    Vector2d meas_pred;
    Eigen::Vector3d p_BinV = R_VtoW.transpose() * (state.position - p_VinW);
    double distance = p_BinV.norm();
    Eigen::Vector2d ob_norm;
    ob_norm<<p_BinV.x()/p_BinV.z(), p_BinV.y()/p_BinV.z();


    cout<<"uvnorm meas vs. pred: "<<uv_norm_meas.transpose()<<" "<<ob_norm.transpose()<<endl;
    Vector2d z = uv_norm_meas-ob_norm;
    Eigen::Matrix<double,8,1> cam_d = cam_intrinsics_;
    Vector2d uv_dist;
    double r = std::sqrt(ob_norm(0)*ob_norm(0)+ob_norm(1)*ob_norm(1));
    double r_2 = r*r;
    double r_4 = r_2*r_2;
    double x1 = ob_norm(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*ob_norm(0)*ob_norm(1)+cam_d(7)*(r_2+2*ob_norm(0)*ob_norm(0));
    double y1 = ob_norm(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*ob_norm(1)*ob_norm(1))+2*cam_d(7)*ob_norm(0)*ob_norm(1);
    uv_dist(0) = cam_d(0)*x1 + cam_d(2);
    uv_dist(1) = cam_d(1)*y1 + cam_d(3);
    cout<<"uv meas vs. pred: "<<meas.uv.transpose()<<" "<<uv_dist.transpose()<<endl;
    // z = meas.uv - uv_dist;

    Matrix<double,2,6> H = Matrix<double,2,6>::Zero();
    Matrix<double,3,6> d_vpb_dx = Matrix<double,3,6>::Zero();
    Matrix3d d_vpb_d_wpb = R_VtoW.transpose();
    d_vpb_dx.block(0,0,3,3) = d_vpb_d_wpb;
    
    Matrix<double,2,3> d_norm_d_vpb = Matrix<double,2,3>::Zero();
    d_norm_d_vpb<< 1/p_BinV.z(),0,-p_BinV.x()/(p_BinV.z()*p_BinV.z()),
                   0, 1/p_BinV.z(), -p_BinV.y()/(p_BinV.z()*p_BinV.z());
    


    // Jacobian of distorted pixel to normalized pixel
    Eigen::Matrix2d dz_dzn = Eigen::Matrix2d::Identity();
    // double x = ob_norm(0);
    // double y = ob_norm(1);
    // double x_2 = ob_norm(0)*ob_norm(0);
    // double y_2 = ob_norm(1)*ob_norm(1);
    // double x_y = ob_norm(0)*ob_norm(1);
    // dz_dzn(0,0) = cam_d(0)*((1+cam_d(4)*r_2+cam_d(5)*r_4)+(2*cam_d(4)*x_2+4*cam_d(5)*x_2*r)+2*cam_d(6)*y+(2*cam_d(7)*x+4*cam_d(7)*x));
    // dz_dzn(0,1) = cam_d(0)*(2*cam_d(4)*x_y+4*cam_d(5)*x_y*r+2*cam_d(6)*x+2*cam_d(7)*y);
    // dz_dzn(1,0) = cam_d(1)*(2*cam_d(4)*x_y+4*cam_d(5)*x_y*r+2*cam_d(6)*x+2*cam_d(7)*y);
    // dz_dzn(1,1) = cam_d(1)*((1+cam_d(4)*r_2+cam_d(5)*r_4)+(2*cam_d(4)*y_2+4*cam_d(5)*y_2*r)+2*cam_d(7)*x+(2*cam_d(6)*y+4*cam_d(6)*y));





    H = dz_dzn * d_norm_d_vpb * d_vpb_dx;
    double noise = exp(0.05 * distance)-1;
    // cout<<"noise: "<<noise<<" distance:"<<distance<<endl;
    Eigen::Matrix2d R = noise * noise * Eigen::Matrix2d::Identity();
    
    R *= scale_factor;
    // cout<<"R: "<<endl<<R<<endl;
    
    Eigen::MatrixXd M = state.Cov * H.transpose();
    Eigen::MatrixXd S = H * M + R; 

    double chi2 = z.dot(S.llt().solve(z));
    double chi2_check;

    chi2_check = chi_squared_table[z.rows()];


    // Check if we should delete or not
    if(chi2 > chi2_check) {
      cout<<"outlier!...."<<endl;
        //cout << "featid = " << feat.featid << endl;
        //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
        //cout << "res = " << endl << res.transpose() << endl;
        return;
    }




    Eigen::MatrixXd K = M * S.inverse();
    Eigen::VectorXd dx = K * z;
    // if(dx.head<3>().norm()>0.5 || dx.tail<3>().norm()>0.2)
    if(dx.tail<3>().norm()>0.2)
    {
      cout<<"update dx is too large!... "<<endl;
      flag = true;
      // return;
    }

    Eigen::MatrixXd Cov = state.Cov - K * M.transpose();
    state.Cov = (Cov + Cov.transpose())*0.5;

    Eigen::VectorXd diags = state.Cov.diagonal();
    bool found_neg = false;
    for(int i=0; i<diags.rows(); i++) {
        if(diags(i) < 0.0) {
            printf("Update - diagonal at %d is %.2f\n" ,i,diags(i));
            found_neg = true;
        }
    }
    assert(!found_neg);


    // std::cout<<"dx rows: "<<dx.rows()<<std::endl;
    state.position = state.position + dx.head<3>();
    state.vel = state.vel + dx.tail<3>();

    cout<<"after update, position: "<<state.position.transpose()<<std::endl;
    cout<<"after update, vel: "<<state.vel.transpose()<<std::endl;
    
  }

  // //using VB-EKF
  // void EKF::update(Meas_pair& meas, bool& flag)
  // {

  //   Eigen::Matrix3d R_VtoW = meas.uav_pose.block(0,0,3,3) ;
  //   Eigen::Vector3d p_VinW = meas.uav_pose.block(0,3,3,1);
  //   cout<<"uav position: "<<p_VinW.transpose()<<endl;

  //   // if(meas.uv(0)<40||meas.uv(0)>600||meas.uv(1)<40||meas.uv(1)>440)
  //   // {
  //   //   return;
  //   // }

  //   Vector2d uv_norm_meas = meas.uv_norm;

  //   Vector2d meas_pred;
  //   Eigen::Vector3d p_BinV = R_VtoW.transpose() * (state.position - p_VinW);
  //   double distance = p_BinV.norm();
  //   Eigen::Vector2d ob_norm;
  //   ob_norm<<p_BinV.x()/p_BinV.z(), p_BinV.y()/p_BinV.z();


  //   cout<<"uvnorm meas vs. pred: "<<uv_norm_meas.transpose()<<" "<<ob_norm.transpose()<<endl;
  //   Vector2d z = uv_norm_meas-ob_norm;
  //   Eigen::Matrix<double,8,1> cam_d = cam_intrinsics_;
  //   Vector2d uv_dist;
  //   double r = std::sqrt(ob_norm(0)*ob_norm(0)+ob_norm(1)*ob_norm(1));
  //   double r_2 = r*r;
  //   double r_4 = r_2*r_2;
  //   double x1 = ob_norm(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*ob_norm(0)*ob_norm(1)+cam_d(7)*(r_2+2*ob_norm(0)*ob_norm(0));
  //   double y1 = ob_norm(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*ob_norm(1)*ob_norm(1))+2*cam_d(7)*ob_norm(0)*ob_norm(1);
  //   uv_dist(0) = cam_d(0)*x1 + cam_d(2);
  //   uv_dist(1) = cam_d(1)*y1 + cam_d(3);
  //   cout<<"uv meas vs. pred: "<<meas.uv.transpose()<<" "<<uv_dist.transpose()<<endl;
  //   // z = meas.uv - uv_dist;

  //   Matrix<double,2,6> H = Matrix<double,2,6>::Zero();
  //   Matrix<double,3,6> d_vpb_dx = Matrix<double,3,6>::Zero();
  //   Matrix3d d_vpb_d_wpb = R_VtoW.transpose();
  //   d_vpb_dx.block(0,0,3,3) = d_vpb_d_wpb;
    
  //   Matrix<double,2,3> d_norm_d_vpb = Matrix<double,2,3>::Zero();
  //   d_norm_d_vpb<< 1/p_BinV.z(),0,-p_BinV.x()/(p_BinV.z()*p_BinV.z()),
  //                  0, 1/p_BinV.z(), -p_BinV.y()/(p_BinV.z()*p_BinV.z());
    


  //   // Jacobian of distorted pixel to normalized pixel
  //   Eigen::Matrix2d dz_dzn = Eigen::Matrix2d::Identity();
  //   // double x = ob_norm(0);
  //   // double y = ob_norm(1);
  //   // double x_2 = ob_norm(0)*ob_norm(0);
  //   // double y_2 = ob_norm(1)*ob_norm(1);
  //   // double x_y = ob_norm(0)*ob_norm(1);
  //   // dz_dzn(0,0) = cam_d(0)*((1+cam_d(4)*r_2+cam_d(5)*r_4)+(2*cam_d(4)*x_2+4*cam_d(5)*x_2*r)+2*cam_d(6)*y+(2*cam_d(7)*x+4*cam_d(7)*x));
  //   // dz_dzn(0,1) = cam_d(0)*(2*cam_d(4)*x_y+4*cam_d(5)*x_y*r+2*cam_d(6)*x+2*cam_d(7)*y);
  //   // dz_dzn(1,0) = cam_d(1)*(2*cam_d(4)*x_y+4*cam_d(5)*x_y*r+2*cam_d(6)*x+2*cam_d(7)*y);
  //   // dz_dzn(1,1) = cam_d(1)*((1+cam_d(4)*r_2+cam_d(5)*r_4)+(2*cam_d(4)*y_2+4*cam_d(5)*y_2*r)+2*cam_d(7)*x+(2*cam_d(6)*y+4*cam_d(6)*y));





  //   H = dz_dzn * d_norm_d_vpb * d_vpb_dx;
  //   double noise = exp(0.05 * distance)-1;
  //   cout<<"noise: "<<noise<<endl;
  //   Eigen::Matrix2d R = noise * noise * Eigen::Matrix2d::Identity();
    
  //   R *= scale_factor;
  //   cout<<"R: "<<endl<<R<<endl;
    
  //   Eigen::MatrixXd M = state.Cov * H.transpose();
  //   // Eigen::MatrixXd S = H * M + R; 

  //   Eigen::MatrixXd X = z * z.transpose() + H * M;
  //   Eigen::MatrixXd Omega = (1-a)* v * V + X;
  //   v = (1-a)*v + a*(d-1) + 1;
  //   V = Omega / v;

  //   Eigen::MatrixXd S = H * M + V;





  //   double chi2 = z.dot(S.llt().solve(z));
  //   double chi2_check;

  //   chi2_check = chi_squared_table[z.rows()];


  //   // Check if we should delete or not
  //   if(chi2 > chi2_check) {
  //     cout<<"outlier!...."<<endl;
  //       //cout << "featid = " << feat.featid << endl;
  //       //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
  //       //cout << "res = " << endl << res.transpose() << endl;
  //       return;
  //   }




  //   Eigen::MatrixXd K = M * S.inverse();
  //   Eigen::VectorXd dx = K * z;
  //   if(dx.head<3>().norm()>0.5 || dx.tail<3>().norm()>0.5)
  //   {
  //     cout<<"update dx is too large!... "<<endl;
  //     flag = true;
  //     // return;
  //   }

  //   Eigen::MatrixXd Cov = (Eigen::MatrixXd::Identity(6,6) - K * H) * state.Cov *
  //                         ((Eigen::MatrixXd::Identity(6,6) - K * H)).transpose() + 
  //                         K * V * K.transpose();
  //   // Eigen::MatrixXd Cov = state.Cov - K * M.transpose();
  //   state.Cov = (Cov + Cov.transpose())*0.5;

  //   Eigen::VectorXd diags = state.Cov.diagonal();
  //   bool found_neg = false;
  //   for(int i=0; i<diags.rows(); i++) {
  //       if(diags(i) < 0.0) {
  //           printf("Update - diagonal at %d is %.2f\n" ,i,diags(i));
  //           found_neg = true;
  //       }
  //   }
  //   assert(!found_neg);


  //   std::cout<<"dx rows: "<<dx.rows()<<std::endl;
  //   state.position = state.position + dx.head<3>();
  //   state.vel = state.vel + dx.tail<3>();

  //   cout<<"after update, position: "<<state.position.transpose()<<std::endl;
  //   cout<<"after update, vel: "<<state.vel.transpose()<<std::endl;
    
  // }

  State EKF::get_State()
  { 
    return state;

  }
}