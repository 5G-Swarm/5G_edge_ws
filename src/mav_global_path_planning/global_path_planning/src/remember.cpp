 void position_callback(const nav_msgs::Odometry::ConstPtr& msg)
            {
                if(global_costmap.info.width!=0)
                {