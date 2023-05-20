#include <functional>
#include <mutex>
#include <thread>
#include <string>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <map_msgs/OccupancyGridUpdate.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
#include "bspline.cpp"
#include <queue>
#include <unordered_map>
// #include <boost>
#include <time.h>
#include "KDTree/KDTree.hpp"
#include "dynamic_voronoi/dynamicvoronoi.h"
#include <ctime>
//目前实现一个地图的基础路径规划，之后在考虑把定位添加进去
//现在开始把无人机的信息接进去

#define INF INT_MAX
using namespace std;
//test:: todo
namespace frontier_detection
{

    struct Node{
    float cost;
    int index;
    bool operator<(const Node &a) const {
        return cost < a.cost;
    }
    };
    class frontierdetection
    {
        public:
            frontierdetection(ros::NodeHandle private_nh)
            {
                getCostmap = false;
                NODE = private_nh;
                // costmap_sub_constructing_voronoi_ = private_nh.subscribe<nav_msgs::OccupancyGrid>("/map", 1, &frontierdetection::costmap_callback_constructing_voronoi, this); 
                costmap_sub_global_ = private_nh.subscribe<nav_msgs::OccupancyGrid>("/map_pub_cyclic", 1, &frontierdetection::CostmapSubCallback_global, this);
                goal_sub_global_ = private_nh.subscribe<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1, &frontierdetection::goal_callback, this);
                costmap_with_path_ = private_nh.advertise<nav_msgs::OccupancyGrid>("/costmap_with_path", 1);
                robot_position_pub_=private_nh.advertise<geometry_msgs::PoseStamped>("/robot_position", 1);
                global_path_pub_ = private_nh.advertise<nav_msgs::Path>("/global_path", 1);
                global_target_pub_ = private_nh.advertise<geometry_msgs::PoseStamped>("/target_set_xy",1);

                planning_vxy_pub_ = private_nh.advertise<geometry_msgs::Twist>("/plan_vxy", 1);
                obstacle_xy_sub_ = private_nh.subscribe<nav_msgs::Path>( "/obstacle_xy", 1, &frontierdetection::obstacle_xy_callback, this);
                target_xy_sub_ = private_nh.subscribe<geometry_msgs::Pose>( "/target_xy", 1, &frontierdetection::target_xy_callback, this);
                place_xy_sub = private_nh.subscribe<nav_msgs::Path>( "/place_xy", 1, &frontierdetection::place_xy_callback, this);
            }


        private:

            void obstacle_xy_callback(const nav_msgs::Path::ConstPtr& msg)
            {
                
                if(global_costmap.info.width!=0)
                {
                     nav_msgs::Path obstacle_xy_;
                    obstacle_xy_ = *msg;
                    obstacle_x_position.clear();
                    obstacle_y_position.clear();
                    obstacle_id.clear();
                    for(auto i:obstacle_xy_.poses)
                    {
                        obstacle_x_position.push_back(i.pose.position.x);
                        obstacle_y_position.push_back(i.pose.position.y);
                        obstacle_id.push_back((int)((i.pose.position.x-x0_place)/image_resolution)+(int)(global_costmap.info.width)*(int)((i.pose.position.y-y0_place)/image_resolution));
                        int obstacle=(int)((i.pose.position.x-x0_place)/image_resolution)+(int)(global_costmap.info.width*((int)i.pose.position.y-y0_place)/image_resolution);
                    }
                    get_obstacle_xy=true;
                }
            }
            void target_xy_callback(const geometry_msgs::Pose::ConstPtr& msg)
            {
                if(global_costmap.info.width!=0)
                {
                    geometry_msgs::Pose target_xy_;
                    target_xy_ = *msg;
                    target_x_position = target_xy_.position.x;
                    target_y_position = target_xy_.position.y;
                    robot_index = (int)((target_x_position-x0_place)/image_resolution)+(int)global_costmap.info.width*int((target_y_position-y0_place)/image_resolution);//todo:get the goal idx;
                }
                
            }
            void place_xy_callback(const nav_msgs::Path::ConstPtr& msg)
            {
                if(global_costmap.info.width==0)
                    {
                        cout<<"waiting for global_costmap"<<endl;}
                else
                {
                    if(!get_place_xy)
                    {
                        nav_msgs::Path place_xy_;
                        place_xy_ = *msg;
                        place_x_position.clear();
                        place_y_position.clear();
                        for(auto i:place_xy_.poses)
                        {
                            place_x_position.push_back(i.pose.position.x);
                            place_y_position.push_back(i.pose.position.y);
                        }
                        // for(auto i:place_x_position)
                        //     cout<<"place_x_position:"<<i<<" "<<endl;
                        // for(auto i:place_y_position)
                        //     cout<<"place_y_position:"<<i<<" "<<endl;
                        for(auto i:place_x_position)
                        {
                            if(x0_place>i)
                            {
                                x0_place = i;
                            }
                        }
                        for(auto i:place_y_position)
                        {
                            if(y0_place>i)
                            {
                                y0_place = i;
                            }
                        }
                        double t=0;
                        x0_place=x0_place-t;
                        y0_place=y0_place-t;
                        cout<<"x0_place:"<<x0_place<<" "<<endl;
                        cout<<"y0_place:"<<y0_place<<" "<<endl;
                        

                            playground_ids.clear();
                            int playground_id;
                            for(int i=0;i<place_x_position.size();i++)
                            {
                                // cout<<"x:"<<place_x_position[i]-x0_place<<endl;
                                // cout<<"y:"<<place_y_position[i]-y0_place<<endl;
                                playground_id=(place_x_position[i]-x0_place)/image_resolution+(int)global_costmap.info.width*(int)((place_y_position[i]-y0_place)/image_resolution);
                                playground_ids.push_back(playground_id);
                            }



                            // cout<<"x:"<<place_x_position[3]-x0_place<<endl;
                            // cout<<"y:"<<place_y_position[3]-y0_place<<endl;
                            // cout<<"playground_id:"<<playground_ids[3]<<endl;
                            // cout<<global_costmap.info.width<<endl;
                            get_place_xy = true;
                        
                    }
                
                }
                
            }

            void find_closest_voronoi_point(int& robot_id, int& closest_voronoi_id, std::vector<bool> is_voronoi)
            {
                
                int a[4]={1,(int)global_costmap.info.width,-1,-(int)global_costmap.info.width};
                std::unordered_map<int, int> visited, unvisited;
                std::queue<int> id;
                int current;
                std::cout<<"start finding closest voronoi point"<<std::endl;
                if(is_voronoi[robot_id])
                {
                    closest_voronoi_id = robot_id;
                    std::cout<<"find closest voronoi point:"<<closest_voronoi_id<<std::endl;
                    return;
                }
                int count=10000;
                if(!is_voronoi[robot_id]&&count>0)
                {
                    count--;
                    id.push(robot_id);
                    while (id.size()) 
                    {
                        current = id.front();
                        id.pop();
                        for(int i=0;i<4;i++)
                        {
                            int neighbor_id=current+a[i];
                            // std::cout<<"neighbor_id:"<<neighbor_id<<std::endl;
                            if(!(visited[neighbor_id])&&neighbor_id>0&&neighbor_id<is_voronoi.size())
                            {
                                if(is_voronoi[current])
                                {
                                    closest_voronoi_id=current;
                                    std::cout<<"find closest voronoi point:"<<closest_voronoi_id<<std::endl;
                                    // id=queue<int>();
                                    return;
                                }
                                    id.push(neighbor_id);
                                    ++visited[neighbor_id];
                            }
                        }
                    }
                }
            }
            
            double calculte_distance(int x, int y)
            {
                
                double distance = sqrt(pow(x/(int)global_costmap.info.width-y/(int)global_costmap.info.width,2)+pow(x%(int)global_costmap.info.width-y%(int)global_costmap.info.width,2));
                return distance;
            }
            void plot_playground(nav_msgs::OccupancyGrid& global_costmap, std::vector<int> playground_ids)
            {
                int x0,y0,x1,y1;
                double k;

                    // x0 = playground_ids[0]%(int)global_costmap.info.width;
                    // y0 = playground_ids[0]/(int)global_costmap.info.width;
                    // x1 = playground_ids[1]%(int)global_costmap.info.width;
                    // y1 = playground_ids[1]/(int)global_costmap.info.width;
                    // cout<<"x0:"<<x0<<" "<<"y0:"<<y0<<endl;
                    // cout<<"x1:"<<x1<<" "<<"y1:"<<y1<<endl;
                    // k=double(y1-y0)/double(x1-x0);
                    // cout<<"k:"<<k<<endl;
                    // int step=1000;
                    // for(int j=0;j<step;j++)
                    // {
                    //     int idx,x_temp,y_temp;
                    //     x_temp = x0+(x1-x0)*j/step;
                    //     y_temp = y0+k*(x_temp-x0);

                    //     // cout<<"x_temp:"<<x_temp<<" "<<"y_temp:"<<y_temp<<endl;
                    //     global_costmap.data[x_temp+y_temp*global_costmap.info.width]=100;
                    // }
                    



                for(int i=0;i<2;i++)//plot the 1-2 and 3-0
                {
                    x0 = playground_ids[2*i+1]%(int)global_costmap.info.width;
                    y0 = playground_ids[2*i+1]/(int)global_costmap.info.width;
                    x1 = playground_ids[2*(i+1)%playground_ids.size()]%(int)global_costmap.info.width;
                    y1 = playground_ids[2*(i+1)%playground_ids.size()]/(int)global_costmap.info.width;
                    k=(y1-y0)/(x1-x0);
                    int step=1000;
                    for(int j=0;j<step;j++)
                    {
                        int idx,x_temp,y_temp;
                        x_temp = x0+k*(x1-x0)*j/step;
                        y_temp = y0+k*(y1-y0)*j/step;
                        global_costmap.data[x_temp+y_temp*global_costmap.info.width]=100;
                    }
                }

                for(int i=0;i<2;i++)//plot the 0-1 and 2-3
                {
                    x0 = playground_ids[2*i]%(int)global_costmap.info.width;
                    y0 = playground_ids[2*i]/(int)global_costmap.info.width;
                    x1 = playground_ids[(2*i+1)%playground_ids.size()]%(int)global_costmap.info.width;
                    y1 = playground_ids[(2*i+1)%playground_ids.size()]/(int)global_costmap.info.width;
                    k=double(y1-y0)/double(x1-x0);
                    int step=1000;
                    for(int j=0;j<step;j++)
                    {
                        int idx,x_temp,y_temp;
                        x_temp = x0+(x1-x0)*j/step;
                        y_temp = y0+k*(x_temp-x0);
                        global_costmap.data[x_temp+y_temp*global_costmap.info.width]=100;
                    }
                }
            }
            void CostmapSubCallback_global(const nav_msgs::OccupancyGrid::ConstPtr& map_msg)
            {
                
                if(goal_x_position!=0||goal_y_position!=0)
                {
                    geometry_msgs::PoseStamped goal_position;
                    goal_position.header=map_msg->header;
                    goal_position.pose.position.x = goal_x_position;
                    goal_position.pose.position.y = goal_y_position;
                    std::cout<<"goal_position_x:"<<goal_x_position<<endl;
                    std::cout<<"goal_position_y:"<<goal_y_position<<endl;
                    global_target_pub_.publish(goal_position);

                }
                global_costmap = *map_msg;
                // for(int i=0;i<global_costmap.data.size();i++)
                // {
                //     if(global_costmap.data[i]==100)
                //     {
                //         obstacle_id.push_back(i);
                //     }
                // }
                // for(int i=0;i<global_costmap.info.width;i++)
                // {
                //     global_costmap.data[i]=100;
                //     global_costmap.data[global_costmap.data.size()-i-1]=100;
                // }
                // for(int j=0; j<global_costmap.info.height; j++)
                // {
                //     global_costmap.data[j*global_costmap.info.width]=100;
                //     global_costmap.data[j*global_costmap.info.width+global_costmap.info.width-1]=100;
                // }
                if(get_place_xy&&get_obstacle_xy)
                {
                    // global_costmap.data[playground_ids[3]]=100;
                    for(int i=0; i<playground_ids.size();i++)
                    {
                        // cout<<"playground_ids"<<" i:"<<i<<" "<<playground_ids[i]<<endl;
                        // cout<<"test1:"<<playground_ids[i]/global_costmap.info.width<<endl;
                        global_costmap.data[playground_ids[i]]=100;
                    }
                    plot_playground(global_costmap,playground_ids);


                    //~code:inflation playground
                    // for(int i=0;i<global_costmap.data.size();i++)
                    // {
                    //     if(global_costmap.data[i]==100)
                    //     {
                    //         playground_ids_sum.push_back(i);
                    //     }
                    // }
                    // int a[4]={-1,1,(int)global_costmap.info.width,-(int)global_costmap.info.width};
                    // bool exit=0;
                    // queue<int> obstacle;
                    // std::unordered_map<int, int> visited;
                    // for(auto i:playground_ids_sum)
                    // {
                    //     obstacle.push(i);
                    //     int current_obstacle_id=i;
                    //     while(!obstacle.empty())
                    //     {
                    //         int temp_id=obstacle.front();
                    //         visited[temp_id]=1;
                    //         obstacle.pop();
                    //             for(int j=0;j<4;j++)
                    //             {
                    //                 int temp_id_new=temp_id+a[j];
                    //                 if(temp_id_new>=0&&temp_id_new<global_costmap.data.size()&&calculte_distance(temp_id_new,current_obstacle_id)<=playground_obstacle_radius&&!visited[temp_id_new])
                    //                 {
                    //                     visited[temp_id_new]++;
                    //                     obstacle.push(temp_id_new);
                    //                     global_costmap.data[temp_id_new]=100;
                    //                 }
                    //             }
                    //     }
                    // }

                    // if(!obstacle_id.empty())
                    // {
                    //     for(auto i:obstacle_id)
                    //     {
                    //         global_costmap.data[i]=100;
                    //     }
                    // }

                    //~code:inflation obstacle
                    int a[4]={-1,1,(int)global_costmap.info.width,-(int)global_costmap.info.width};
                    bool exit=0;
                    queue<int> obstacle;
                    std::unordered_map<int, int> visited;
                    for(auto i:obstacle_id)
                    {
                        visited.clear();
                        obstacle.push(i);
                        int current_obstacle_id=i;
                        global_costmap.data[current_obstacle_id]=100;
                        while(!obstacle.empty())
                        {
                            int temp_id=obstacle.front();
                            visited[temp_id]=1;
                            obstacle.pop();
                                for(int j=0;j<4;j++)
                                {
                                    int temp_id_new=temp_id+a[j];
                                    if(temp_id_new>=0&&temp_id_new<global_costmap.data.size()&&calculte_distance(temp_id_new,current_obstacle_id)<=obstacle_radius&&!visited[temp_id_new])
                                    {
                                        visited[temp_id_new]++;
                                        obstacle.push(temp_id_new);
                                        global_costmap.data[temp_id_new]=100;
                                    }
                                }
                        }
                    }



                    //~code:generate voronoi

                    voronoi_sizeX = global_costmap.info.width;
                    voronoi_sizeY = global_costmap.info.height;
                    bool **map1=NULL;
                    map1 = new bool*[voronoi_sizeX];

                    for (int x=0; x<voronoi_sizeX; x++) 
                    {
                        (map1)[x] = new bool[voronoi_sizeY];
                    }
                    for (int x=0; x<voronoi_sizeX; x++) 
                    {
                        for (int y=0; y<voronoi_sizeY; y++) {
                            if (((int)global_costmap.data[x+y*voronoi_sizeX]<100) && ((int)global_costmap.data[x+y*voronoi_sizeX]>=0))
                                map1[x][y]=false; //cell is free
                            else map1[x][y] = true; // cell is occupied
                            }
                    }

                    clock_t startime, endtime;
                    startime=clock();//记录开始时间
                    // if(!voronoi_generated)
                    // {
                        voronoi.initializeMap(voronoi_sizeX, voronoi_sizeY, map1);
                        voronoi.update(); // update distance map and Voronoi diagram
                        bool doPrune = false, doPruneAlternative = false;
                        if (doPrune) voronoi.prune();  // prune the Voronoi
                        if (doPruneAlternative) voronoi.updateAlternativePrunedDiagram();  // prune the Voronoi
                        voronoi_generated = true;
                    // }*5
                    endtime=clock();//记录结束时间


                    double tot_time = (double)(endtime - startime);
                    cout<<"total time:"<<tot_time<<endl;

                    OGM.resize(global_costmap.data.size());
                    // for(int i=0;i<global_costmap.data.size();i++)
                    // {
                    //     if(global_costmap.data[i]==10)
                    //         OGM[i]=true;
                    //     else
                    //         OGM[i]=false;
                    // }

                    for(int i = 0; i < voronoi_sizeX ; i++)
                    {
                        for(int j = 0; j < voronoi_sizeY ; j++)
                        {
                            if(voronoi.isVoronoi(i,j)==1)
                            {
                                // global_costmap.data[i+j*voronoi_sizeX]=10;
                                OGM[i+j*voronoi_sizeX]=true;
                            }
                            else
                                OGM[i+j*voronoi_sizeX]=false;
                        }
                    }


                    for (int x=0; x<voronoi_sizeX; x++) 
                    {
                        delete [] map1[x];
                    }
                    delete [] map1;

                    

                    std::vector<int> path;
                    std::vector<std::vector<double> > path_separted;
                    // if(goal_index!=0)
                    // {
                    //     path_planning(robot_index, goal_index, global_costmap, path);
                    // }
                    // for(int i=10; i<100; i++)
                    // {
                    //     global_costmap.data[i+robot_index]=100;
                    // }
                    int closest_robot_in_voronoi_idx=0;
                    int closest_goal_in_voronoi_idx=0;
                    std::cout<<"robot_index:"<<robot_index<<std::endl;
                    find_closest_voronoi_point(robot_index, closest_robot_in_voronoi_idx, OGM);
                    if(goal_index!=0)
                    {
                        cout<<"goal index:"<<goal_index<<endl;
                        find_closest_voronoi_point(goal_index, closest_goal_in_voronoi_idx, OGM);
                        
                        // for(int i=10; i<100; i++)
                        // {
                        //     global_costmap.data[i+closest_robot_in_voronoi_idx]=100;
                        // }
                        if(sqrt(pow(robot_index/global_costmap.info.width-goal_index/global_costmap.info.width,2)+pow(robot_index%global_costmap.info.width-goal_index%global_costmap.info.width,2)<=distance_to_goal))
                        {
                            cout<<"goal_reached"<<endl;
                            geometry_msgs::Twist vel_msg;
                            vel_msg.linear.x=0;
                            vel_msg.linear.y=0;
                            planning_vxy_pub_.publish(vel_msg);
                        }
                        else
                        {
                            path_planning(closest_robot_in_voronoi_idx, closest_goal_in_voronoi_idx, global_costmap, path);
                            if(!path.empty())
                            {
                                // for(auto i: path)
                                //     global_costmap.data[i]=100;
                                for(int i=0;i<path.size();i+=3)//todo  the path points to be smoothed
                                {
                                    std::vector<double> path_temp;
                                    path_temp.push_back((double)(path[i]%voronoi_sizeX));
                                    path_temp.push_back((double)(path[i]/voronoi_sizeX));
                                    path_separted.push_back(path_temp);
                                }

                                nav_msgs::Path Path_published;
                                geometry_msgs::PoseStamped this_pose_stamped;
                                std::vector<int> refined_path = Path_smooth(path_separted);
                                refined_path.push_back(goal_index);
                                for(int i = 0; i < refined_path.size(); i++){
                                    int temp_path_id=path[i];
                                    Path.push_back(temp_path_id);
                                    this_pose_stamped.pose.position.x = temp_path_id%global_costmap.info.width;
                                    this_pose_stamped.pose.position.y = global_costmap.info.height-temp_path_id/global_costmap.info.width-1;
                                    Path_published.header=global_costmap.header;
                                    // std::cout<<"u:"<<this_pose_stamped.pose.position.x<<" v:"<<this_pose_stamped.pose.position.y<<std::endl;
                                    Path_published.poses.push_back(this_pose_stamped);
                                    
                                }
                                global_path_pub_.publish(Path_published);
                                
                                for(auto i: refined_path)
                                {
                                    global_costmap.data[i]=100;
                                }
                                geometry_msgs::Twist vel_msg;

                                int start_index = refined_path[0];
                                int goal_index = refined_path[1];
                                int start_index_x = start_index%global_costmap.info.width;
                                int start_index_y = start_index/global_costmap.info.width;
                                int goal_index_x = goal_index%global_costmap.info.width;
                                int goal_index_y = goal_index/global_costmap.info.width;
                                double distance = sqrt(pow(goal_index_x-start_index_x,2)+pow(goal_index_y-start_index_y,2));
                                if(distance>=2)
                                {
                                    double vx = (goal_index_x-start_index_x)/distance;
                                    double vy = (goal_index_y-start_index_y)/distance;
                                    global_costmap.data[start_index]=100;
                                    global_costmap.data[goal_index]=100;
                                    cout<<"vx:"<<vx<<endl;
                                    cout<<"vy:"<<vy<<endl;
                                    if(vx<-1)
                                        vx=-1;
                                    else if(vx>1)
                                        vx=1;
                                    if(vy<-1)
                                        vy=-1;
                                    else if(vy>1)
                                        vy=1;
                                    vel_msg.linear.x=vx;
                                    vel_msg.linear.y=vy;
                                    planning_vxy_pub_.publish(vel_msg);
                                }
                                else
                                {
                                    cout<<"goal reached!"<<endl;
                                    vel_msg.linear.x=0;
                                    vel_msg.linear.y=0;
                                    planning_vxy_pub_.publish(vel_msg);
                                }

                            }
                        }


                        // for(int i=0;i<5;i++)
                        // {
                        //     global_costmap.data[i+closest_robot_in_voronoi_idx]=100;
                        //     global_costmap.data[i+closest_goal_in_voronoi_idx]=100;
                        // }

                    }
                }
                if(robot_index!=0)
                {
                    for(int i=-2 ; i<3 ; i++)
                    {
                        for(int j=-2 ; j<3; j++)
                        {
                            if(i+j*global_costmap.info.width+robot_index>=0 && i+j*global_costmap.info.width+robot_index<global_costmap.data.size())
                            {
                                global_costmap.data[i+goal_index]=80;
                            }
                        }
                    }
                }
                if(goal_index!=0)
                {
                    for(int i=-4 ; i<5 ; i++)
                    {
                        if(i+goal_index>=0 && i+goal_index<global_costmap.data.size())
                        {
                            global_costmap.data[i+goal_index]=80;
                        }
                        
                    }
                    for(int i=-4 ; i<5 ; i++)
                    {
                        if(i*global_costmap.info.width+goal_index>=0 && i*global_costmap.info.width+goal_index<global_costmap.data.size())
                            global_costmap.data[i*global_costmap.info.width+goal_index]=80;
                    }
                }
                    
                    costmap_with_path_.publish(global_costmap);
                    
                    
                
                
            }

            std::vector<int> Path_smooth(std::vector<std::vector<double> >& pair_path)
            {
                int refinded_path_size=pair_path.size();
                std::vector<std::vector<double> > refined_path;
                refined_path=pair_path;
                double tolerance_=0.5;
                double change=10000;
                int count=1000;
                double d1, d2;
                while(change>tolerance_&&count>0)
                {
                    count--;
                    change=0;
                    for(int i=1; i<refined_path.size()-1; i++)
                    {
                        for(int j=0; j<refined_path[i].size(); j++)
                        {
                            d1=weight_data * (refined_path[i][j] - pair_path[i][j]);
                            d2=weight_smooth * (refined_path[i-1][j]-refined_path[i+1][j]-2*refined_path[i][j]);
                            change = abs(d1+d2);
                            // std::cout<<"change"<<change<<std::endl;
                            refined_path[i][j] += d1+d2;
                        }
                    }
                }
                std::vector<int> result;
                int temp=0;
                for(int i=0; i<refined_path.size(); i++)
                {
                    // for(int j=0; j<refined_path[i].size(); j++)
                    // {
                    //     std::cout<<"refined_path:"<<refined_path[i][j];
                    // }
                    temp=(int)(refined_path[i][0]+refined_path[i][1]*global_costmap.info.width);
                    result.push_back(temp);
                    // std::cout<<std::endl;
                }
                return result;
            }
            point_t get_kdtree_index(point_t point, KDTree tree)
            {
                point_t point_temp;
                point_temp.push_back(point[0]);
                point_temp.push_back(point[1]);
                point_t closest_point = tree.nearest_point(point_temp);
                return closest_point;
            }
            void indextocell(int index, int &x, int &y)
            {
                x = index % global_costmap.info.width;
                y = index / global_costmap.info.width;
            }
            void maptoworld(int x, int y, double &wx, double &wy)
            {
                // wx = origin_x_ + (x + 0.5) * resolution_;
                // wy = origin_y_ + (y + 0.5) * resolution_;
                int u=x, v=global_costmap.info.height-y-1;
                wx = (741-v)/20;
                wy = (1165-u)/20;
            }
            bool isInBounds(int x, int y)
            {
                if( x < 0 || y < 0 || x >= global_costmap.info.width || y >= global_costmap.info.height)//todo:changed with "width and height"
                    return false;
                return true;
            }
            vector<int> get_neighbors(int current_cell)
            {   
                vector<int> neighborIndexes;
                
                for (int i = -1; i <= 1; i++)
                {
                    for (int j = -1; j <= 1; j++)
                    {
                        unsigned tmp1, tmp2;
                        // costmap_->indexToCells(current_cell, tmp1, tmp2);
                        tmp1 = current_cell%global_costmap.info.width;
                        tmp2 = current_cell/global_costmap.info.width;
                        int nextX = tmp1 + i;
                        int nextY = tmp2 + j;
                        // int nextIndex = costmap_->getIndex(nextX, nextY);
                        int nextIndex = nextX+nextY*global_costmap.info.width;

                        if(!( i == 0 && j == 0) && isInBounds(nextX, nextY) && OGM[nextIndex])
                        {
                            neighborIndexes.push_back(nextIndex);
                        }
                    }
                }
                return neighborIndexes;
            }
            double getMoveCost(int firstIndex, int secondIndex)
            {
                unsigned int tmp1, tmp2;
                // costmap_->indexToCells(firstIndex, tmp1, tmp2);
                tmp1 = firstIndex%global_costmap.info.width;
                tmp2 = firstIndex/global_costmap.info.width;
                int firstXCord = tmp1,firstYCord = tmp2;
                // costmap_->indexToCells(secondIndex, tmp1, tmp2);
                tmp1 = secondIndex%global_costmap.info.width;
                tmp2 = secondIndex/global_costmap.info.width;
                int secondXCord = tmp1, secondYCord = tmp2;
                
                int difference = abs(firstXCord - secondXCord) + abs(firstYCord - secondYCord);
                // Error checking
                if(difference != 1 && difference != 2){
                    ROS_ERROR("Astar global planner: Error in getMoveCost - difference not valid");
                    return 1.0;
                }
                if(difference == 1)
                    return 1.0;
                else
                    return 1.4;
            }
            double getHeuristic(int cell_index, int goal_index)
            {
                unsigned int tmp1, tmp2;
                tmp1 = cell_index%global_costmap.info.width;
                tmp2 = cell_index/global_costmap.info.width;
                int startX = tmp1, startY = tmp2;
                tmp1 = goal_index%global_costmap.info.width;
                tmp2 = goal_index/global_costmap.info.width;
                int goalX = tmp1, goalY = tmp2;
                
                return abs(goalY - startY) + abs(goalX - startX);
            }

            bool path_planning(int start_index, int goal_index, nav_msgs::OccupancyGrid& global_costmap, std::vector<int>& plan_result)
            {
                // ROS_INFO("Got a start: %.2f, %.2f, and a goal: %.2f, %.2f", start.pose.position.x, start.pose.position.y, 
                // goal.pose.position.x,goal.pose.position.y);
                // ros::Time time_1 = ros::Time::now();
                // double wx = start.pose.position.x;
                // double wy = start.pose.position.y;
                // unsigned int start_x, start_y;
                // costmap_->worldToMap(wx, wy, start_x, start_y);
                // int start_index = costmap_->getIndex(start_x, start_y);

                
                // wx = goal.pose.position.x;
                // wy = goal.pose.position.y;

                // unsigned int goal_x, goal_y;
                // costmap_->worldToMap(wx, wy, goal_x, goal_y);
                // int goal_index = costmap_->getIndex(goal_x, goal_y);
                int map_size = global_costmap.data.size();
                vector<float> gCosts(map_size, INF);
                vector<int> cameFrom(map_size, -1);
                
                multiset<Node> priority_costs;
                
                gCosts[start_index] = 0;
                
                Node currentNode;
                currentNode.index = start_index;
                currentNode.cost = gCosts[start_index] + 0;
                priority_costs.insert(currentNode);
                vector<geometry_msgs::PoseStamped> plan;
                vector<geometry_msgs::PoseStamped> replan;
                
                plan.clear();
                replan.clear();
                std::cout<<"begin planning"<<std::endl;
                
                while(!priority_costs.empty())
                {
                    // Take the element from the top
                    currentNode = *priority_costs.begin();
                    //Delete the element from the top
                    priority_costs.erase(priority_costs.begin());
                    if (currentNode.index == goal_index){
                        break;
                    }
                    // Get neighbors
                    vector<int> neighborIndexes = get_neighbors(currentNode.index);
                    
                    for(int i = 0; i < neighborIndexes.size(); i++){
                        if(cameFrom[neighborIndexes[i]] == -1){
                        gCosts[neighborIndexes[i]] = gCosts[currentNode.index] + getMoveCost(currentNode.index, neighborIndexes[i]);
                        Node nextNode;
                        nextNode.index = neighborIndexes[i];
                        //nextNode.cost = gCosts[neighborIndexes[i]];    //Dijkstra Algorithm
                        nextNode.cost = gCosts[neighborIndexes[i]] + getHeuristic(neighborIndexes[i], goal_index);    //A* Algorithm
                        cameFrom[neighborIndexes[i]] = currentNode.index;
                        priority_costs.insert(nextNode);
                        }
                    }
                }
                
                if(cameFrom[goal_index] == -1){
                    cout << "Goal not reachable, failed making a global path." << endl;
                    return false;
                }
                
                if(start_index == goal_index)
                {
                    cout<<"start_index==goal_index"<<endl;
                    return false;
                }
                    
                //Finding the best path
                vector<int> bestPath;
                currentNode.index = goal_index;
                while(currentNode.index != start_index){
                    bestPath.push_back(cameFrom[currentNode.index]);
                    currentNode.index = cameFrom[currentNode.index];
                }
                reverse(bestPath.begin(), bestPath.end());
                std::cout<<"before: "<<Path.size()<<std::endl;
                Path.clear();
                int path;
                for(int i = 0; i < bestPath.size(); i=i+3){//todo i+=3 change the distance among path
                    path = bestPath[i];
                    Path.push_back(path);
                }
                std::cout<<"after: "<<Path.size()<<std::endl;
                cout << "/***********/" << "bestPath.size():" << bestPath.size() << "*****" <<"Path.size():" << Path.size() << endl;
                plan_result=Path;
                return true;
            
            }

            void goal_callback(const geometry_msgs::PoseStamped::ConstPtr& goal_pose)
            {
                global_goal_pose=*goal_pose;
                int a[3];//calculate the index of goal position;
                a[0]=(global_goal_pose.pose.position.x)*100000;
                a[1]=(global_goal_pose.pose.position.y)*100000;
                a[2]=global_costmap.info.resolution*100000;
                goal_index = (a[1]/a[2])*global_costmap.info.width+a[0]/a[2];
                std::cout<<"goal_index:"<<goal_index<<std::endl;
                std::cout<<"goal_value:"<<(int)global_costmap.data[goal_index]<<std::endl;
                // std::vector<int> path;
                // path_planning(robot_index, goal_index, global_costmap, path);
                
                if(x0_place!=0||y0_place!=0)
                {                
                    goal_x_position = goal_index%(int)global_costmap.info.width*global_costmap.info.resolution+x0_place;//  /2:the resolution of the costmap
                    goal_y_position = goal_index/(int)global_costmap.info.width*global_costmap.info.resolution+y0_place;
                }


            }

            void GetRobotpose(std::string iden_frame_id,geometry_msgs::PoseStamped& global_local_pose, ros::Time timestamp)
            {
            tf::StampedTransform transform;
            geometry_msgs::PoseStamped iden_pose;
            iden_pose.header.frame_id = iden_frame_id;
            iden_pose.header.stamp = ros::Time::now(); 
            iden_pose.pose.orientation.w = 1;
            tf_listener_.waitForTransform("/robot0/map",iden_frame_id, ros::Time(0), ros::Duration(2.0));
            tf_listener_.lookupTransform( "/robot0/map",iden_frame_id, ros::Time(0), transform);
            global_local_pose.pose.position.x=transform.getOrigin().x();
            global_local_pose.pose.position.y=transform.getOrigin().y();
            global_local_pose.pose.position.z=transform.getOrigin().z();
            }

            bool getCostmap;
            ros::NodeHandle NODE;
            ros::Subscriber costmap_sub_constructing_voronoi_;
            ros::Subscriber costmap_sub_global_;
            ros::Subscriber goal_sub_global_;
            ros::Subscriber position_sub_global_;
            ros::Subscriber obstacle_xy_sub_;
            ros::Subscriber target_xy_sub_;
            ros::Subscriber place_xy_sub;
            ros::Publisher costmap_with_path_;
            ros::Publisher robot_position_pub_;
            ros::Publisher global_path_pub_;
            ros::Publisher global_target_pub_;
            ros::Publisher planning_vxy_pub_;
            nav_msgs::OccupancyGrid global_costmap;
            nav_msgs::OccupancyGrid global_costmap_for_voronoi;
            nav_msgs::OccupancyGrid local_costmap;
            geometry_msgs::PoseStamped global_local_pose;
            geometry_msgs::PoseStamped global_goal_pose;
            geometry_msgs::PoseStamped closest_frontier_point;
            std::vector<bool> OGM;//check whether the cell is known
            vector<int> Path;
            vector<int> Path_smoothed;
            int robot_index_x=0;
            int robot_index_y=0;
            DynamicVoronoi voronoi;
            bool voronoi_generated=false;//generate voronoi once
            int voronoi_sizeX, voronoi_sizeY;
            double weight_data=0.005;//the two argument below is for path smooth
            double weight_smooth=0.0000001;

            std::mutex lock_costmap;
            std::thread tf_thread_;
            tf::TransformListener tf_listener_;
            int robot_index=0;
            int goal_index=0;
            int temp_goal_index;
            vector<int> obstacle_id;
            vector<int> playground_ids;
            vector<int> playground_ids_sum;
            double playground_obstacle_radius=3;
            double obstacle_radius=20;
            vector<double> obstacle_x_position;//the x position of mav obstacle
            vector<double> obstacle_y_position;//the y position of mav obstacle
            vector<double> place_x_position;//the x position of the playground
            vector<double> place_y_position;//the y position of the playground
            double target_x_position;//the x position of the target circle
            double target_y_position;//the y position of the target circle
            double image_resolution=0.5;
            double x0_place=0;// the min x position of the playground
            double y0_place=0;// the min y position of the playground
            bool get_obstacle_xy=false;//whether get the obstacle xy position message
            bool get_place_xy=false;

            int goal_x_position = 0;//the x position of the target place 
            int goal_y_position = 0;//the y position of the target place
            double distance_to_goal = 3;//the distance to the goal

    };
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "frontier_detection");
    ros::NodeHandle private_nh1("~");
    frontier_detection::frontierdetection ta(private_nh1);
    ros::MultiThreadedSpinner spinner;
    spinner.spin();
    return 0;
}
