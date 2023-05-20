;; Auto-generated. Do not edit!


(when (boundp 'autoware_msgs::DroneSyn)
  (if (not (find-package "AUTOWARE_MSGS"))
    (make-package "AUTOWARE_MSGS"))
  (shadow 'DroneSyn (find-package "AUTOWARE_MSGS")))
(unless (find-package "AUTOWARE_MSGS::DRONESYN")
  (make-package "AUTOWARE_MSGS::DRONESYN"))

(in-package "ROS")
;;//! \htmlinclude DroneSyn.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass autoware_msgs::DroneSyn
  :super ros::object
  :slots (_header _gps _imu ))

(defmethod autoware_msgs::DroneSyn
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:gps __gps) (make-array 6 :initial-element 0.0 :element-type :float))
    ((:imu __imu) (make-array 7 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _header __header)
   (setq _gps __gps)
   (setq _imu __imu)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:gps
   (&optional __gps)
   (if __gps (setq _gps __gps)) _gps)
  (:imu
   (&optional __imu)
   (if __imu (setq _imu __imu)) _imu)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float64[6] _gps
    (* 8    6)
    ;; float64[7] _imu
    (* 8    7)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float64[6] _gps
     (dotimes (i 6)
       (sys::poke (elt _gps i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;; float64[7] _imu
     (dotimes (i 7)
       (sys::poke (elt _imu i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float64[6] _gps
   (dotimes (i (length _gps))
     (setf (elt _gps i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;; float64[7] _imu
   (dotimes (i (length _imu))
     (setf (elt _imu i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     )
   ;;
   self)
  )

(setf (get autoware_msgs::DroneSyn :md5sum-) "cbadf538072dc3bef18cfe582f63dae4")
(setf (get autoware_msgs::DroneSyn :datatype-) "autoware_msgs/DroneSyn")
(setf (get autoware_msgs::DroneSyn :definition-)
      "std_msgs/Header header
float64[6] gps   
float64[7] imu

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

")



(provide :autoware_msgs/DroneSyn "cbadf538072dc3bef18cfe582f63dae4")


