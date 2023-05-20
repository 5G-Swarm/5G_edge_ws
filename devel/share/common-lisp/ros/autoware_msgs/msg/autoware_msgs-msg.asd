
(cl:in-package :asdf)

(defsystem "autoware_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :std_msgs-msg
               :visualization_msgs-msg
)
  :components ((:file "_package")
    (:file "DroneSyn" :depends-on ("_package_DroneSyn"))
    (:file "_package_DroneSyn" :depends-on ("_package"))
    (:file "Pose2DArray" :depends-on ("_package_Pose2DArray"))
    (:file "_package_Pose2DArray" :depends-on ("_package"))
    (:file "TrackingObjectMarker" :depends-on ("_package_TrackingObjectMarker"))
    (:file "_package_TrackingObjectMarker" :depends-on ("_package"))
    (:file "TrackingObjectMarkerArray" :depends-on ("_package_TrackingObjectMarkerArray"))
    (:file "_package_TrackingObjectMarkerArray" :depends-on ("_package"))
  ))