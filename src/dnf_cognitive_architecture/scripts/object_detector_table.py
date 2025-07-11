#!/usr/bin/env python3

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import time
from datetime import datetime
import json

# ROS imports
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import MarkerArray, Marker

import pyzed.sl as sl
import utils.functions as functions

# Initialize ROS publisher (global so we can use it in the function)
rospy.init_node('zed_object_detector', anonymous=True, disable_signals=True)
detection_pub = rospy.Publisher('/object_detections', String, queue_size=10)
marker_pub = rospy.Publisher('/object_markers', MarkerArray, queue_size=10)

rospy.loginfo("Object detector initialized, publishing to /object_detections")

init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER
init_params.camera_fps = 30
init_params.sdk_verbose = 1
runtime = sl.RuntimeParameters()

point_cloud = sl.Mat()
zed = sl.Camera()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    print("Using model path: ", functions.model_path)
    model = YOLO(functions.model_path)
    model.to(device)
    print("Model loaded on device: ", next(model.parameters()).device)
except Exception as e:
    print(f"Load model error: {e}")
    exit()

# Add this at the start of your script, outside the function
EXCLUDED_CLASSES = ['nut1', 'nut2']  # modify this list as needed

# Define table boundaries
TABLE_BOUNDS = {
    'x_min': -0.82,
    'x_max': 0.71,
    'y_min': 0.06,
    'y_max': 0.52
}

def is_point_on_table(x, y):
    """Check if a point (x, y) is within the table boundaries"""
    if x is None or y is None:
        return False
    
    return (TABLE_BOUNDS['x_min'] <= x <= TABLE_BOUNDS['x_max'] and 
            TABLE_BOUNDS['y_min'] <= y <= TABLE_BOUNDS['y_max'])

def publish_detections(detections_data):
    """Publish detection data to ROS topic"""
    try:
        # Publish as JSON string
        msg = String()
        msg.data = json.dumps(detections_data)
        detection_pub.publish(msg)
        
        # Also publish visualization markers
        marker_array = MarkerArray()
        for idx, detection in enumerate(detections_data['detections']):
            if detection['position']['x'] is not None:
                marker = Marker()
                marker.header.frame_id = "camera_link"  # Adjust frame_id as needed
                marker.header.stamp = rospy.Time.now()
                marker.ns = "object_detections"
                marker.id = idx
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                marker.pose.position.x = detection['position']['x']
                marker.pose.position.y = detection['position']['y']
                marker.pose.position.z = detection['position']['z']
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                
                marker.lifetime = rospy.Duration(1.0)  # Marker expires after 1 second
                
                marker_array.markers.append(marker)
        
        marker_pub.publish(marker_array)
        
    except Exception as e:
        rospy.logerr(f"Failed to publish detection: {e}")

def read_webcam_frames_hom(stop_event):
    detection_enabled = False  # Control detection via keypress
    last_print_time = time.time()  # Initialize timer
    last_publish_time = time.time()  # Add publish timer
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Camera error.")
        exit(1)
    image_zed = sl.Mat()

    while True:
        worlds_coordinates = []
        classes = []
        scores = []
        image_bboxs = []
        current_time = time.time()

        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            image_ocv = image_zed.get_data()
            image_ocv = cv2.cvtColor(image_ocv, cv2.COLOR_RGBA2RGB)

            bboxes, class_ids, confidence_scores = functions.object_detection_image(model, image_ocv)
            
            # Filter out excluded classes and low confidence detections
            valid_indices = [i for i in range(len(class_ids)) 
                           if (functions.names_class[class_ids[i]] not in EXCLUDED_CLASSES 
                               and confidence_scores[i] >= functions.conf)]
            
            # Create filtered lists
            boxes = [bboxes[i] for i in valid_indices]
            filtered_class_ids = [class_ids[i] for i in valid_indices]
            filtered_scores = [confidence_scores[i] for i in valid_indices]

            for index, box in enumerate(boxes):
                x1, y1, x2, y2 = box.tolist()
                
                image_bbox = [x1, y1, x2 - x1, y2 - y1]
                bbox_center_image = [image_bbox[0] + image_bbox[2] / 2, 
                                   image_bbox[1] + image_bbox[3] / 2]

                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)
                err, point3D = point_cloud.get_value(round(bbox_center_image[0]), 
                                                   round(bbox_center_image[1]))

                if err == sl.ERROR_CODE.SUCCESS:
                    try:
                        x = round(point3D[0], 2)
                        y = round(point3D[1], 2)
                        z = round(point3D[2], 2)
                        
                        # Only process and draw objects that are on the table
                        if is_point_on_table(x, y):
                            # Draw rectangle for objects on table
                            cv2.rectangle(image_ocv, (x1, y1), (x2, y2), 
                                        functions.colors[filtered_class_ids[index]], 
                                        functions.rectangle_thickness)
                            
                            cv2.circle(image_ocv, (round(bbox_center_image[0]), 
                                      round(bbox_center_image[1])), 10, (0, 0, 255), -1)
                            
                            cv2.putText(image_ocv,
                                      f"{functions.names_class[filtered_class_ids[index]]}:{filtered_scores[index]:.2f} 'x:'{x} 'y:'{y} 'z:'{z} ",
                                      (x2, y2), cv2.FONT_HERSHEY_PLAIN, 2, 
                                      functions.colors[filtered_class_ids[index]], 
                                      functions.text_thickness)
                            
                            # Store only table objects
                            worlds_coordinates.append([x, y, z])
                            classes.append(functions.names_class[filtered_class_ids[index]])
                            scores.append(filtered_scores[index])
                            image_bboxs.append(image_bbox)
                    except Exception:
                        pass

            if detection_enabled and len(worlds_coordinates) > 0:
                # Print only if 1 second has elapsed since last print
                if current_time - last_print_time >= 1.0:
                    print("\n" + "="*50)
                    print(f"Detection Update [{datetime.now().strftime('%H:%M:%S')}]")
                    print(f"Table bounds: X[{TABLE_BOUNDS['x_min']}, {TABLE_BOUNDS['x_max']}], Y[{TABLE_BOUNDS['y_min']}, {TABLE_BOUNDS['y_max']}]")
                    print("-"*50)
                    for v in range(len(worlds_coordinates)):
                        x, y, z = worlds_coordinates[v]
                        if x is not None and y is not None and z is not None:
                            print(f"Object: {classes[v]:<15} | Position (x,y,z): ({x:>6.2f}, {y:>6.2f}, {z:>6.2f}) m | Confidence: {scores[v]:.2f}")
                    print("="*50)
                    last_print_time = current_time
                
                # Publish to ROS at a configurable rate (e.g., 10 Hz)
                if current_time - last_publish_time >= 0.5:  # 0.5 = 2 Hz
                    # Prepare detection data
                    detections_data = {
                        "timestamp": rospy.get_time(),
                        "frame_time": datetime.now().isoformat(),
                        "table_bounds": TABLE_BOUNDS,
                        "detections": []
                    }
                    
                    for v in range(len(worlds_coordinates)):
                        x, y, z = worlds_coordinates[v]
                        detection = {
                            "object": classes[v],
                            "position": {
                                "x": x,
                                "y": y,
                                "z": z
                            },
                            "confidence": float(scores[v]),
                            "bbox": image_bboxs[v]
                        }
                        detections_data["detections"].append(detection)
                    
                    # Publish the detections
                    publish_detections(detections_data)
                    last_publish_time = current_time

            cv2.namedWindow(f"{functions.camera_type} Frame", cv2.WINDOW_NORMAL)
            cv2.imshow(f"{functions.camera_type} Frame", image_ocv)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                detection_enabled = True
                print("Detection started.")
                rospy.loginfo("Detection started - publishing to ROS topics")
            elif key == ord('p'):
                detection_enabled = False
                print("Detection paused.")
                rospy.loginfo("Detection paused - stopped publishing")
            elif key == ord('q'):
                stop_event.set()
                break

    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import multiprocessing

    manager = multiprocessing.Manager()
    stop_event = multiprocessing.Event()

    try:
        read_webcam_frames_hom(stop_event)
    except KeyboardInterrupt:
        print("\nShutting down object detector...")
    finally:
        # Clean shutdown
        if 'zed' in locals():
            zed.close()
        cv2.destroyAllWindows()
