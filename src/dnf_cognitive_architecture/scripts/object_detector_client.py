import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import time
from datetime import datetime


import pyzed.sl as sl
import utils.functions as functions

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

def read_webcam_frames_hom(stop_event):
    detection_enabled = False  # Control detection via keypress
    last_print_time = time.time()  # Initialize timer
    
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
                cv2.rectangle(image_ocv, (x1, y1), (x2, y2), 
                            functions.colors[filtered_class_ids[index]], 
                            functions.rectangle_thickness)

                image_bbox = [x1, y1, x2 - x1, y2 - y1]
                bbox_center_image = [image_bbox[0] + image_bbox[2] / 2, 
                                   image_bbox[1] + image_bbox[3] / 2]
                cv2.circle(image_ocv, (round(bbox_center_image[0]), 
                          round(bbox_center_image[1])), 10, (0, 0, 255), -1)

                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)
                err, point3D = point_cloud.get_value(round(bbox_center_image[0]), 
                                                   round(bbox_center_image[1]))

                if err == sl.ERROR_CODE.SUCCESS:
                    try:
                        x = round(point3D[0], 2)
                        y = round(point3D[1], 2)
                        z = round(point3D[2], 2)
                        cv2.putText(image_ocv,
                                  f"{functions.names_class[filtered_class_ids[index]]}:{filtered_scores[index]:.2f} 'x:'{x} 'y:'{y} 'z:'{z} ",
                                  (x2, y2), cv2.FONT_HERSHEY_PLAIN, 2, 
                                  functions.colors[filtered_class_ids[index]], 
                                  functions.text_thickness)
                    except Exception:
                        pass
                else:
                    x = y = z = None

                worlds_coordinates.append([x, y, z])
                classes.append(functions.names_class[filtered_class_ids[index]])
                scores.append(filtered_scores[index])
                image_bboxs.append(image_bbox)

            if detection_enabled and len(worlds_coordinates) > 0:
                # Print only if 1 second has elapsed since last print
                if current_time - last_print_time >= 1.0:
                    print("\n" + "="*50)
                    print(f"Detection Update [{datetime.now().strftime('%H:%M:%S')}]")
                    print("-"*50)
                    for v in range(len(worlds_coordinates)):
                        x, y, z = worlds_coordinates[v]
                        if x is not None and y is not None and z is not None:
                            print(f"Object: {classes[v]:<15} | Position (x,y,z): ({x:>6.2f}, {y:>6.2f}, {z:>6.2f}) m | Confidence: {scores[v]:.2f}")
                    print("="*50)
                    last_print_time = current_time

            cv2.namedWindow(f"{functions.camera_type} Frame", cv2.WINDOW_NORMAL)
            cv2.imshow(f"{functions.camera_type} Frame", image_ocv)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                detection_enabled = True
                print("Detection started.")
            elif key == ord('p'):
                detection_enabled = False
                print("Detection paused.")
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

    read_webcam_frames_hom(stop_event)
