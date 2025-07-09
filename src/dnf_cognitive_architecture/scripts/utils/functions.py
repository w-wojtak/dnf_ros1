import cv2
import numpy as np
import os
from dotenv import load_dotenv
import pandas as pd
from ultralytics import YOLO
import os
import torch
import ast
from threading import Lock
# import pika
import json


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


import copy
import itertools
import csv

import time

# import tensorflow as tf



lock = Lock()

load_dotenv()

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)


string_connexion = int(os.getenv("STRING_CONNECTION"))
camera_type = os.getenv("CAMERA_TYPE")
lens = os.getenv("LENS")

world_points = None
'''
world_points_X1 = os.getenv("world_points_X1")
world_points_Y1 = os.getenv("world_points_Y1")
world_points_X2 = os.getenv("world_points_X2")
world_points_Y2 = os.getenv("world_points_Y2")
world_points_X3 = os.getenv("world_points_X3")
world_points_Y3 = os.getenv("world_points_Y3")
world_points_X4 = os.getenv("world_points_X4")
world_points_Y4 = os.getenv("world_points_Y4")    
'''
names_class         =  ast.literal_eval(os.getenv("NAME_CLASS"))
model_path          = os.getenv("MODEL_PATH_FILE") + "/" + os.getenv("MODEL_NAME_FILE")
conf                = float(os.getenv("CONF") )
rectangle_thickness = int(os.getenv("RECTANGLE_THICKNESS") )
text_thickness      = int(os.getenv("TEXT_THICKNESS") )
# colors              = ast.literal_eval(os.getenv("COLORS"))
def generate_distinct_colors(n):
    hsv_colors = []
    step = 360 / n
    for i in range(n):
        hue = int(i * step)
        hsv = np.array([[[hue / 2, 180, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        hsv_colors.append(bgr.tolist())
 
    return hsv_colors
 
def generate_colors_for_labels(labels):
    unique_labels = list(set(labels))
    base_colors = generate_distinct_colors(len(unique_labels))
    label2color = {label: color for label, color in zip(unique_labels, base_colors)}
    return [label2color[label] for label in labels]
 
 
 
colors              = generate_colors_for_labels(names_class)

action = os.getenv("ACTION")
filename = os.getenv("FILENAME")


csv_path = '../gesture_dataset/' +  os.getenv("CSV_PATH")
np_path = '../' +  os.getenv("NP_PATH")


homography_matrix = []
image_bbox        = None
corners_bottoms   = []   
do_hom            = False
selected_points   = []


connection    = channel = None
exchange      = os.getenv("RMQ_EXCHANGE")
exchange_type = os.getenv("RMQ_EXCHANGETYPE")


def object_detection_image(model, img, save=False, save_txt=False, verbose=False):
    with lock:
        results = model.predict(source=img, save=save, save_txt=save_txt, verbose=verbose)
    result = results[0]
    bboxes = result.boxes.xyxy.cpu().numpy().astype("int")
    class_ids = result.boxes.cls.cpu().numpy().astype("int")
    scores = result.boxes.conf.cpu().numpy().astype("float").round(2)
    del results, result
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
    return bboxes, class_ids, scores

def object_detection_real_word(model, img):
    global homography_matrix
    global image_bbox
    global corners_bottoms
    worlds_coordinates = []
    classes = []
    score = []
    image_bboxs = []
    if do_hom:
        corners_bottoms = np.array(corners_bottoms)
        x_min, y_min = np.min(corners_bottoms, axis=0)
        x_max, y_max = np.max(corners_bottoms, axis=0)
        mask = np.ones_like(img, dtype=np.uint8) * 255
        mask[int(y_min):int(y_max), int(x_min):int(x_max)] = img[int(y_min):int(y_max), int(x_min):int(x_max)].copy()
        bboxes, class_ids, scores = object_detection_image(model, mask)
        boxes = [bboxes[i] for i in range(len(class_ids)) if scores[i] >= conf]
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[class_ids[index]], rectangle_thickness)
            cv2.putText(img, f"{names_class[class_ids[index]]}:{scores[index]}", (x2, y2), cv2.FONT_HERSHEY_PLAIN, 2, colors[class_ids[index]], text_thickness)
            image_bbox = [x1, y1, x2 - x1, y2 - y1]
            bbox_center_image = [image_bbox[0] + image_bbox[2] / 2, image_bbox[1] + image_bbox[3] / 2]
            cv2.circle(img, [round(value) for value in bbox_center_image], 10, (0,0,255), -1)
            worlds_coordinates.append(transform_to_world_coordinates(homography_matrix, bbox_center_image))
            classes.append(names_class[class_ids[index]])
            score.append(scores[index])  
            image_bboxs.append(image_bbox)
    return worlds_coordinates, image_bbox, image_bboxs, classes, score



def compute_homography(image_points, world_points):
    homography_matrix, _ = cv2.findHomography(image_points, world_points)
    return homography_matrix

def transform_to_world_coordinates(homography_matrix, image_point):
    image_point_homogeneous = np.append(image_point, 1)
    world_point_homogeneous = np.dot(homography_matrix, image_point_homogeneous)
    world_point = world_point_homogeneous / world_point_homogeneous[2]
    return world_point[:2]

def paint_markers(frame, corners, ids):
    centers_x = []
    centers_y = []
    markers_id = []
    corners_bottoms = []
    if len(corners) > 0:
        ids = ids.flatten()
        for marker_corner, marker_id in zip(corners, ids):
            corners = marker_corner.reshape((4, 2))
            top_left, top_right, bottom_right, bottom_left = corners
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            if marker_id == 1:
                corners_bottoms.append(bottom_right)
            elif marker_id == 2:
                corners_bottoms.append(bottom_left)
            elif marker_id == 3:
                corners_bottoms.append(bottom_left)
            else:
                corners_bottoms.append(top_left)
            cv2.line(frame, top_left, top_right, (0, 255, 255), 1)
            cv2.line(frame, top_right, bottom_right, (0, 255, 255), 1)
            cv2.line(frame, bottom_right, bottom_left, (0, 255, 255), 1)
            cv2.line(frame, bottom_left, top_left, (0, 255, 255), 1)
            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            centers_x.append(center_x)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
            centers_y.append(center_y)
            markers_id.append(marker_id)
            cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)
            cv2.putText(frame, str(marker_id),
                        (top_left[0], top_left[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
    return frame, centers_x, centers_y, markers_id, corners_bottoms

def select_pixel(event, x, y, flags, param):
    global homography_matrix
    global image_bbox
    world_coordinates = ""
    if event == cv2.EVENT_LBUTTONDOWN:
        if do_hom:
            image_bbox = [x, y, 1, 1]
            bbox_center_image = [image_bbox[0] + image_bbox[2] / 2, image_bbox[1] + image_bbox[3] / 2]
            world_coordinates = transform_to_world_coordinates(homography_matrix, bbox_center_image)
            print("World Coordinates Manual Point:", world_coordinates)
        else:
            print("Not detect all markers")
    return world_coordinates, image_bbox

def order_points_z(points):
    points_sorted = sorted(points, key=lambda p: p[1])
    top_points = points_sorted[:2]
    bottom_points = points_sorted[2:]
    top_points = sorted(top_points, key=lambda p: p[0])
    bottom_points = sorted(bottom_points, key=lambda p: p[0])
    z_ordered_points = top_points + bottom_points
    return z_ordered_points

def load_points_from_file():
    global selected_points
    try:
        with open(filename, 'r+') as file:
            for line in file:
                x, y = map(int, line.strip().split(','))
                selected_points.append((x, y))
                selected_points = order_points_z(selected_points)      
        print(f"Points loaded: {selected_points}")
    except Exception as e:
        print(f"Error loading points: {e}")

def save_point_to_file():
    global selected_points
    try:
        with open(filename, 'w') as file:
            for point in selected_points:
                file.write(f"{point[0]},{point[1]}\n")
    except Exception as e:
        print(f"Error saving point:: {e}")

def select_pixel_coordinates(event, x, y, flags, param):
    global selected_points
    if event == cv2.EVENT_RBUTTONDOWN:
        for point in selected_points:
            if abs(point[0] - x) < 5 and abs(point[1] - y) < 5:
                selected_points.remove(point)
                selected_points = order_points_z(selected_points)
                print(f"Point deleted: {point}")
                save_point_to_file()
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        selected_points = order_points_z(selected_points)
        print(f"Selected coordinates: X={x}, Y={y}")
        save_point_to_file()

def read_webcam_frames():
    global selected_points
    cap = cv2.VideoCapture(string_connexion)
    if not cap.isOpened():
        print("Error")
        return
    cv2.namedWindow("Webcam Frame", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Webcam Frame', select_pixel_coordinates)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error")
            break
        for index, selected_point in enumerate(selected_points):
            cv2.circle(frame, selected_point, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{int(index + 1)}", selected_point, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        cv2.imshow('Webcam Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



def draw_landmarks(image, result, gesture_res):
  hand_landmarks_list = result.hand_landmarks
  handedness_list = result.handedness
  annotated_image = np.copy(image)
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN
    cv2.putText(annotated_image, f"{gesture_res}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def process_hand_landmarks(results, image, label):
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            landmark_coordinates = extract_landmark_coordinates(image, hand_landmarks)
            normalized_landmarks = normalize_landmark_coordinates(landmark_coordinates)
            save_landmark_data_to_csv(label, normalized_landmarks)

def extract_landmark_coordinates(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_points = []
    for landmark in landmarks:  
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_points.append([landmark_x, landmark_y])

    return landmark_points

def normalize_landmark_coordinates(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]
    for index, (x, y) in enumerate(temp_landmark_list):
        temp_landmark_list[index] = [x - base_x, y - base_y]
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list), default=1) 
    return [n / max_value for n in temp_landmark_list]

def normalize_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = temp_point_history[0] 
    for index, (x, y) in enumerate(temp_point_history):
        temp_point_history[index] = [(x - base_x) / image_width, (y - base_y) / image_height]
    return list(itertools.chain.from_iterable(temp_point_history))

def save_landmark_data_to_csv(label, landmark_list, csv_path=csv_path):
    try:
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([label, *landmark_list])
    except Exception as e:
        print(f"Error save CSV: {e}")


def gesture_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], np.array([input_data]))
    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]['index'])
    #print(tflite_results)
    #print("Predict:", np.argmax(np.squeeze(tflite_results)))
    return np.argmax(np.squeeze(tflite_results))


def process_hand_landmarks_lstm(frame, results, data, class_number, this_action, mp_drawing, mp_hands):
    if not results.multi_hand_landmarks:
        return data, frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
            angle = np.degrees(angle) 
            angle_label = np.array([angle], dtype=np.float32)
            angle_label = np.append(angle_label, int(class_number))
            d = np.concatenate([joint.flatten(), angle_label])
            data.append(d)            
            cv2.putText(frame, f'{this_action.upper()}', org=(int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return data, frame

