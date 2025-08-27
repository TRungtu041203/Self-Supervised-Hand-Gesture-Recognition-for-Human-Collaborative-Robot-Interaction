import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time
from statistics import mode
import csv
import os
from net.aimclr_v2_3views import AimCLR_v2_3views
import torch
from cleaning import clean_micro_gaps, fit_to_length

#B1: Set up mediapipe
# pose_model = mp.solutions.pose
model_path = r"models/pose_landmarker_full.task" 
hand_model_path = r"models/hand_landmarker.task"

# detect_func = pose_model.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
# detect_func2 = pose_model.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=2)
VisionRunningMode = mp.tasks.vision.RunningMode
base_options = python.BaseOptions(model_path)
hand_base_options = python.BaseOptions(hand_model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    running_mode=VisionRunningMode.VIDEO)
hand_option = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    running_mode=VisionRunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=0.2)
detector = vision.PoseLandmarker.create_from_options(options)
hand_detector = vision.HandLandmarker.create_from_options(hand_option)

count_down=0
bottom_line_y = 0
trigger_line = 0
state_dict = {
    0: 'Idle',
    1: 'Moving',
    2: 'Trigger'
}
state_color = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255)
}
previous_moving_hand_index = 0
moving_hand_index = 0
number_diff_move = 0
current_state = 0 

#B2: Extract pose and hand landmarks
# def detect_pose(imagee, detect_func):
#     #imagee = cv2.imread(image_path)
#     image_copy = imagee.copy()
#     result = detect_func.process(image_copy)

#     drawing = mp.solutions.drawing_utils
#     drawing.draw_landmarks(image=image_copy, landmark_list=result.pose_landmarks, connections=pose_model.POSE_CONNECTIONS)

#     return image_copy
def extract_hand(result):
    rh = lh = np.zeros((21, 3))
    
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        assert len(handedness) == 1, "Unexpected Error in handedness"
        
        if handedness[0].category_name == 'Left':
            lh = np.array([[lm.x, lm.y, 0] for lm in hand_landmarks])
        if handedness[0].category_name == 'Right':
            rh = np.array([[lm.x, lm.y, 0] for lm in hand_landmarks])
        
    hand_pose = np.concatenate((rh, lh), axis=0)
    return hand_pose


def extract_arm(result):
    extracted_keypoints = [
        solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
        solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
        solutions.pose.PoseLandmark.LEFT_ELBOW.value,
        solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
        solutions.pose.PoseLandmark.LEFT_WRIST.value,
        solutions.pose.PoseLandmark.RIGHT_WRIST.value
    ]
    n_keypoints = len(extracted_keypoints)
    arm_pose = np.zeros((n_keypoints, 3))
    
    for landmarks in result.pose_landmarks:
        extracted_lms = [landmarks[i] for i in extracted_keypoints]
        for i in range(n_keypoints):
            lm = extracted_lms[i]
            arm_pose[i] = (lm.x, lm.y, 0)
            
    return arm_pose   
 
def fill_missing_pose(pose_concat, previous_2_hands):
    if previous_2_hands[0][0][0] != 0 and pose_concat[0][0] == 0:
        pose_concat[0:21] = previous_2_hands[0]
    if previous_2_hands[1][0][0] != 0 and pose_concat[21][0] == 0:
        pose_concat[21:42] = previous_2_hands[1]
    return pose_concat


def draw_concat_pose(image, concat_pose, handedness=None, thickness=2):
    annotated_image = np.copy(image)

    right_hand = concat_pose[0:21]
    left_hand = concat_pose[21:42]
    arms = concat_pose[42:]

    # [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
    arm_connections = [
        (0, 2), (2, 4),  # Left arm
        (1, 3), (3, 5),  # Right arm
        (0, 1)           # Shoulders
    ]

    height, width, _ = annotated_image.shape

    def to_landmark_list(points):
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        landmark_list.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=p[0], y=p[1], z=p[2]) for p in points
        ])
        return landmark_list

    # Draw right hand
    if np.any(right_hand[0] != 0):
        hand_landmarks_proto = to_landmark_list(right_hand)
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

    # Draw left hand
    if np.any(left_hand[0] != 0):
        hand_landmarks_proto = to_landmark_list(left_hand)
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

    # Draw arms
    if np.any(arms[0] != 0):
        pose_landmarks_proto = to_landmark_list(arms)
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            arm_connections,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )

    return annotated_image

#B3: Detect state
def set_bottom_line_y(pose_list):
    global bottom_line_y
    global trigger_line
    margin = 0.05
    bottom_line_y = max(max(pose_list[42:48, 1]), bottom_line_y) 
    bottom_line_y -= margin*bottom_line_y  
    trigger_line = bottom_line_y + 1.25*margin*bottom_line_y 


def compute_var_when_move(pose_list):
    # compute variance of the y value of the keypoints when the pose is moving
    # if the variance is small, then the pose is stable
    # if the variance is large, then the pose is moving
    pose_np = np.array(pose_list) # shape:(frame, 48, 3)
    hand_1_var = np.var(pose_np[:, :21, 1])
    hand_2_var = np.var(pose_np[:, 21:42, 1])
    # print('var:', hand_1_var, hand_2_var)
    
    return hand_1_var, hand_2_var

import numpy as np

def set_state(pose_list):
    pose_list = list(pose_list) 
    if len(pose_list) >0:
        pose_list = np.array(pose_list)
    else:
        return 0
    if pose_list.shape[0] < 33:
        return 0
    if np.mean(pose_list[:, :, :]) == 0:
        return 0
    
    global moving_hand_index
    global previous_moving_hand_index
    global current_state

    hand_var_1, hand_var_2 = compute_var_when_move(pose_list)


    frame_5 = int((infer_len-6))
    key_43 = np.array(pose_list[frame_5:, 43, 0])
    key_45 = np.array(pose_list[frame_5:, 45, 0])
    key_47 = np.array(pose_list[frame_5:, 47, 0])

    key_42 = np.array(pose_list[frame_5:, 42, 0])
    key_44 = np.array(pose_list[frame_5:, 44, 0])
    key_46 = np.array(pose_list[frame_5:, 46, 0])

    key_43_y = np.array(pose_list[frame_5:, 43, 1])
    key_45_y = np.array(pose_list[frame_5:, 45, 1])
    key_47_y = np.array(pose_list[frame_5:, 47, 1])

    key_42_y = np.array(pose_list[frame_5:, 42, 1])
    key_44_y = np.array(pose_list[frame_5:, 44, 1])
    key_46_y = np.array(pose_list[frame_5:, 46, 1])

    threshold = 0.03

    if not(abs(np.mean(key_43)- np.mean(key_45)) <threshold) or not(abs(np.mean(key_43)- np.mean(key_47)) <threshold) and hand_var_1>hand_var_2:
        if not(np.mean(key_43_y) < np.mean(key_45_y) < np.mean(key_47_y)):
            moving_hand_index=0 #right hand
    if not(abs(np.mean(key_42)- np.mean(key_44)) <threshold) or not(abs(np.mean(key_42)- np.mean(key_46)) <threshold) and hand_var_2>hand_var_1:
        if not(np.mean(key_42_y) < np.mean(key_44_y) < np.mean(key_46_y)):
            moving_hand_index=1 #left hand

    
    #Control moving hand index not to jump around
    if moving_hand_index != previous_moving_hand_index:
        moving_hand_index = previous_moving_hand_index
        global number_diff_move
        number_diff_move += 1
    if number_diff_move > 5:
        moving_hand_index = 1 - moving_hand_index
        number_diff_move = 0
        previous_moving_hand_index = moving_hand_index

    
    n = len(pose_list)
    n_5 = n // 11
    first_5 = np.array(pose_list[:n_5])
    last_5 = np.array(pose_list[-n_5:])

    if moving_hand_index == 0:
        max_first_5 = np.max(first_5[:,:21, 1])
        min_last_5 = np.min(last_5[:,:21, 1])
        max_last_5 = np.max(last_5[:,:21, 1])

        elbow_x = np.mean(last_5[:, 45, 0]) 
        wrist_x = np.mean(last_5[:, 47, 0])
        wrist_y_last = np.max(last_5[:, 47, 1])
        wrist_y_first = np.max(first_5[:, 47, 1])

        wrist_var = np.var(last_5[:,47,1])

    else:
        max_first_5 = np.max(first_5[:,21:42 , 1])
        min_last_5 = np.min(last_5[:,21:42, 1])
        max_last_5 = np.max(last_5[:,21:42, 1])

        elbow_x = np.mean(last_5[:, 44, 0])  
        wrist_x = np.mean(last_5[:, 46, 0])
        wrist_y_last = np.max(last_5[:, 46, 1])
        wrist_y_first = np.max(first_5[:, 46, 1])
        wrist_var = np.var(last_5[:,46,1])

    if wrist_y_first>bottom_line_y and wrist_y_last < bottom_line_y:
        if current_state == 0:
            current_state = 1

    # if max_first_5 < bottom_line_y and min_last_5 > bottom_line_y and max_last_5 > trigger_line and abs(wrist_x-elbow_x)<0.03:
    #     if current_state == 1:
    #         current_state = 2
    if wrist_y_first<bottom_line_y and wrist_y_last > bottom_line_y and max_last_5 > trigger_line and abs(wrist_x-elbow_x)<0.03 :
        if current_state == 1:
            current_state = 2

    global count_down
    if wrist_y_last>bottom_line_y:
        count_down+=1
        if count_down>30 and wrist_var <0.01:
            if current_state ==1:
                current_state=2
                count_down=0
    else:
        count_down = 0

#B4: Load model and predict
# model = loaded_model 
# def predict(pose_list):
#     pose_list = np.array(pose_list)
#     sample = {
#         'pose': [pose_list]
#     }
#     X_0, X_1 = data_generator(sample, C)
#     Y = model.predict([X_0, X_1])
#     print(Y)
#     return np.argmax(Y)

model = AimCLR_v2_3views(base_encoder= 'net.st_gcn.Model', pretrain=False,
                 in_channels=3, hidden_channels=32,
                 hidden_dim=256, num_class=19, dropout=0.5,
                 graph_args={'layout': 'cobot', 'strategy': 'distance'},
                 edge_importance_weighting=True)

model.load_state_dict(torch.load(r"C:\AimCLR-v2\work_dir\cobot_3views_2D_xsub_medgap_aug1\finetune\best_model.pt", map_location='cuda'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
#pose_list: N,48,3
def predict(pose_list):
    pose_list = np.array(pose_list)
    pose_list,_ = clean_micro_gaps(pose_list,6) #(N,48,3)
    pose_list = np.transpose(pose_list, (2, 0, 1)).astype(np.float32, copy=False) #(3,N,48)
    pose_list = pose_list[..., np.newaxis] #(3,N,48,1)
    pose_list,_,_ = fit_to_length(pose_list, 64,'uniform-sample') #(3,64,48,1)
    pose_list = torch.tensor(pose_list, dtype=torch.float32).unsqueeze(0) #(1,3,64,48,1)
    pose_list = pose_list.to(device)
    with torch.no_grad():
        output = model(None, pose_list)
        pred = output.argmax(dim=1)
        print("Predicted class:", pred.item())
    return pred.item()

#B5: Real time inference
video_path = r"C:\AimCLR-v2\s2_Alex_1.mp4"
cap = cv2.VideoCapture(video_path) 
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Resolution: {int(width)}x{int(height)}")

# fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"Estimated FPS: {fps:.2f}")

previous_2_hands = np.zeros((2, 21, 3))
from collections import deque
infer_len = 45
pose_list = deque(maxlen=infer_len)

start_time = time.time()
register_time = 3
pose_time =0
frame_count = 0 

predict_list= deque(maxlen=15)
previous_label = None
potential_list = deque(maxlen=20)

path = r'C:\AimCLR-v2\webcam_s08_HaAnhh'

while True:

    ret, frame = cap.read()

    if not ret:
        break
    frame_count += 1
    timecount = time.time()
    frame = cv2.resize(frame, (640, 480))
    height, width, _ = frame.shape

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    #frame = detect_pose(frame, detect_func2)
    #detection_result = detector.detect_for_video(mp_image,frame_timestamp_ms)
    pose_detection_result = detector.detect_for_video(mp_image,frame_timestamp_ms)
    hand_detection_result = hand_detector.detect_for_video(mp_image,frame_timestamp_ms)

    pose = extract_arm(pose_detection_result)
    hand_pose = extract_hand(hand_detection_result)
    hand_pose = fill_missing_pose(hand_pose, previous_2_hands)

    if hand_pose[0][0] != 0:
        previous_2_hands[0] = hand_pose[0]
    if hand_pose[21][0] != 0:
        previous_2_hands[1] = hand_pose[1]
    # concatenate pose and hand_pose
    concat_pose = np.concatenate((hand_pose, pose), axis=0)

    pose_list.append(concat_pose) 

    file_name = "predict_s02_Alex1.csv"
    set_state(pose_list)

    with open(file_name, "a",newline="") as file:
        writer = csv.writer(file)
        if file.tell() ==0:
            writer.writerow(["ID","start","stop"])

        if current_state==1:
            pose_time+=1
            if pose_time==1: 
                start_frame = frame_count
                starttime = time.time()

            input_pose = np.array(list(pose_list))
            predicted_label = predict(input_pose)

                #log_file.write(f"Predicted Label: {predicted_label}\n\n")
            predict_list.append(predicted_label)

            if predicted_label !=0 and predicted_label == previous_label and pose_time <=75:  
                potential_list.append(predicted_label)

            previous_label = predicted_label

        elif current_state==2:
            stop_frame = frame_count
            stoptime = time.time()
            if len(potential_list) > 0:  
                final_label = mode(potential_list)
            else:
                final_label = mode(predict_list)

            writer.writerow([final_label, start_frame, stop_frame])

            predict_list.clear()
            potential_list.clear()
            current_state=0
            pose_time=0
    # set_state(pose_list)
    current_time = time.time()
    if current_time - start_time < register_time:
        set_bottom_line_y(concat_pose)
    # draw bottom line and trigger line by a rectangle
    _state_color = state_color[current_state]

    frame = draw_concat_pose(frame,concat_pose)
    cv2.rectangle(frame, (0, int(bottom_line_y * height)), (width, int(trigger_line * height)), _state_color, 2)
    cv2.putText(frame, state_dict[current_state], (10, int(bottom_line_y * height) + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, _state_color, 2)



    cv2.imshow('Result',cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # timecount= f"{timecount:.3f}".replace('.', '_')

    # frame_path = os.path.join(path,f'frame_{frame_count}_time_{timecount}.jpg')

    # cv2.imwrite(frame_path,cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(1)&0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


