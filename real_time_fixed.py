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
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# COBOT action names mapping - corrected order
ACTION_NAMES = [
    'Start',         # 0
    'Stop',         # 1 (duplicate)
    'Slower',       # 2
    'Faster',       # 3
    'Done',         # 4
    'FollowMe',     # 5
    'Lift',         # 6
    'Home',         # 7
    'Interaction',  # 8
    'Look',         # 9
    'PickPart',     # 10
    'DepositPart',  # 11
    'Report',       # 12
    'Ok',           # 13
    'Again',        # 14
    'Help',         # 15
    'Joystick',     # 16
    'Identification', # 17
    'Change'        # 18
]

class COBOTActionRecognition:
    def __init__(self, model_path, pose_model_path, hand_model_path, sequence_length=64):
        """
        Initialize COBOT action recognition system with original segmentation logic
        """
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MediaPipe
        self._init_mediapipe(pose_model_path, hand_model_path)
        
        # Load COBOT model
        self._load_model(model_path)
        
        self._init_original_tracking()
        
        print(f"✅ COBOT Action Recognition initialized on {self.device}")
        print(f"📊 Using original segmentation logic with gap filling after action collection")
        
    def _init_mediapipe(self, pose_model_path, hand_model_path):
        """Initialize MediaPipe pose and hand detection"""
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Pose detection
        base_options = python.BaseOptions(pose_model_path)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            running_mode=VisionRunningMode.VIDEO
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
        
        # Hand detection
        hand_base_options = python.BaseOptions(hand_model_path)
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            running_mode=VisionRunningMode.VIDEO, 
            num_hands=2, 
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        
    def _load_model(self, model_path):
        """Load the trained COBOT model"""
        self.model = AimCLR_v2_3views(
            base_encoder='net.st_gcn.Model', 
            pretrain=False,
            in_channels=3, 
            hidden_channels=32,
            hidden_dim=256, 
            num_class=19, 
            dropout=0.5,
            graph_args={'layout': 'cobot', 'strategy': 'distance'},
            edge_importance_weighting=True
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        self.model = self.model.to(self.device)
        print(f"✅ Model loaded from: {model_path}")
        
    def _init_original_tracking(self):
        """Initialize tracking variables using original logic from real_time.py"""
        # Original variables from real_time.py
        self.infer_len = 45  # From original code
        self.pose_list = deque(maxlen=self.infer_len)
        self.previous_2_hands = np.zeros((2, 21, 3))
        
        # State tracking (original logic)
        self.bottom_line_y = 0
        self.trigger_line = 0
        self.moving_hand_index = 0
        self.previous_moving_hand_index = 0
        self.number_diff_move = 0
        self.current_state = 0  # 0: idle, 1: moving, 2: trigger
        self.count_down = 0
        
        # Action collection
        self.pose_time = 0
        self.start_time = time.time()
        self.register_time = 3  # Time to establish bottom line
        
        # Prediction tracking
        self.predict_list = deque(maxlen=15)
        self.previous_label = None
        self.potential_list = deque(maxlen=20)
        
        # Probability aggregation option
        self.use_probability_aggregation = True  # Set to False for hard label voting
        self.probability_buffer = deque(maxlen=20)  # Store probabilities for aggregation
        
        # Statistics
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        
    def extract_raw_skeleton(self, frame, timestamp_ms):
        """
        Extract RAW 48-joint skeleton (preserving gaps for later cleaning)
        
        Returns:
            skeleton: (48, 3) numpy array with potential gaps (zeros)
            confidence: Overall detection confidence
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect pose and hands
        pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
        hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
        
        # Extract hands (preserve gaps - don't fill with previous)
        hands = self._extract_hands_raw(hand_result)
        
        # Extract arms
        arms = self._extract_arms_raw(pose_result)
        
        # Combine into 48-joint skeleton
        skeleton = np.concatenate([hands, arms], axis=0)  # (48, 3)
        
        # Zero out z-axis for 2D consistency
        skeleton[:, 2] = 0.0
        
        # Calculate confidence
        confidence = self._calculate_confidence(pose_result, hand_result)
        
        return skeleton, confidence
        
    def _extract_hands_raw(self, hand_result):
        """Extract hand landmarks preserving gaps (no gap filling here)"""
        right_hand = np.zeros((21, 3))
        left_hand = np.zeros((21, 3))
        
        if hand_result.hand_landmarks:
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                # Extract x, y coordinates, zero z-axis
                hand_array = np.array([[lm.x, lm.y, 0.0] for lm in hand_landmarks])
                
                if handedness[0].category_name == 'Right':
                    right_hand = hand_array
                elif handedness[0].category_name == 'Left':
                    left_hand = hand_array
        
        # Update previous hands for potential visualization (but don't use for gap filling)
        if not np.all(right_hand == 0):
            self.previous_2_hands[0] = right_hand
        if not np.all(left_hand == 0):
            self.previous_2_hands[1] = left_hand
            
        return np.concatenate([right_hand, left_hand], axis=0)  # (42, 3)
    
    def _extract_arms_raw(self, pose_result):
        """Extract arm landmarks"""
        arm_joints = np.zeros((6, 3))
        
        if pose_result.pose_landmarks:
            keypoints = [
                solutions.pose.PoseLandmark.LEFT_SHOULDER.value,   # 0
                solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,  # 1
                solutions.pose.PoseLandmark.LEFT_ELBOW.value,      # 2
                solutions.pose.PoseLandmark.RIGHT_ELBOW.value,     # 3
                solutions.pose.PoseLandmark.LEFT_WRIST.value,      # 4
                solutions.pose.PoseLandmark.RIGHT_WRIST.value      # 5
            ]
            
            for i, keypoint_idx in enumerate(keypoints):
                if keypoint_idx < len(pose_result.pose_landmarks[0]):
                    landmark = pose_result.pose_landmarks[0][keypoint_idx]
                    arm_joints[i] = [landmark.x, landmark.y, 0.0]
        
        return arm_joints  # (6, 3)
    
    def _calculate_confidence(self, pose_result, hand_result):
        """Calculate overall detection confidence"""
        confidences = []
        
        if hand_result.hand_landmarks:
            for handedness in hand_result.handedness:
                confidences.append(handedness[0].score)
        
        if pose_result.pose_landmarks:
            visible_joints = 0
            total_joints = len(pose_result.pose_landmarks[0])
            for landmark in pose_result.pose_landmarks[0]:
                if landmark.visibility > 0.5:
                    visible_joints += 1
            confidences.append(visible_joints / total_joints)
        
        return np.mean(confidences) if confidences else 0.0
    
    def set_bottom_line_y(self, pose):
        """Original bottom line detection"""
        margin = 0.05
        self.bottom_line_y = max(max(pose[42:48, 1]), self.bottom_line_y) 
        self.bottom_line_y -= margin * self.bottom_line_y  
        self.trigger_line = self.bottom_line_y + 1.25 * margin * self.bottom_line_y 

    def compute_var_when_move(self, pose_list):
        """Original movement variance calculation"""
        pose_np = np.array(pose_list)  # shape:(frame, 48, 3)
        hand_1_var = np.var(pose_np[:, :21, 1])
        hand_2_var = np.var(pose_np[:, 21:42, 1])
        return hand_1_var, hand_2_var

    def set_state(self, pose_list):
        """Original state detection logic"""
        pose_list = list(pose_list) 
        if len(pose_list) > 0:
            pose_list = np.array(pose_list)
        else:
            return 0
        if pose_list.shape[0] < 33:
            return 0
        if np.mean(pose_list[:, :, :]) == 0:
            return 0
        
        hand_var_1, hand_var_2 = self.compute_var_when_move(pose_list)

        frame_5 = int((self.infer_len-6))
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
                self.moving_hand_index=0 #right hand
        if not(abs(np.mean(key_42)- np.mean(key_44)) <threshold) or not(abs(np.mean(key_42)- np.mean(key_46)) <threshold) and hand_var_2>hand_var_1:
            if not(np.mean(key_42_y) < np.mean(key_44_y) < np.mean(key_46_y)):
                self.moving_hand_index=1 #left hand

        # Control moving hand index not to jump around
        if self.moving_hand_index != self.previous_moving_hand_index:
            self.moving_hand_index = self.previous_moving_hand_index
            self.number_diff_move += 1
        if self.number_diff_move > 5:
            self.moving_hand_index = 1 - self.moving_hand_index
            self.number_diff_move = 0
            self.previous_moving_hand_index = self.moving_hand_index

        n = len(pose_list)
        n_5 = n // 11
        first_5 = np.array(pose_list[:n_5])
        last_5 = np.array(pose_list[-n_5:])

        if self.moving_hand_index == 0:
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

        if wrist_y_first>self.bottom_line_y and wrist_y_last < self.bottom_line_y:
            if self.current_state == 0:
                self.current_state = 1

        if wrist_y_first<self.bottom_line_y and wrist_y_last > self.bottom_line_y and max_last_5 > self.trigger_line and abs(wrist_x-elbow_x)<0.03 :
            if self.current_state == 1:
                self.current_state = 2

        if wrist_y_last>self.bottom_line_y:
            self.count_down+=1
            if self.count_down>30 and wrist_var <0.01:
                if self.current_state ==1:
                    self.current_state=2
                    self.count_down=0
        else:
            self.count_down = 0

    def predict_action(self, pose_list):
        """
        Predict action from pose sequence - applies gap filling HERE (like original)
        
        Args:
            pose_list: Raw pose sequence with potential gaps (N, 48, 3)
            
        Returns:
            predicted_class: Predicted action class
            confidence: Prediction confidence  
            class_name: Action name
        """
        if len(pose_list) == 0:
            return -1, 0.0, "Unknown", np.zeros(19)
            
        try:
            pose_array = np.array(pose_list)  # (N, 48, 3)
            
            print(f"📊 Raw action sequence: {pose_array.shape}")
            
            # Calculate initial data quality
            initial_valid_ratio = np.sum(~np.all(pose_array == 0, axis=2)) / (pose_array.shape[0] * pose_array.shape[1])
            print(f"📊 Initial data quality: {initial_valid_ratio:.2%} valid joints")
            
            pose_cleaned, cleaning_stats = clean_micro_gaps(pose_array, max_gap=20)
            
            print(f"🧹 Gap filling results:")
            print(f"   - Initial valid ratio: {cleaning_stats['initial_valid_ratio']:.2%}")
            print(f"   - Final valid ratio: {cleaning_stats['final_valid_ratio']:.2%}")
            print(f"   - Data improvement: {cleaning_stats['improvement']:+.2%}")
            print(f"   - Joints imputed: {cleaning_stats['imputed_joints']}")
            print(f"   - Target bones computed: {cleaning_stats['target_bones']}")
            
            if cleaning_stats['improvement'] > 0:
                print(f"✅ Gap filling successfully improved data quality")
            else:
                print(f"ℹ️ No significant gaps found to fill")
            
            # Step 2: Transform to model format (like original)
            pose_transposed = np.transpose(pose_cleaned, (2, 0, 1)).astype(np.float32, copy=False)  # (3,N,48)
            pose_transposed = pose_transposed[..., np.newaxis]  # (3,N,48,1)
            
            # Step 3: Resize to target length
            pose_resized, original_length, method_used = fit_to_length(
                pose_transposed, 
                self.sequence_length, 
                'uniform-sample'
            )  # (3, sequence_length, 48, 1)
            
            print(f"📏 Sequence resized from {original_length} to {self.sequence_length} frames using {method_used}")
            
            # Step 4: Convert to tensor and predict
            pose_tensor = torch.tensor(pose_resized, dtype=torch.float32)
            pose_tensor = pose_tensor.unsqueeze(0).to(self.device)  # (1, 3, T, 48, 1)
            
            with torch.no_grad():
                start_time = time.time()
                output = self.model(None, pose_tensor, stream='all')
                inference_time = time.time() - start_time
                
                # Get prediction
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)
                
                predicted_class = predicted_class.item()
                confidence = confidence.item()
                probabilities_np = probabilities.cpu().numpy()[0]  # Convert to numpy for aggregation
                
                print(f"🔮 Prediction: {ACTION_NAMES[predicted_class]} (confidence: {confidence:.3f}, time: {inference_time:.3f}s)")
                
                return predicted_class, confidence, ACTION_NAMES[predicted_class], probabilities_np
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return -1, 0.0, "Error", np.zeros(19)  # Return zero probabilities on error

    def aggregate_probabilities(self):
        """
        Aggregate probabilities across the segment and make final decision
        
        Returns:
            final_class: Final predicted class based on aggregated probabilities
            final_confidence: Confidence of the final prediction
            final_class_name: Name of the final predicted class
        """
        if len(self.probability_buffer) == 0:
            return 0, 0.0, ACTION_NAMES[0]
        
        # Convert to numpy array and aggregate
        prob_array = np.array(list(self.probability_buffer))  # Shape: (N, 19)
        
        # Aggregate probabilities (mean across time)
        aggregated_probs = np.mean(prob_array, axis=0)
        
        # Get final prediction
        final_class = np.argmax(aggregated_probs)
        final_confidence = aggregated_probs[final_class]
        
        print(f"📊 Probability aggregation: {len(self.probability_buffer)} predictions averaged")
        print(f"🎯 Final aggregated prediction: {ACTION_NAMES[final_class]} (confidence: {final_confidence:.3f})")
        
        return final_class, final_confidence, ACTION_NAMES[final_class]

    def draw_visualization(self, frame, skeleton, state_info):
        """Draw visualization overlay on frame"""
        height, width = frame.shape[:2]
        frame_vis = frame.copy()
        
        # Draw skeleton with gaps (for debugging)
        frame_vis = self._draw_skeleton(frame_vis, skeleton)
        
        # State colors
        state_colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}  # idle, moving, trigger
        state_names = {0: 'IDLE', 1: 'MOVING', 2: 'TRIGGER'}
        
        color = state_colors.get(self.current_state, (255, 255, 255))
        
        # State text
        cv2.putText(frame_vis, f"State: {state_names[self.current_state]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Pose collection info
        cv2.putText(frame_vis, f"Buffer: {len(self.pose_list)}/{self.infer_len}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw bottom line and trigger line
        if self.bottom_line_y > 0:
            cv2.rectangle(frame_vis, (0, int(self.bottom_line_y * height)), 
                         (width, int(self.trigger_line * height)), color, 2)
        
        # Performance info
        if self.processing_times:
            avg_time = np.mean(list(self.processing_times))
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(frame_vis, f"FPS: {fps:.1f}", 
                       (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_vis
    
    def _draw_skeleton(self, frame, skeleton):
        """Draw skeleton on frame"""
        height, width = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        skeleton_px = skeleton.copy()
        skeleton_px[:, 0] *= width   # x coordinates
        skeleton_px[:, 1] *= height  # y coordinates
        
        # Draw hand connections
        self._draw_hand_connections(frame, skeleton_px[:21], (0, 255, 0))    # Right hand - green
        self._draw_hand_connections(frame, skeleton_px[21:42], (255, 0, 0))  # Left hand - blue
        
        # Draw arm connections  
        self._draw_arm_connections(frame, skeleton_px[42:], (0, 0, 255))     # Arms - red
        
        return frame
    
    def _draw_hand_connections(self, frame, hand_joints, color):
        """Draw hand joint connections"""
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (5, 6), (6, 7), (7, 8),
            # Middle  
            (9, 10), (10, 11), (11, 12),
            # Ring
            (13, 14), (14, 15), (15, 16),
            # Pinky
            (17, 18), (18, 19), (19, 20),
            # Palm
            (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
        ]
        
        for start_idx, end_idx in connections:
            if (start_idx < len(hand_joints) and end_idx < len(hand_joints) and
                not np.all(hand_joints[start_idx] == 0) and not np.all(hand_joints[end_idx] == 0)):
                
                start_point = tuple(map(int, hand_joints[start_idx][:2]))
                end_point = tuple(map(int, hand_joints[end_idx][:2]))
                cv2.line(frame, start_point, end_point, color, 2)
        
        # Draw joints
        for joint in hand_joints:
            if not np.all(joint == 0):
                center = tuple(map(int, joint[:2]))
                cv2.circle(frame, center, 3, color, -1)
    
    def _draw_arm_connections(self, frame, arm_joints, color):
        """Draw arm joint connections"""
        connections = [
            (0, 2), (2, 4),  # Left arm
            (1, 3), (3, 5),  # Right arm
            (0, 1)           # Shoulders
        ]
        
        for start_idx, end_idx in connections:
            if (start_idx < len(arm_joints) and end_idx < len(arm_joints) and
                not np.all(arm_joints[start_idx] == 0) and not np.all(arm_joints[end_idx] == 0)):
                
                start_point = tuple(map(int, arm_joints[start_idx][:2]))
                end_point = tuple(map(int, arm_joints[end_idx][:2]))
                cv2.line(frame, start_point, end_point, color, 2)
        
        # Draw joints
        for joint in arm_joints:
            if not np.all(joint == 0):
                center = tuple(map(int, joint[:2]))
                cv2.circle(frame, center, 5, color, -1)

    def process_video(self, video_path, output_csv=None, show_visualization=True):
        """Process video using original segmentation logic"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 Processing video: {video_path}")
        print(f"📊 FPS: {fps:.2f}, Total frames: {total_frames}")
        
        # Initialize CSV output
        csv_writer = None
        if output_csv:
            csv_file = open(output_csv, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['frame', 'timestamp', 'start_frame', 'end_frame', 'action_class', 'action_name', 'confidence'])
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get timestamp
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                
                # Extract RAW skeleton (preserve gaps)
                skeleton, confidence = self.extract_raw_skeleton(frame_rgb, timestamp_ms)
                
                # Add to pose buffer
                self.pose_list.append(skeleton)
                
                current_time = time.time()
                if current_time - self.start_time < self.register_time:
                    self.set_bottom_line_y(skeleton)
                
                self.set_state(self.pose_list)
                
                prediction_info = None
                if self.current_state == 1:  # Moving
                    self.pose_time += 1
                    if self.pose_time == 1: 
                        self.start_frame = self.frame_count
                        self.start_time_action = time.time()

                    # Collect pose for prediction
                    input_pose = np.array(list(self.pose_list))
                    predicted_label, pred_confidence, class_name, probabilities = self.predict_action(input_pose)

                    # Store predictions based on aggregation method
                    if self.use_probability_aggregation:
                        # Store probabilities for later aggregation
                        self.probability_buffer.append(probabilities)
                    else:
                        # Use traditional hard label voting
                        self.predict_list.append(predicted_label)
                        if predicted_label != 0 and predicted_label == self.previous_label and self.pose_time <= 75:  
                            self.potential_list.append(predicted_label)

                    self.previous_label = predicted_label
                    prediction_info = (predicted_label, pred_confidence, class_name)

                elif self.current_state == 2:  # Trigger (action complete)
                    self.end_frame = self.frame_count
                    self.end_time_action = time.time()
                    
                    # Determine final prediction based on aggregation method
                    if self.use_probability_aggregation and len(self.probability_buffer) > 0:
                        # Use probability aggregation
                        final_label, final_confidence, final_class_name = self.aggregate_probabilities()
                        print(f"🎯 Action completed (Probability Aggregation): {final_class_name} (frames {self.start_frame}-{self.end_frame})")
                    else:
                        # Use traditional voting
                        if len(self.potential_list) > 0:  
                            final_label = mode(self.potential_list)
                            final_confidence = pred_confidence if 'pred_confidence' in locals() else 0.0
                        else:
                            final_label = mode(self.predict_list) if self.predict_list else 0
                            final_confidence = pred_confidence if 'pred_confidence' in locals() else 0.0
                        
                        final_class_name = ACTION_NAMES[final_label]
                        print(f"🎯 Action completed (Hard Label Voting): {final_class_name} (frames {self.start_frame}-{self.end_frame})")

                    # Save to CSV
                    if csv_writer:
                        csv_writer.writerow([
                            self.frame_count, timestamp_ms, self.start_frame, self.end_frame,
                            final_label, final_class_name, final_confidence
                        ])

                    # Reset for next action
                    self.predict_list.clear()
                    self.potential_list.clear()
                    self.probability_buffer.clear()
                    self.current_state = 0
                    self.pose_time = 0
                
                # Update timing
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Visualization
                if show_visualization:
                    frame_vis = self.draw_visualization(
                        frame_rgb, skeleton, 
                        {'state': self.current_state, 'confidence': confidence}
                    )
                    
                    # Show prediction if available
                    if prediction_info:
                        pred_class, pred_confidence, class_name = prediction_info
                        if pred_confidence > 0.3:
                            cv2.putText(frame_vis, f"Action: {class_name}", 
                                       (10, frame_vis.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame_vis, f"Confidence: {pred_confidence:.2f}", 
                                       (10, frame_vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Convert back to BGR for OpenCV display
                    frame_vis_bgr = cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR)
                    cv2.imshow('COBOT Action Recognition', frame_vis_bgr)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                self.frame_count += 1
                
                # Progress update
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    avg_fps = 1.0 / np.mean(list(self.processing_times))
                    print(f"📈 Progress: {progress:.1f}% ({self.frame_count}/{total_frames}) - Avg FPS: {avg_fps:.1f}")
        
        finally:
            cap.release()
            if show_visualization:
                cv2.destroyAllWindows()
            if csv_writer:
                csv_file.close()
                print(f"💾 Results saved to: {output_csv}")
        
        print(f"✅ Video processing completed!")

def main():
    """Main function"""
    
    # Configuration
    MODEL_PATH = r"C:\AimCLR-v2\work_dir\cobot_3views_2D_xsub_medgap_aug1\finetune\best_model.pt"
    POSE_MODEL_PATH = r"mediapipe_weights\pose_landmarker_full.task"
    HAND_MODEL_PATH = r"mediapipe_weights\hand_landmarker.task"
    VIDEO_PATH = r"C:\AimCLR-v2\s2_Alex_1.mp4"
    OUTPUT_CSV = "fixed_predictions.csv"
    
    # Prediction aggregation method
    USE_PROBABILITY_AGGREGATION = True  # Set to False for traditional hard label voting
    
    # Initialize system
    try:
        recognition_system = COBOTActionRecognition(
            model_path=MODEL_PATH,
            pose_model_path=POSE_MODEL_PATH,
            hand_model_path=HAND_MODEL_PATH,
            sequence_length=64
        )
        
        # Configure aggregation method
        recognition_system.use_probability_aggregation = USE_PROBABILITY_AGGREGATION
        aggregation_method = "Probability Aggregation" if USE_PROBABILITY_AGGREGATION else "Hard Label Voting"
        print(f"🎯 Using {aggregation_method} for final predictions")
        
        # Process video
        recognition_system.process_video(
            video_path=VIDEO_PATH,
            output_csv=OUTPUT_CSV,
            show_visualization=True
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
