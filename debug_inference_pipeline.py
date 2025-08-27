#!/usr/bin/env python3
"""
Debug script for COBOT real-time inference pipeline
Helps identify issues in each step of the pipeline
"""

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from real_time_improved import COBOTRealTimeInference, ACTION_NAMES
import time

def test_model_loading():
    """Test if the model loads correctly"""
    print("🔍 Testing model loading...")
    
    MODEL_PATH = r"C:\AimCLR-v2\work_dir\cobot_3views_2D_xsub_medgap_aug1\finetune\best_model.pt"
    
    try:
        # Test model loading
        from net.aimclr_v2_3views import AimCLR_v2_3views
        
        model = AimCLR_v2_3views(
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
        
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        print(f"✅ Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model loaded with 'model_state_dict' key")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Model loaded directly from checkpoint")
            
        model.eval()
        print(f"✅ Model architecture: {model}")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 3, 64, 48, 1)
        with torch.no_grad():
            output = model(None, dummy_input, stream='all')
            print(f"✅ Model forward pass successful: {output.shape}")
            print(f"✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mediapipe_detection():
    """Test MediaPipe pose and hand detection"""
    print("\n🔍 Testing MediaPipe detection...")
    
    POSE_MODEL_PATH = r"mediapipe_weights\pose_landmarker_full.task"
    HAND_MODEL_PATH = r"mediapipe_weights\hand_landmarker.task"
    
    try:
        # Initialize inference system
        inference_system = COBOTRealTimeInference(
            model_path=r"C:\AimCLR-v2\work_dir\cobot_3views_2D_xsub_medgap_aug1\finetune\best_model.pt",
            pose_model_path=POSE_MODEL_PATH,
            hand_model_path=HAND_MODEL_PATH,
            sequence_length=64
        )
        
        # Test with a sample frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        timestamp_ms = int(time.time() * 1000)
        
        skeleton, confidence = inference_system.extract_skeleton(test_frame, timestamp_ms)
        
        print(f"✅ Skeleton extraction successful")
        print(f"✅ Skeleton shape: {skeleton.shape}")
        print(f"✅ Detection confidence: {confidence:.3f}")
        print(f"✅ Non-zero joints: {np.sum(~np.all(skeleton == 0, axis=1))}/48")
        
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocessing_pipeline():
    """Test the preprocessing pipeline with synthetic data"""
    print("\n🔍 Testing preprocessing pipeline...")
    
    try:
        # Create synthetic skeleton sequence
        sequence_length = 45
        skeleton_sequence = []
        
        for t in range(sequence_length):
            # Create realistic hand movements
            skeleton = np.zeros((48, 3))
            
            # Add some movement to hands (joints 0-41)
            for j in range(42):
                skeleton[j, 0] = 0.5 + 0.1 * np.sin(t * 0.1 + j * 0.1)  # x
                skeleton[j, 1] = 0.5 + 0.1 * np.cos(t * 0.1 + j * 0.1)  # y
                skeleton[j, 2] = 0.0  # z
            
            # Add arm joints (joints 42-47)
            arm_positions = [
                [0.3, 0.3, 0.0],  # LEFT_SHOULDER
                [0.7, 0.3, 0.0],  # RIGHT_SHOULDER
                [0.2, 0.5, 0.0],  # LEFT_ELBOW
                [0.8, 0.5, 0.0],  # RIGHT_ELBOW
                [0.1, 0.7, 0.0],  # LEFT_WRIST
                [0.9, 0.7, 0.0],  # RIGHT_WRIST
            ]
            skeleton[42:48] = arm_positions
            
            skeleton_sequence.append(skeleton)
        
        sequence_array = np.array(skeleton_sequence)  # (T, 48, 3)
        print(f"✅ Synthetic sequence created: {sequence_array.shape}")
        
        # Test cleaning
        from cleaning import clean_micro_gaps, fit_to_length
        
        sequence_cleaned, cleaning_stats = clean_micro_gaps(sequence_array, max_gap=6)
        print(f"✅ Cleaning successful: improvement {cleaning_stats['improvement']:+.2%}")
        
        # Test resizing
        sequence_transposed = np.transpose(sequence_cleaned, (2, 0, 1))  # (3, T, 48)
        sequence_transposed = sequence_transposed[..., np.newaxis]  # (3, T, 48, 1)
        
        sequence_resized, _, _ = fit_to_length(sequence_transposed, 64, 'uniform-sample')
        print(f"✅ Resizing successful: {sequence_resized.shape}")
        
        # Test tensor conversion
        sequence_tensor = torch.tensor(sequence_resized, dtype=torch.float32)
        sequence_tensor = sequence_tensor.unsqueeze(0)  # (1, 3, 64, 48, 1)
        print(f"✅ Tensor conversion successful: {sequence_tensor.shape}")
        
        return sequence_tensor
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_full_inference():
    """Test the complete inference pipeline"""
    print("\n🔍 Testing full inference pipeline...")
    
    try:
        # Initialize inference system
        inference_system = COBOTRealTimeInference(
            model_path=r"C:\AimCLR-v2\work_dir\cobot_3views_2D_xsub_medgap_aug1\finetune\best_model.pt",
            pose_model_path=r"mediapipe_weights\pose_landmarker_full.task",
            hand_model_path=r"mediapipe_weights\hand_landmarker.task",
            sequence_length=64
        )
        
        # Get preprocessed data from previous test
        preprocessed_data = test_preprocessing_pipeline()
        
        if preprocessed_data is not None:
            # Test prediction
            pred_class, confidence, class_name = inference_system.predict_action(preprocessed_data)
            
            print(f"✅ Prediction successful:")
            print(f"   Class: {pred_class}")
            print(f"   Name: {class_name}")
            print(f"   Confidence: {confidence:.3f}")
            
            return True
        else:
            print("❌ Cannot test inference without preprocessed data")
            return False
            
    except Exception as e:
        print(f"❌ Full inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_action_segmentation():
    """Test action boundary detection"""
    print("\n🔍 Testing action segmentation...")
    
    try:
        # Initialize inference system
        inference_system = COBOTRealTimeInference(
            model_path=r"C:\AimCLR-v2\work_dir\cobot_3views_2D_xsub_medgap_aug1\finetune\best_model.pt",
            pose_model_path=r"mediapipe_weights\pose_landmarker_full.task",
            hand_model_path=r"mediapipe_weights\hand_landmarker.task",
            sequence_length=64
        )
        
        print("Testing action boundary detection with synthetic movement...")
        
        # Simulate a sequence with clear action boundaries
        for frame_idx in range(100):
            # Create skeleton with varying movement
            skeleton = np.zeros((48, 3))
            
            # Simulate action: high movement in middle, low at start/end
            if 20 <= frame_idx <= 70:  # Action period
                movement_amplitude = 0.2
            else:  # Rest period
                movement_amplitude = 0.01
            
            # Add movement to hands
            for j in range(42):
                skeleton[j, 0] = 0.5 + movement_amplitude * np.sin(frame_idx * 0.2 + j * 0.1)
                skeleton[j, 1] = 0.5 + movement_amplitude * np.cos(frame_idx * 0.2 + j * 0.1)
                skeleton[j, 2] = 0.0
            
            # Add static arms
            skeleton[42:48] = [[0.3, 0.3, 0.0], [0.7, 0.3, 0.0], [0.2, 0.5, 0.0], 
                              [0.8, 0.5, 0.0], [0.1, 0.7, 0.0], [0.9, 0.7, 0.0]]
            
            confidence = 0.8  # High confidence
            
            # Test boundary detection
            action_detected, action_data = inference_system.detect_action_boundaries(skeleton, confidence)
            
            if action_detected:
                print(f"✅ Action detected at frame {frame_idx}")
                if action_data is not None:
                    print(f"   Preprocessed data shape: {action_data.shape}")
                    
                    # Test prediction on detected action
                    pred_class, pred_confidence, class_name = inference_system.predict_action(action_data)
                    print(f"   Prediction: {class_name} (confidence: {pred_confidence:.3f})")
            
            # Print state changes
            current_state = inference_system.state
            if hasattr(test_action_segmentation, 'prev_state'):
                if current_state != test_action_segmentation.prev_state:
                    print(f"🔄 State changed: {test_action_segmentation.prev_state} → {current_state} at frame {frame_idx}")
            test_action_segmentation.prev_state = current_state
        
        return True
        
    except Exception as e:
        print(f"❌ Action segmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all pipeline tests"""
    print("🧪 COBOT Inference Pipeline Debug Tests")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("MediaPipe Detection", test_mediapipe_detection),
        ("Preprocessing Pipeline", lambda: test_preprocessing_pipeline() is not None),
        ("Full Inference", test_full_inference),
        ("Action Segmentation", test_action_segmentation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"❌ FAILED: {test_name} - {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY:")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Your pipeline should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("💡 Common issues:")
        print("   - Missing MediaPipe model files")
        print("   - Incorrect model path or architecture mismatch")
        print("   - Missing dependencies (scipy for cleaning)")
        print("   - GPU/CUDA issues")

if __name__ == "__main__":
    run_all_tests()
