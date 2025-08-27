import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_cobot_skeleton():
    """Visualize COBOT skeleton structure with bone connections"""
    
    # COBOT bone connections
    bone_connections = [
        # Arm and shoulder connections (6 joints: 43-48)
        (43, 44), (44, 45), (45, 46), (46, 47), (47, 48),
        # Right hand connections (21 joints: 1-21)
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
        (17, 18), (18, 19), (19, 20), (20, 21),
        # Left hand connections (21 joints: 22-42)
        (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31),
        (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),
        (38, 39), (39, 40), (40, 41), (41, 42),
        # Connect hands to arms
        (21, 43), (42, 48),  # Connect right hand to right arm, left hand to left arm
    ]
    
    # Load sample data to get joint positions
    import os
    sample_file = [f for f in os.listdir('pose_clean') if f.endswith('.npy')][0]
    data = np.load(os.path.join('pose_clean', sample_file))
    
    # Use first frame
    joints = data[0]  # Shape: (48, 3)
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='red', s=50, alpha=0.7)
    
    # Plot bone connections
    for start_joint, end_joint in bone_connections:
        # Convert to 0-indexed
        start_idx = start_joint - 1
        end_idx = end_joint - 1
        
        if start_idx < len(joints) and end_idx < len(joints):
            start_pos = joints[start_idx]
            end_pos = joints[end_idx]
            
            # Draw bone connection
            ax.plot([start_pos[0], end_pos[0]], 
                   [start_pos[1], end_pos[1]], 
                   [start_pos[2], end_pos[2]], 
                   'b-', linewidth=2, alpha=0.8)
    
    # Add joint numbers
    for i, (x, y, z) in enumerate(joints):
        ax.text(x, y, z, str(i+1), fontsize=8)
    
    # Add labels for different parts
    ax.text(joints[0, 0], joints[0, 1], joints[0, 2], 'Right Hand\n(Joints 1-21)', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    ax.text(joints[21, 0], joints[21, 1], joints[21, 2], 'Left Hand\n(Joints 22-42)', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7))
    ax.text(joints[42, 0], joints[42, 1], joints[42, 2], 'Arm/Shoulder\n(Joints 43-48)', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('COBOT Skeleton Structure\n42 Hand Joints (21 per hand) + 6 Arm/Shoulder Joints')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()
    
    # Print bone connection summary
    print("=== COBOT Bone Connections ===")
    print("Right Hand (Joints 1-21):")
    right_hand_connections = [conn for conn in bone_connections if 1 <= conn[0] <= 21 and 1 <= conn[1] <= 21]
    for conn in right_hand_connections:
        print(f"  Joint {conn[0]} - Joint {conn[1]}")
    
    print("\nLeft Hand (Joints 22-42):")
    left_hand_connections = [conn for conn in bone_connections if 22 <= conn[0] <= 42 and 22 <= conn[1] <= 42]
    for conn in left_hand_connections:
        print(f"  Joint {conn[0]} - Joint {conn[1]}")
    
    print("\nArm/Shoulder (Joints 43-48):")
    arm_connections = [conn for conn in bone_connections if 43 <= conn[0] <= 48 and 43 <= conn[1] <= 48]
    for conn in arm_connections:
        print(f"  Joint {conn[0]} - Joint {conn[1]}")
    
    print("\nHand-Arm Connections:")
    hand_arm_connections = [conn for conn in bone_connections if 
                          (1 <= conn[0] <= 21 and 43 <= conn[1] <= 48) or
                          (22 <= conn[0] <= 42 and 43 <= conn[1] <= 48) or
                          (43 <= conn[0] <= 48 and 1 <= conn[1] <= 21) or
                          (43 <= conn[0] <= 48 and 22 <= conn[1] <= 42)]
    for conn in hand_arm_connections:
        print(f"  Joint {conn[0]} - Joint {conn[1]}")

def analyze_joint_movement():
    """Analyze joint movement patterns in COBOT data"""
    
    import os
    files = [f for f in os.listdir('pose_clean') if f.endswith('.npy')][:5]
    
    print("\n=== Joint Movement Analysis ===")
    
    for filename in files:
        data = np.load(os.path.join('pose_clean', filename))
        
        # Calculate movement for each joint
        movement = np.linalg.norm(data[1:] - data[:-1], axis=2)  # Shape: (T-1, 48)
        avg_movement = movement.mean(axis=0)
        
        print(f"\nFile: {filename}")
        print("Most active joints (highest movement):")
        most_active = np.argsort(avg_movement)[-10:]  # Top 10
        for i, joint_idx in enumerate(most_active):
            print(f"  Joint {joint_idx+1}: {avg_movement[joint_idx]:.4f}")
        
        print("Least active joints (lowest movement):")
        least_active = np.argsort(avg_movement)[:10]  # Bottom 10
        for i, joint_idx in enumerate(least_active):
            print(f"  Joint {joint_idx+1}: {avg_movement[joint_idx]:.4f}")

if __name__ == '__main__':
    print("COBOT Skeleton Visualization and Analysis")
    print("=" * 50)
    
    # Visualize skeleton
    visualize_cobot_skeleton()
    
    # Analyze joint movement
    analyze_joint_movement() 