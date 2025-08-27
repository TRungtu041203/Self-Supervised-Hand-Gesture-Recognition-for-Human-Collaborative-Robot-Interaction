import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_cobot_data(data_path='pose_clean'):
    """Analyze COBOT data structure to help define bone connections"""
    
    print("=== COBOT Data Analysis ===")
    
    # Get all files
    files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    print(f"Found {len(files)} files")
    
    # Analyze a few sample files
    sample_files = files[:5]
    
    for filename in sample_files:
        file_path = os.path.join(data_path, filename)
        data = np.load(file_path)
        
        print(f"\nFile: {filename}")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Value range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"Mean: {data.mean():.4f}, Std: {data.std():.4f}")
        
        # Analyze joint positions
        T, V, C = data.shape
        print(f"Frames: {T}, Joints: {V}, Coordinates: {C}")
        
        # Check for zero joints (might indicate missing data)
        zero_joints = np.sum(np.all(data == 0, axis=(0, 2)))
        print(f"Zero joints: {zero_joints}")
        
        # Analyze joint movement
        if T > 1:
            movement = np.linalg.norm(data[1:] - data[:-1], axis=2)
            print(f"Average joint movement: {movement.mean():.4f}")
            print(f"Most active joints: {np.argsort(movement.mean(axis=0))[-5:]}")
    
    # Analyze file naming pattern
    print("\n=== File Naming Analysis ===")
    subjects = set()
    actions = set()
    
    for filename in files:
        parts = filename.replace('.npy', '').split('_')
        if len(parts) >= 3:
            subject = parts[0]
            action = parts[-1]
            subjects.add(subject)
            actions.add(action)
    
    print(f"Unique subjects: {len(subjects)}")
    print(f"Unique actions: {len(actions)}")
    print(f"Subjects: {sorted(subjects)}")
    print(f"Actions: {sorted(actions)}")
    
    # Count samples per subject
    subject_counts = defaultdict(int)
    for filename in files:
        subject = filename.split('_')[0]
        subject_counts[subject] += 1
    
    print("\nSamples per subject:")
    for subject, count in sorted(subject_counts.items()):
        print(f"  {subject}: {count}")

def visualize_skeleton(data, frame_idx=0, joint_idx=None):
    """Visualize skeleton structure"""
    if joint_idx is None:
        joint_idx = slice(None)
    
    skeleton = data[frame_idx, joint_idx, :]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot joints
    ax.scatter(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], c='red', s=50)
    
    # Add joint numbers
    for i, (x, y, z) in enumerate(skeleton):
        ax.text(x, y, z, str(i), fontsize=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Skeleton Visualization (Frame {frame_idx})')
    
    plt.show()

def suggest_bone_connections(data, threshold=0.1):
    """Suggest bone connections based on joint proximity"""
    # Use first frame of first file
    sample_file = [f for f in os.listdir('pose_clean') if f.endswith('.npy')][0]
    data = np.load(os.path.join('pose_clean', sample_file))
    
    # Get joint positions from first frame
    joints = data[0]  # Shape: (48, 3)
    
    # Calculate distances between all joint pairs
    n_joints = joints.shape[0]
    distances = np.zeros((n_joints, n_joints))
    
    for i in range(n_joints):
        for j in range(n_joints):
            if i != j:
                dist = np.linalg.norm(joints[i] - joints[j])
                distances[i, j] = dist
    
    # Find close pairs (potential bone connections)
    close_pairs = []
    for i in range(n_joints):
        for j in range(i+1, n_joints):
            if distances[i, j] < threshold:
                close_pairs.append((i+1, j+1, distances[i, j]))  # +1 for 1-indexed
    
    # Sort by distance
    close_pairs.sort(key=lambda x: x[2])
    
    print(f"\n=== Suggested Bone Connections (distance < {threshold}) ===")
    for i, j, dist in close_pairs[:20]:  # Show top 20
        print(f"Joint {i} - Joint {j}: {dist:.4f}")
    
    return close_pairs

if __name__ == '__main__':
    # Analyze data structure
    analyze_cobot_data()
    
    # Suggest bone connections
    print("\n" + "="*50)
    suggest_bone_connections(None)
    
    # Optional: Visualize skeleton (uncomment to use)
    # sample_file = [f for f in os.listdir('pose_clean') if f.endswith('.npy')][0]
    # data = np.load(os.path.join('pose_clean', sample_file))
    # visualize_skeleton(data) 