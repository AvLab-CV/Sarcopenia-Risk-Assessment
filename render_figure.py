import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.collections import LineCollection
import einops

# Kinect V2 skeleton bone connections (25 joints)
BONE_CONNECTIONS = [
    (0, 1), (1, 20), (20, 2), (2, 3),  # Spine to head
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),  # Left arm
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),  # Right arm
    (0, 12), (12, 13), (13, 14), (14, 15),  # Left leg
    (0, 16), (16, 17), (17, 18), (18, 19),  # Right leg
]
OUT = Path("output/figures/")

def apply_isometric_projection(points):
    """Apply isometric projection to 3D points."""
    # Isometric projection matrix
    # Standard isometric: rotate 45° around Y, then ~35.264° around X
    angle_y = np.pi / 4
    angle_x = np.arctan(1/np.sqrt(2))
    
    # Rotation around Y
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    
    # Rotation around X
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    
    # Combined rotation
    R = Rx @ Ry
    
    # Apply rotation and drop Z coordinate for 2D projection
    projected = points @ R.T
    return projected[:, :2]

def render_skeleton_sequence(skeleton_data, img_width=400, img_height=800, 
                             padding=50, line_width=2):
    """
    Render a sequence of skeleton frames as 2D images.
    
    Parameters:
    -----------
    skeleton_data : np.ndarray
        Shape (T, 25, 3) - T frames, 25 joints, 3D coordinates
    img_width : int
        Width of output images (portrait aspect ratio 1:2)
    img_height : int
        Height of output images (should be 2x width)
    padding : int
        Padding around the skeleton in pixels
    line_width : float
        Width of skeleton bones
    
    Returns:
    --------
    images : np.ndarray
        Shape (T, img_height, img_width, 3) - RGB images
    """
    T = skeleton_data.shape[0]
    images = []
    
    # Find global bounds for consistent scaling
    all_projected = []
    for t in range(T):
        projected = apply_isometric_projection(skeleton_data[t])
        all_projected.append(projected)
    
    all_projected = np.concatenate(all_projected, axis=0)
    x_min, y_min = all_projected.min(axis=0)
    x_max, y_max = all_projected.max(axis=0)
    
    # Add padding to bounds
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    for t in range(T):
        # Create figure with portrait aspect ratio
        fig, ax = plt.subplots(figsize=(img_width/100, img_height/100), dpi=100)
        ax.set_aspect('equal')
        
        # Project 3D skeleton to 2D isometric view
        projected = apply_isometric_projection(skeleton_data[t])
        
        # Draw bones
        segments = []
        for joint1, joint2 in BONE_CONNECTIONS:
            segments.append([projected[joint1], projected[joint2]])
        
        line_collection = LineCollection(segments, colors='blue', 
                                        linewidths=line_width)
        ax.add_collection(line_collection)
        
        # Draw joints
        ax.scatter(projected[:, 0], projected[:, 1], c='red', s=30, zorder=5)
        
        # Set axis limits with padding
        ax.set_xlim(x_min - padding/100, x_max + padding/100)
        ax.set_ylim(y_min - padding/100, y_max + padding/100)
        
        # Remove axes
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # Convert to numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(img)
        
        plt.close(fig)
    
    return np.array(images)

def render_motion_trail(skeleton_data, frame_indices, img_width=400, 
                       img_height=800, spacing_x=0.5, spacing_y=0.5, padding=50, 
                       line_width=2, alpha_fade=True):
    """
    Render multiple skeleton poses in a single image with fixed spacing between them.
    
    Parameters:
    -----------
    skeleton_data : np.ndarray
        Shape (T, 25, 3) - T frames, 25 joints, 3D coordinates
    frame_indices : list or np.ndarray
        Indices of frames to render (e.g., [0, 5, 10, 15])
    img_width : int
        Width of output image (portrait aspect ratio 1:2)
    img_height : int
        Height of output image (should be 2x width)
    spacing : float
        Horizontal spacing between consecutive skeletons in world units
    padding : int
        Padding around the composed skeletons in pixels
    line_width : float
        Width of skeleton bones
    alpha_fade : bool
        If True, earlier poses are more transparent
    
    Returns:
    --------
    image : np.ndarray
        Shape (img_height, img_width, 3) - RGB image
    """
    n_poses = len(frame_indices)
    
    # Project all selected frames
    projected_poses = []
    for idx in frame_indices:
        projected = apply_isometric_projection(skeleton_data[idx])
        projected_poses.append(projected)
    
    # Apply horizontal translation to each pose
    translated_poses = []
    for i, projected in enumerate(projected_poses):
        translated = projected.copy()
        translated[:, 0] += i * spacing_x  # Shift horizontally
        translated[:, 1] += i * spacing_y  # Shift vertically
        translated_poses.append(translated)
    
    # Find global bounds for all translated poses
    all_points = np.concatenate(translated_poses, axis=0)
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(img_width/100, img_height/100), dpi=100)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_aspect('equal')
    
    # Draw each skeleton pose
    for i, (idx, projected) in enumerate(zip(frame_indices, translated_poses)):
        # Calculate alpha for fading effect
        if alpha_fade:
            alpha = 0.3 + 0.7 * (i / max(n_poses - 1, 1))
        else:
            alpha = 1.0
        
        # Color gradient from blue to red
        color_ratio = i / max(n_poses - 1, 1)
        color = plt.cm.coolwarm(color_ratio)
        # color = (1,0,0)
        
        # Draw bones
        segments = []
        for joint1, joint2 in BONE_CONNECTIONS:
            segments.append([projected[joint1], projected[joint2]])
        
        line_collection = LineCollection(segments, colors=[color], 
                                        linewidths=line_width, alpha=alpha)
        ax.add_collection(line_collection)
        
        # Draw joints
        ax.scatter(projected[:, 0], projected[:, 1], 
                  c=[color], s=20, alpha=alpha, zorder=5)
    
    # Set axis limits with padding
    ax.set_xlim(x_min - padding/100, x_max + padding/100)
    ax.set_ylim(y_min - padding/100, y_max + padding/100)
    
    # Remove axes
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Convert to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Convert ARGB to RGBA
    img = np.roll(img, 3, axis=2)
    
    plt.close(fig)
    
    return img

# Example usage
if __name__ == "__main__":
    skeleton_data = dict(np.load('output/merged/skels.npz'))
    skeleton_data = skeleton_data["2_022.mp4"]
    skeleton_data = einops.rearrange(skeleton_data, "time (joints coord) -> time joints coord", joints=25, coord=3)
    skeleton_data[:, :, 1] *= -1
    T = skeleton_data.shape[0]

    # images = render_skeleton_sequence(skeleton_data[::5], img_width=400, img_height=800)

    frame_indices = range(0, 90, 15)  # Select specific frames
    motion_trail = render_motion_trail(
        skeleton_data, 
        frame_indices,
        img_width=800,  # Wider to accommodate multiple poses
        img_height=800,
        padding=0,
        spacing_x=0.5,  # Spacing between skeletons
        spacing_y=-0.3,  # Spacing between skeletons
        alpha_fade=False,  # Fade earlier poses
        line_width=5
    )
    
    print(f"Generated motion trail image of shape {motion_trail.shape}")
    
    # Display the motion trail
    plt.figure(figsize=(8, 8))
    plt.imshow(motion_trail)
    plt.axis('off')
    plt.title('Motion Trail - Isometric View')
    plt.tight_layout()
    plt.show()

    plt.imsave(OUT / 'motion_trail.png', motion_trail)

