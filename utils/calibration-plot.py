import numpy as np
import plotly.graph_objects as go
import tomli
import sys
from scipy.spatial.transform import Rotation

def get_transform_matrix(rvec, tvec):
    """Convert rotation vector and translation to 4x4 transform matrix"""
    R = Rotation.from_rotvec(rvec).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T

def translate_cameras_to_reference(data, reference_camera='cam_0'):
    """Translate and rotate all cameras so reference camera is at origin with no rotation"""
    # Get reference camera transform
    ref_cam = data[reference_camera]
    ref_transform = get_transform_matrix(
        np.array(ref_cam['rotation']),
        np.array(ref_cam['translation'])
    )
    
    # Calculate the transform that would move reference camera to identity
    ref_transform_inv = np.linalg.inv(ref_transform)
    
    # Apply this transform to all cameras
    new_data = {}
    for key, cam in data.items():
        if not key.startswith('cam_'):
            continue
            
        # Get original camera transform
        orig_transform = get_transform_matrix(
            np.array(cam['rotation']),
            np.array(cam['translation'])
        )
        
        # Apply reference camera's inverse transform
        new_transform = ref_transform_inv @ orig_transform
        
        # Extract new rotation vector and translation
        new_rot = Rotation.from_matrix(new_transform[:3, :3])
        new_trans = new_transform[:3, 3]
        
        # Create new camera data
        new_data[key] = {
            'name': cam['name'],
            'size': cam['size'],
            'rotation': new_rot.as_rotvec().tolist(),
            'translation': new_trans.tolist()
        }
    
    return new_data

def create_camera_frustum(transform_matrix, size, scale=100):
    """Create camera frustum vertices and faces"""
    aspect_ratio = size[0] / size[1]
    
    # Create frustum points in camera's local frame
    base_points = np.array([
        [-aspect_ratio, -1, 2],  # Bottom left
        [aspect_ratio, -1, 2],   # Bottom right
        [aspect_ratio, 1, 2],    # Top right
        [-aspect_ratio, 1, 2],   # Top left
        [0, 0, 0]               # Camera center
    ]) * scale * 0.5
    
    # Add homogeneous coordinate
    points_h = np.hstack([base_points, np.ones((base_points.shape[0], 1))])
    
    # Transform points
    transformed_points = (transform_matrix @ points_h.T).T[:, :3]
    center = transformed_points[-1]
    points = transformed_points[:-1]
    
    # Create arrays for lines
    x, y, z = [], [], []
    
    # Add lines from center to corners
    for point in points:
        x.extend([center[0], point[0], None])
        y.extend([center[1], point[1], None])
        z.extend([center[2], point[2], None])
    
    # Add lines between corners
    for i in range(4):
        j = (i + 1) % 4
        x.extend([points[i,0], points[j,0], None])
        y.extend([points[i,1], points[j,1], None])
        z.extend([points[i,2], points[j,2], None])
        
    return x, y, z

def create_coordinate_axes(transform_matrix, scale=100):
    """Create coordinate axes lines"""
    axis_colors = ['red', 'green', 'blue']
    traces = []
    
    # Create axis endpoints in local frame
    origin = np.array([0, 0, 0, 1])
    for i in range(3):
        direction = np.zeros(4)
        direction[i] = scale
        direction[3] = 1
        
        # Transform points
        start = (transform_matrix @ origin)[:3]
        end = (transform_matrix @ direction)[:3]
        
        traces.append({
            'x': [start[0], end[0]],
            'y': [start[1], end[1]],
            'z': [start[2], end[2]],
            'color': axis_colors[i]
        })
        
    return traces

def plot_cameras(toml_path, reference_camera='cam_0', scale=100):
    # Load camera data
    with open(toml_path, 'rb') as f:
        data = tomli.load(f)
    
    # Transform all cameras relative to reference camera
    transformed_data = translate_cameras_to_reference(data, reference_camera)
    
    # Create figure
    fig = go.Figure()
    
    # Plot cameras
    positions = []
    for key, cam in transformed_data.items():
        # Get transform matrix
        transform = get_transform_matrix(
            np.array(cam['rotation']),
            np.array(cam['translation'])
        )
        position = transform[:3, 3]
        positions.append(position)
        
        # Add camera center
        fig.add_trace(go.Scatter3d(
            x=[position[0]],
            y=[position[1]],
            z=[position[2]],
            mode='markers+text',
            text=[cam['name']],
            textposition='top center',
            marker=dict(size=8, color='black'),
            name=cam['name'],
            showlegend=False
        ))
        
        # Add camera frustum
        fx, fy, fz = create_camera_frustum(transform, cam['size'], scale)
        fig.add_trace(go.Scatter3d(
            x=fx,
            y=fy,
            z=fz,
            mode='lines',
            line=dict(color='gray', width=2),
            name=f'{cam["name"]} frustum',
            showlegend=False
        ))
        
        # Add coordinate axes
        axis_traces = create_coordinate_axes(transform, scale)
        for trace in axis_traces:
            fig.add_trace(go.Scatter3d(
                x=trace['x'],
                y=trace['y'],
                z=trace['z'],
                mode='lines',
                line=dict(color=trace['color'], width=3),
                name=f'{cam["name"]} {trace["color"]} axis',
                showlegend=False
            ))
    
    # Add camera connection line
    positions = np.array(positions)
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='lines',
        line=dict(color='black', width=4),
        name='Camera Line'
    ))
    
    # Add ground plane grid
    max_range = np.max(np.abs(positions)) + scale
    x = np.linspace(-max_range, max_range, 10)
    z = np.linspace(-max_range, max_range, 10)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)
    
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=[[0, 'gray'], [1, 'gray']],
        showscale=False,
        opacity=0.3,
        name='Ground Plane'
    ))
    
    # Calculate camera metrics
    camera_line = positions[-1] - positions[0]
    camera_separation = np.linalg.norm(camera_line)
    
    # Calculate axis ranges to be equal
    max_range = np.max(np.abs(positions)) + scale
    range_dict = [-max_range, max_range]

    # Update layout
    ref_cam_name = transformed_data[reference_camera]['name']
    fig.update_layout(
        title=f'Camera Positions and Orientations ({ref_cam_name} at origin)',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            xaxis=dict(range=range_dict),
            yaxis=dict(range=range_dict),
            zaxis=dict(range=range_dict),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        annotations=[
            dict(
                x=0.02,
                y=0.02,
                xref='paper',
                yref='paper',
                text=f'Camera Separation: {camera_separation:.1f} mm',
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )
    
    return fig

def main():
    if len(sys.argv) < 2:
        print("Usage: python camera_viz.py path/to/calibration.toml [reference_camera]")
        sys.exit(1)
        
    toml_path = sys.argv[1]
    reference_camera = sys.argv[2] if len(sys.argv) > 2 else 'cam_0'
    
    # Show the visualization
    fig = plot_cameras(toml_path, reference_camera)
    fig.show()

if __name__ == "__main__":
    main()