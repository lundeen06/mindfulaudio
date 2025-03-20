import numpy as np
import plotly.graph_objects as go
import tomli
import sys
from scipy.spatial.transform import Rotation
import dash
from dash import html, dcc
from dash.dependencies import Input, Output

def get_transform_matrix(rvec, tvec):
    """Convert rotation vector and translation to 4x4 transform matrix"""
    R = Rotation.from_rotvec(rvec).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T

def translate_cameras_to_reference(data, reference_camera='cam_0'):
    """Translate and rotate all cameras so reference camera is at origin with no rotation"""
    ref_cam = data[reference_camera]
    ref_transform = get_transform_matrix(
        np.array(ref_cam['rotation']),
        np.array(ref_cam['translation'])
    )
    
    ref_transform_inv = np.linalg.inv(ref_transform)
    
    new_data = {}
    for key, cam in data.items():
        if not key.startswith('cam_'):
            continue
            
        orig_transform = get_transform_matrix(
            np.array(cam['rotation']),
            np.array(cam['translation'])
        )
        
        new_transform = ref_transform_inv @ orig_transform
        
        new_rot = Rotation.from_matrix(new_transform[:3, :3])
        new_trans = new_transform[:3, 3]
        
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
    
    base_points = np.array([
        [-aspect_ratio, -1, 2],
        [aspect_ratio, -1, 2],
        [aspect_ratio, 1, 2],
        [-aspect_ratio, 1, 2],
        [0, 0, 0]
    ]) * scale * 0.5
    
    points_h = np.hstack([base_points, np.ones((base_points.shape[0], 1))])
    transformed_points = (transform_matrix @ points_h.T).T[:, :3]
    center = transformed_points[-1]
    points = transformed_points[:-1]
    
    x, y, z = [], [], []
    
    for point in points:
        x.extend([center[0], point[0], None])
        y.extend([center[1], point[1], None])
        z.extend([center[2], point[2], None])
    
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
    
    origin = np.array([0, 0, 0, 1])
    for i in range(3):
        direction = np.zeros(4)
        direction[i] = scale
        direction[3] = 1
        
        start = (transform_matrix @ origin)[:3]
        end = (transform_matrix @ direction)[:3]
        
        traces.append({
            'x': [start[0], end[0]],
            'y': [start[1], end[1]],
            'z': [start[2], end[2]],
            'color': axis_colors[i]
        })
        
    return traces

def apply_ground_plane_transform(data, tx, ty, tz, rx, ry, rz):
    """Apply ground plane transformation to all cameras"""
    # Define ground plane transform matrix
    ground_rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    ground_transform = np.eye(4)
    ground_transform[:3, :3] = ground_rotation.as_matrix()
    ground_transform[:3, 3] = [tx, ty, tz]
    
    # Apply transform to all cameras
    transformed_data = {}
    for key, cam in data.items():
        if not key.startswith('cam_'):
            continue
            
        orig_transform = get_transform_matrix(
            np.array(cam['rotation']),
            np.array(cam['translation'])
        )
        
        new_transform = ground_transform @ orig_transform
        
        new_rot = Rotation.from_matrix(new_transform[:3, :3])
        new_trans = new_transform[:3, 3]
        
        transformed_data[key] = {
            'name': cam['name'],
            'size': cam['size'],
            'rotation': new_rot.as_rotvec().tolist(),
            'translation': new_trans.tolist()
        }
    
    return transformed_data

def create_dash_app(toml_path, reference_camera='cam_0'):
    # Load camera data
    with open(toml_path, 'rb') as f:
        original_data = tomli.load(f)
    
    app = dash.Dash(__name__)
    
    # Create a more intuitive layout with sliders
    app.layout = html.Div([
        html.H1('VHIL Camera Calibration Visualization', 
                style={'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div([
            # Control panel
            html.Div([
                html.H3('Ground Plane Controls'),
                
                # Translation sliders
                html.Div([
                    html.H4('Translation (mm)', style={'marginBottom': '10px'}),
                    html.Label('X Translation'),
                    dcc.Slider(id='tx', min=-500, max=500, value=0, 
                             marks={i: str(i) for i in range(-500, 501, 100)},
                             updatemode='drag'),
                    
                    html.Label('Y Translation'),
                    dcc.Slider(id='ty', min=-500, max=500, value=0,
                             marks={i: str(i) for i in range(-500, 501, 100)},
                             updatemode='drag'),
                    
                    html.Label('Z Translation'),
                    dcc.Slider(id='tz', min=-500, max=500, value=0,
                             marks={i: str(i) for i in range(-500, 501, 100)},
                             updatemode='drag'),
                ], style={'marginBottom': '30px'}),
                
                # Rotation sliders
                html.Div([
                    html.H4('Rotation (degrees)', style={'marginBottom': '10px'}),
                    html.Label('X Rotation (Roll)'),
                    dcc.Slider(id='rx', min=-180, max=180, value=0,
                             marks={i: str(i) for i in range(-180, 181, 45)},
                             updatemode='drag'),
                    
                    html.Label('Y Rotation (Pitch)'),
                    dcc.Slider(id='ry', min=-180, max=180, value=0,
                             marks={i: str(i) for i in range(-180, 181, 45)},
                             updatemode='drag'),
                    
                    html.Label('Z Rotation (Yaw)'),
                    dcc.Slider(id='rz', min=-180, max=180, value=0,
                             marks={i: str(i) for i in range(-180, 181, 45)},
                             updatemode='drag'),
                ]),
                
                # Reset button
                html.Button('Reset Transform', id='reset-button', 
                          style={'marginTop': '20px', 'width': '100%'}),
                
                # Current values display
                html.Div(id='current-values', 
                        style={'marginTop': '20px', 'padding': '10px', 
                               'backgroundColor': '#f0f0f0', 'borderRadius': '5px'}),
                
            ], style={'width': '400px', 'padding': '20px', 'backgroundColor': '#ffffff',
                     'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'borderRadius': '10px'}),
            
            # 3D visualization
            dcc.Graph(id='camera-plot', 
                     style={'height': '800px', 'width': '800px', 'backgroundColor': '#ffffff',
                            'boxShadow': '0px 0px 10px rgba(0,0,0,0.1)', 'borderRadius': '10px'}),
            
        ], style={'display': 'flex', 'gap': '20px', 'justifyContent': 'center',
                  'margin': '20px', 'backgroundColor': '#f8f9fa', 'padding': '20px'}),
    ])
    
    @app.callback(
        [Output('camera-plot', 'figure'),
         Output('current-values', 'children')],
        [Input('tx', 'value'),
         Input('ty', 'value'),
         Input('tz', 'value'),
         Input('rx', 'value'),
         Input('ry', 'value'),
         Input('rz', 'value'),
         Input('reset-button', 'n_clicks')]
    )
    def update_plot(tx, ty, tz, rx, ry, rz, reset_clicks):
        # Reset values if reset button is clicked
        ctx = dash.callback_context
        if ctx.triggered and 'reset-button' in ctx.triggered[0]['prop_id']:
            tx, ty, tz, rx, ry, rz = 0, 0, 0, 0, 0, 0
        
        # First transform relative to reference camera
        ref_transformed = translate_cameras_to_reference(original_data, reference_camera)
        
        # Then apply ground plane transform
        transformed_data = apply_ground_plane_transform(
            ref_transformed,
            tx or 0, ty or 0, tz or 0,
            rx or 0, ry or 0, rz or 0
        )
        
        # Create figure
        fig = go.Figure()
        
        # Plot cameras
        positions = []
        for key, cam in transformed_data.items():
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
            fx, fy, fz = create_camera_frustum(transform, cam['size'])
            fig.add_trace(go.Scatter3d(
                x=fx, y=fy, z=fz,
                mode='lines',
                line=dict(color='gray', width=2),
                name=f'{cam["name"]} frustum',
                showlegend=False
            ))
            
            # Add coordinate axes
            axis_traces = create_coordinate_axes(transform)
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
        
        # Add transformed ground plane grid
        max_range = np.max(np.abs(positions)) + 100
        x = np.linspace(-max_range, max_range, 10)
        z = np.linspace(-max_range, max_range, 10)
        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X)
        
        # Create ground plane transform
        ground_rotation = Rotation.from_euler('xyz', [rx or 0, ry or 0, rz or 0], degrees=True)
        ground_transform = np.eye(4)
        ground_transform[:3, :3] = ground_rotation.as_matrix()
        ground_transform[:3, 3] = [tx or 0, ty or 0, tz or 0]
        
        # Transform ground plane points
        points = np.stack([X, Y, Z, np.ones_like(X)], axis=-1)
        transformed_points = np.einsum('ij,klj->kli', ground_transform, points)[..., :3]
        
        fig.add_trace(go.Surface(
            x=transformed_points[..., 0],
            y=transformed_points[..., 1],
            z=transformed_points[..., 2],
            colorscale=[[0, 'gray'], [1, 'gray']],
            showscale=False,
            opacity=0.3,
            name='Ground Plane'
        ))
        
        # Calculate camera metrics
        camera_line = positions[-1] - positions[0]
        camera_separation = np.linalg.norm(camera_line)
        
        # Calculate axis ranges to be equal
        max_range = np.max(np.abs(positions)) + 100
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
        
        # Create current values display
        current_values = html.Div([
            html.P(f'Translation: X={tx:.1f}, Y={ty:.1f}, Z={tz:.1f} mm'),
            html.P(f'Rotation: Roll={rx:.1f}°, Pitch={ry:.1f}°, Yaw={rz:.1f}°')
        ])
        
        return fig, current_values

    @app.callback(
        [Output('tx', 'value'),
         Output('ty', 'value'),
         Output('tz', 'value'),
         Output('rx', 'value'),
         Output('ry', 'value'),
         Output('rz', 'value')],
        [Input('reset-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def reset_values(n_clicks):
        return 0, 0, 0, 0, 0, 0
    
    return app

def main():
    if len(sys.argv) < 2:
        print("Usage: python camera_viz.py path/to/calibration.toml [reference_camera]")
        sys.exit(1)
        
    toml_path = sys.argv[1]
    reference_camera = sys.argv[2] if len(sys.argv) > 2 else 'cam_0'
    
    app = create_dash_app(toml_path, reference_camera)
    app.run_server(debug=True)

if __name__ == "__main__":
    main()