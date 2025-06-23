import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox

squatting_pose = {'neck': 90, 'torso': 90, 'upper arm': 170, 'forearm': 180, 
                      'hand': 180, 'thigh': 0, 'shank': -90, 'toe': 0}

swimming_pose = {'neck': 170, 'torso': 10, 'upper arm': -45, 'forearm': -45,
                      'hand': -90, 'thigh': -10, 'shank': -10, 'toe': -90}

class SagittalStickFigure:
    def __init__(self):
        # Segment lengths
        self.lengths = {'head': 12, 'neck': 15, 'torso': 50, 'upper arm': 30, 
                       'forearm': 25, 'hand': 15, 'thigh': 45, 'shank': 45, 'toe': 8}
        
        # Initial angles (swimming pose)
        self.angles = swimming_pose.copy()
        
        # Kinematic chain definition: segment_name: [proximal_joint, distal_joint, [horizontal_dir, vertical_dir]]
        # Horizontal direction: 1=right, -1=left
        # Vertical direction: 1=up, -1=down
        self.segments = {
            'neck': ["shoulder", "neck", [1, 1]],
            'torso': ["hip", "shoulder", [-1, 1]],
            'upper arm': ["shoulder", "elbow", [-1, 1]],
            'forearm': ["elbow", "wrist", [-1, 1]],
            'hand': ["wrist", "hand", [-1, 1]],
            'thigh': ["hip", "knee", [1, 1]],
            'shank': ["knee", "ankle", [1, 1]],
            'toe': ["ankle", "toe", [1, 1]]
        }
        
        # Extract directions from segments definition
        self.dirs = {segment: [info[2][0], info[2][1]] for segment, info in self.segments.items()}
        
        # Special handling for head segment (circle, not line)
        self.head_segment = {'head': ["neck", "head", [1, 1]]}
        
        # Axis indicator settings
        self.axis_length = 10  # Length of axis indicator lines
        self.show_axis = True  # Toggle for showing axis indicators
        
        # Bottom-left coordinate system settings
        self.canvas_width = 200   # Canvas width
        self.canvas_height = 200  # Canvas height
        self.hip_base_x = 100     # Hip x position (center horizontally)
        self.hip_base_y = 100     # Hip y position (center vertically)
        
        self.rotation = 0  # This is now just for display purposes
        self.setup_figure()
        self.create_controls()
        self.update_display()
    
    def setup_figure(self):
        self.fig, (self.ax_main, self.ax_control) = plt.subplots(1, 2, figsize=(16, 8))
        self.ax_main.set_aspect('equal')
        self.ax_control.axis('off')
    
    def _create_button(self, x_pos, label, callback):
        """Helper function to create a button"""
        ax = plt.axes([x_pos, 0.9, 0.08, 0.04])
        btn = Button(ax, label)
        btn.on_clicked(callback)
        return btn
    
    def _create_control_row(self, name, y_pos):
        """Helper function to create a complete control row (slider, textbox, radio buttons)"""
        # Angle slider (-180 to 180)
        ax_slider = plt.axes([0.55, y_pos, 0.2, 0.03])
        slider = Slider(ax_slider, name, -180, 180, valinit=self.angles[name])
        slider.on_changed(lambda val, n=name: self.update_angle_from_slider(n, val))
        
        # Angle text input
        ax_text = plt.axes([0.76, y_pos, 0.05, 0.03])
        textbox = TextBox(ax_text, '', initial=str(int(self.angles[name])))
        textbox.on_submit(lambda val, n=name: self.update_angle_from_text(n, val))
        
        # Horizontal direction radio button
        ax_radio_h = plt.axes([0.82, y_pos, 0.05, 0.03])
        radio_h = RadioButtons(ax_radio_h, ['R', 'L'], active=0 if self.dirs[name][0] == 1 else 1)
        radio_h.on_clicked(lambda val, n=name: self.update_dir_h(n, val))
        
        # Vertical direction radio button
        ax_radio_v = plt.axes([0.88, y_pos, 0.05, 0.03])
        radio_v = RadioButtons(ax_radio_v, ['U', 'D'], active=0 if self.dirs[name][1] == 1 else 1)
        radio_v.on_clicked(lambda val, n=name: self.update_dir_v(n, val))
        
        return slider, textbox, radio_h, radio_v
        
    def create_controls(self):
        x_start = 0.5  # Scale factor for control axes
        w = 0.08 # Width of control buttons
        p = 0.01 # Padding between controls
        
        # Create buttons using helper function
        self.btn_rotate = self._create_button(x_start, 'Rotate 90°', self.rotate)
        self.btn_mirror = self._create_button(x_start + (w+p)*1, 'Mirror', self.mirror)
        self.btn_axis = self._create_button(x_start + (w+p)*2, 'Toggle Axis', self.toggle_axis)
        
        # Sliders, text boxes, and radio buttons
        self.sliders, self.textboxes, self.radios_h, self.radios_v = {}, {}, {}, {}
        y_pos = 0.8
        
        for name in self.angles:
            slider, textbox, radio_h, radio_v = self._create_control_row(name, y_pos)
            self.sliders[name] = slider
            self.textboxes[name] = textbox
            self.radios_h[name] = radio_h
            self.radios_v[name] = radio_v
            y_pos -= 0.08
    
    def compute_distal_pos(self, prox_pos, angle, direction, length):
        """Core function to compute distal joint position from proximal joint"""
        rad = np.deg2rad(angle)
        h_dir_mult = -direction[0]  # direction[0] is now 1 or -1
        v_dir_mult = direction[1]   # direction[1] is now 1 or -1
        dx = np.cos(rad) * h_dir_mult * length
        dy = np.sin(rad) * v_dir_mult * length
        return prox_pos + np.array([dx, dy])
    
    def compute_orientation(self, prox_pos, distal_pos, direction, side=None):
        """
        Compute segment orientation angle from joint positions, returning the same angle
        used in compute_distal_pos.
        Args:
            prox_pos: (x,y) proximal joint position
            distal_pos: (x,y) distal joint position  
            direction: [horizontal_dir, vertical_dir] where horizontal: 1 for right (R), -1 for left (L)
                      and vertical: 1 for up (U), -1 for down (D)
            side: unused
        Returns:
            angle in degrees, matching the angle passed to compute_distal_pos().
        """
        dx = distal_pos[0] - prox_pos[0]
        dy = distal_pos[1] - prox_pos[1]
        
        # Compute raw geometric angle
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.rad2deg(angle_rad)
        
        # Inverse of compute_distal_pos's use of `direction`:
        # If horizontal direction=1 ("R"), compute_distal_pos did dx = cos(angle)*(-1)*length,
        # so raw angle is (180° - stored_angle). To recover stored_angle, do: stored = 180 - raw.
        # If horizontal direction=-1 ("L"), compute_distal_pos did dx = cos(angle)*(+1)*length,
        # so raw angle already equals stored_angle.
        if direction[0] == 1:
            angle_deg = 180 - angle_deg
        
        # Handle vertical direction
        if direction[1] == -1:  # Down direction
            angle_deg = -angle_deg
            
        # Normalize to [-180, 180]
        return self._normalize_angle(angle_deg)

    def _normalize_angle(self, angle):
        """Helper function to normalize angle to [-180, 180] range"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def compute_axis_lines(self, joint_pos, segment_angle, direction):
        """
        Compute axis line endpoints for a segment at its distal joint
        Args:
            joint_pos: (x,y) position of the distal joint
            segment_angle: angle of the segment in degrees
            direction: [horizontal_dir, vertical_dir] where horizontal: 1 for right, -1 for left
        Returns:
            tuple of (horizontal_line, vertical_line) where each is (start_point, end_point)
        """
        # Horizontal axis (0 degrees)
        horizontal_rad = np.deg2rad(0)
        dx_h = np.cos(horizontal_rad) * self.axis_length
        dy_h = np.sin(horizontal_rad) * self.axis_length
        
        # Adjust horizontal direction based on segment direction
        if direction[0] == -1:  # Left side
            dx_h = -dx_h
        
        horizontal_start = joint_pos
        horizontal_end = joint_pos + np.array([dx_h, dy_h])
        
        # Vertical axis (90 degrees)
        vertical_rad = np.deg2rad(90)
        dx_v = np.cos(vertical_rad) * self.axis_length
        dy_v = np.sin(vertical_rad) * self.axis_length
        
        # Adjust vertical direction based on segment direction
        if direction[1] == -1:  # Down direction
            dy_v = -dy_v
        
        vertical_start = joint_pos
        vertical_end = joint_pos + np.array([dx_v, dy_v])
        
        return (horizontal_start, horizontal_end), (vertical_start, vertical_end)

    def toggle_axis(self, event):
        """Toggle the display of axis indicators"""
        self.show_axis = not self.show_axis
        self.update_display()
    
    def _update_ui_controls(self, name, angle):
        """Helper function to update slider and textbox for a given segment"""
        self.sliders[name].set_val(angle)
        self.textboxes[name].set_val(str(int(angle)))
    
    def update_angle_from_slider(self, name, val):
        self.angles[name] = val
        self.textboxes[name].set_val(str(int(val)))
        self.update_display()
    
    def update_angle_from_text(self, name, val):
        try:
            angle = float(val)
            angle = max(-180, min(180, angle))  # Clamp to range
            self.angles[name] = angle
            self.sliders[name].set_val(angle)
            self.update_display()
        except ValueError:
            # Reset to current value if invalid input
            self.textboxes[name].set_val(str(int(self.angles[name])))
    
    def _flip_angle_and_update_ui(self, name, flip_func):
        """Helper function for direction changes that flip angles"""
        # Flip the angle to maintain same visual orientation
        new_angle = flip_func(self.angles[name])
        new_angle = self._normalize_angle(new_angle)
        
        self.angles[name] = new_angle
        self._update_ui_controls(name, new_angle)
        self.update_display()
    
    def update_dir_h(self, name, val):
        old_dir = self.dirs[name][0]
        new_dir = 1 if val == 'R' else -1
        
        if old_dir != new_dir:
            self.dirs[name][0] = new_dir
            self._flip_angle_and_update_ui(name, lambda angle: 180 - angle)
    
    def update_dir_v(self, name, val):
        old_dir = self.dirs[name][1]
        new_dir = 1 if val == 'U' else -1
        
        if old_dir != new_dir:
            self.dirs[name][1] = new_dir
            self._flip_angle_and_update_ui(name, lambda angle: -angle)
    
    def _compute_rotated_angle(self, segment_name, rotation_angle):
        """Helper function to compute new angle after rotation"""
        current_angle = self.angles[segment_name]
        current_dir = self.dirs[segment_name]
        
        # Calculate the effective geometric angle considering current horizontal and vertical directions
        effective_angle = current_angle
        if current_dir[0] == 1:  # Right direction
            effective_angle = current_angle
        else:  # Left direction  
            effective_angle = 180 - current_angle
            
        # Account for vertical direction
        if current_dir[1] == -1:  # Down direction
            effective_angle = -effective_angle
        
        # Apply rotation to the effective angle
        rotated_effective_angle = effective_angle + rotation_angle
        
        # Convert back to the angle representation for the preserved directions
        new_angle = rotated_effective_angle
        
        # Account for vertical direction (reverse the operation)
        if current_dir[1] == -1:  # Down direction
            new_angle = -new_angle
            
        # Account for horizontal direction (reverse the operation)
        if current_dir[0] == -1:  # Left direction
            new_angle = 180 - new_angle
        
        return self._normalize_angle(new_angle)
            
    def rotate(self, event):
        """Rotate the figure by updating the segment angles while preserving user-defined directions"""
        rotation_angle = 90  # degrees
        
        # Update the cumulative rotation for display purposes
        self.rotation = (self.rotation + rotation_angle) % 360
        
        # For each segment, calculate the new angle that achieves the visual rotation
        # while keeping the direction unchanged
        for segment_name in self.angles:
            self.angles[segment_name] = self._compute_rotated_angle(segment_name, rotation_angle)
        
        # Update UI controls - only sliders and textboxes, NOT radio buttons
        for segment_name in self.angles:
            self._update_ui_controls(segment_name, self.angles[segment_name])
            # Note: radio buttons (directions) are NOT updated - they stay as user set them
        
        self.update_display()
            
    def mirror(self, event):
        """Mirror the figure by flipping angles and horizontal directions"""
        # Flip horizontal directions only
        for segment_name in self.angles:
            self.dirs[segment_name][0] = -self.dirs[segment_name][0]
        
        # Update all UI controls to reflect the new angles and directions
        for segment_name in self.angles:
            self._update_ui_controls(segment_name, self.angles[segment_name])
            self.radios_h[segment_name].set_active(0 if self.dirs[segment_name][0] == 1 else 1)
            # Vertical directions remain unchanged
        
        self.update_display()
    
    def compute_joints(self):
        """Compute all joint positions based on segment definitions using bottom-left coordinate system"""
        joints = {}
        
        # Initialize root joint (hip) at specified position within canvas
        joints['hip'] = np.array([self.hip_base_x, self.hip_base_y])
        
        # Process segments in dependency order
        # We need to ensure proximal joints are computed before distal joints
        segment_order = ['torso', 'neck', 'upper arm', 'forearm', 'hand', 'thigh', 'shank', 'toe']
        
        for segment_name in segment_order:
            if segment_name in self.segments:
                proximal_joint, distal_joint, _ = self.segments[segment_name]
                
                # Get proximal joint position (must already exist)
                if proximal_joint not in joints:
                    raise ValueError(f"Proximal joint '{proximal_joint}' not found for segment '{segment_name}'")
                
                prox_pos = joints[proximal_joint]
                
                # Compute distal joint position
                joints[distal_joint] = self.compute_distal_pos(
                    prox_pos, 
                    self.angles[segment_name], 
                    self.dirs[segment_name], 
                    self.lengths[segment_name]
                )
                computed_angle = self.compute_orientation(
                    prox_pos, 
                    joints[distal_joint], 
                    self.dirs[segment_name], 
                    segment_name
                )
                expected_angle = self.angles[segment_name]
                if not np.isclose(computed_angle, expected_angle, atol=1e-2):
                    print(f"Warning: Computed angle {computed_angle:.2f} for segment '{segment_name}' "
                          f"does not match expected angle {expected_angle:.2f}.")
        
        # Handle head separately (it's a circle, not a line segment)
        if 'neck' in joints:
            joints['head'] = self.compute_distal_pos(
                joints['neck'], 
                self.angles['neck'],  # Head uses neck angle
                self.dirs['neck'], 
                self.lengths['head']
            )
        return joints
    
    def get_segment_lines(self):
        """Get line segments for drawing, excluding special cases like head"""
        lines = []
        for segment_name, (proximal, distal, _) in self.segments.items():
            if segment_name != 'neck':  # neck connects to head circle, not a line endpoint
                lines.append((proximal, distal))
        return lines
    
    def _draw_line_segment(self, joints, start, end, style='b-o'):
        """Helper function to draw a line segment between two joints"""
        p1, p2 = joints[start], joints[end]
        self.ax_main.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=2, markersize=4)
    
    def _draw_axis_indicators(self, joints):
        """Helper function to draw all axis indicators (horizontal and vertical)"""
        if not self.show_axis:
            return
            
        for segment_name, (proximal_joint, distal_joint, _) in self.segments.items():
            if distal_joint in joints:
                joint_pos = joints[distal_joint]
                segment_angle = self.angles[segment_name]
                direction = self.dirs[segment_name]
                
                # Compute axis lines (horizontal and vertical)
                horizontal_line, vertical_line = self.compute_axis_lines(joint_pos, segment_angle, direction)
                
                # Draw horizontal line in red with thinner line
                self.ax_main.plot([horizontal_line[0][0], horizontal_line[1][0]], 
                                [horizontal_line[0][1], horizontal_line[1][1]], 
                                'r-', linewidth=1, alpha=0.7)
                
                # Draw vertical line in green with thinner line
                self.ax_main.plot([vertical_line[0][0], vertical_line[1][0]], 
                                [vertical_line[0][1], vertical_line[1][1]], 
                                'g-', linewidth=1, alpha=0.7)
        
        # Add axis indicators for head/neck as well
        if 'head' in joints:
            head_pos = joints['head']
            neck_angle = self.angles['neck']
            neck_direction = self.dirs['neck']
            
            horizontal_line, vertical_line = self.compute_axis_lines(head_pos, neck_angle, neck_direction)
            
            # Draw horizontal line in red
            self.ax_main.plot([horizontal_line[0][0], horizontal_line[1][0]], 
                            [horizontal_line[0][1], horizontal_line[1][1]], 
                            'r-', linewidth=1, alpha=0.7)
            
            # Draw vertical line in green
            self.ax_main.plot([vertical_line[0][0], vertical_line[1][0]], 
                            [vertical_line[0][1], vertical_line[1][1]], 
                            'g-', linewidth=1, alpha=0.7)
    
    def _draw_coordinate_system(self, joints):
        """Helper function to draw coordinate system indicators"""
        # Draw origin marker at (0,0)
        self.ax_main.plot(0, 0, 'ko', markersize=8, label='Origin (0,0)')
        
        # Draw axis lines from origin
        axis_length = 20
        self.ax_main.plot([0, axis_length], [0, 0], 'k--', alpha=0.5, linewidth=1)  # X-axis
        self.ax_main.plot([0, 0], [0, axis_length], 'k--', alpha=0.5, linewidth=1)  # Y-axis
        
        # Add axis labels
        self.ax_main.text(axis_length + 2, -3, 'X+', fontsize=8, ha='left')
        self.ax_main.text(-3, axis_length + 2, 'Y+', fontsize=8, ha='center')
        
        # Show hip position
        hip_pos = joints['hip']
        self.ax_main.plot(hip_pos[0], hip_pos[1], 'ro', markersize=6, alpha=0.8)
        self.ax_main.text(hip_pos[0] + 5, hip_pos[1] + 3, f'Hip ({hip_pos[0]:.0f},{hip_pos[1]:.0f})', 
                         fontsize=8, ha='left')
    
    def update_display(self):
        self.ax_main.clear()
        joints = self.compute_joints()
        
        # Store current joints for dragging functionality
        self.current_joints = joints
        
        # Set up bottom-left coordinate system
        # Force the axes to show (0,0) at bottom-left
        self.ax_main.set_xlim(0, self.canvas_width)
        self.ax_main.set_ylim(0, self.canvas_height)
        
        # Invert y-axis is NOT needed since we want bottom-left as (0,0)
        # matplotlib default has bottom-left as (0,0) which is what we want
        
        # Draw segment lines
        segment_lines = self.get_segment_lines()
        for start, end in segment_lines:
            self._draw_line_segment(joints, start, end)
        
        # Draw neck line (special case)
        if 'shoulder' in joints and 'neck' in joints:
            self._draw_line_segment(joints, 'shoulder', 'neck')
        
        # Draw axis indicators at distal joints
        self._draw_axis_indicators(joints)
        
        # Draw head as circle
        if 'head' in joints:
            head_circle = plt.Circle(joints['head'], self.lengths['head'], fill=False, linewidth=2)
            self.ax_main.add_patch(head_circle)
        
        # Add coordinate system indicators
        self._draw_coordinate_system(joints)
        
        title = f'Bottom-Left Coordinate System (Rotation: {self.rotation}°)'
        if self.show_axis:
            title += ' - Axis: ON'
        else:
            title += ' - Axis: OFF'
        self.ax_main.set_title(title)
        self.ax_main.grid(True, alpha=0.3)
        
        # Add coordinate labels on axes
        self.ax_main.set_xlabel('X (Bottom-Left Origin)')
        self.ax_main.set_ylabel('Y (Bottom-Left Origin)')
        
        self.fig.canvas.draw()
    
    def show(self):
        plt.show()

if __name__ == '__main__':
    figure = SagittalStickFigure()
    figure.show()