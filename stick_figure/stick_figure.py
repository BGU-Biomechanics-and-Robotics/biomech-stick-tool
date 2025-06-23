import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox

squatting_pose = {'neck': 90, 'torso': 90, 'upper arm': 170, 'forearm': 180, 
                  'hand': 180, 'thigh': 0, 'shank': -90, 'toe': 0}

swimming_pose = {'neck': 170, 'torso': 10, 'upper arm': -45, 'forearm': -45,
                 'hand': -90, 'thigh': -10, 'shank': -10, 'toe': -90}

class SagittalStickFigure:
    def __init__(self):
        self.lengths = {'head': 12, 'neck': 15, 'torso': 50, 'upper arm': 30, 
                       'forearm': 25, 'hand': 15, 'thigh': 45, 'shank': 45, 'toe': 8}
        self.angles = swimming_pose.copy()
        
        # Kinematic chain: segment: [proximal_joint, distal_joint, [h_dir, v_dir]]
        self.segments = {
            'neck': ["shoulder", "neck", [1, 1]], 'torso': ["hip", "shoulder", [-1, 1]],
            'upper arm': ["shoulder", "elbow", [-1, 1]], 'forearm': ["elbow", "wrist", [-1, 1]],
            'hand': ["wrist", "hand", [-1, 1]], 'thigh': ["hip", "knee", [1, 1]],
            'shank': ["knee", "ankle", [1, 1]], 'toe': ["ankle", "toe", [1, 1]]
        }
        
        self.dirs = {s: info[2][:] for s, info in self.segments.items()}
        self.axis_length, self.show_axis = 10, True
        self.canvas_width, self.canvas_height = 200, 200
        self.hip_base_x, self.hip_base_y = 100, 100
        self.rotation = 0
        
        self.fig, (self.ax_main, self.ax_control) = plt.subplots(1, 2, figsize=(16, 8))
        self.ax_main.set_aspect('equal')
        self.ax_control.axis('off')
        self.bottons = {}
        self.create_controls()
        self.update_display()
    
    def create_controls(self):
        # Buttons
        x, w, p = 0.5, 0.08, 0.01
        buttons = [('Rotate 90°', self.rotate), ('Mirror', self.mirror), ('Toggle Axis', self.toggle_axis)]
        for i, (label, func) in enumerate(buttons):
            self.bottons[label] = Button(plt.axes([x + i*(w+p), 0.9, w, 0.04]), label)
            self.bottons[label].on_clicked(func)
        
        # Controls for each segment
        self.sliders, self.textboxes, self.radios_h, self.radios_v = {}, {}, {}, {}
        y = 0.8
        for name in self.angles:
            # Slider
            ax_slider = plt.axes([0.55, y, 0.2, 0.03])
            self.sliders[name] = Slider(ax_slider, name, -180, 180, valinit=self.angles[name])
            self.sliders[name].on_changed(lambda val, n=name: self.update_angle_from_slider(n, val))
            
            # Text input
            ax_text = plt.axes([0.76, y, 0.05, 0.03])
            self.textboxes[name] = TextBox(ax_text, '', initial=str(int(self.angles[name])))
            self.textboxes[name].on_submit(lambda val, n=name: self.update_angle_from_text(n, val))
            
            # Direction radios
            ax_h = plt.axes([0.82, y, 0.05, 0.03])
            self.radios_h[name] = RadioButtons(ax_h, ['R', 'L'], active=0 if self.dirs[name][0] == 1 else 1)
            self.radios_h[name].on_clicked(lambda val, n=name: self.update_dir_h(n, val))
            
            ax_v = plt.axes([0.88, y, 0.05, 0.03])
            self.radios_v[name] = RadioButtons(ax_v, ['U', 'D'], active=0 if self.dirs[name][1] == 1 else 1)
            self.radios_v[name].on_clicked(lambda val, n=name: self.update_dir_v(n, val))
            
            y -= 0.08
    
    def compute_distal_pos(self, prox_pos, angle, direction, length):
        rad = np.deg2rad(angle)
        dx = np.cos(rad) * (-direction[0]) * length
        dy = np.sin(rad) * direction[1] * length
        return prox_pos + np.array([dx, dy])
    
    def normalize_angle(self, angle):
        while angle > 180: angle -= 360
        while angle <= -180: angle += 360
        return angle
    
    def update_angle_from_slider(self, name, val):
        self.angles[name] = val
        self.textboxes[name].set_val(str(int(val)))
        self.update_display()
    
    def update_angle_from_text(self, name, val):
        try:
            angle = max(-180, min(180, float(val)))
            self.angles[name] = angle
            self.sliders[name].set_val(angle)
            self.update_display()
        except ValueError:
            self.textboxes[name].set_val(str(int(self.angles[name])))
    
    def update_dir_h(self, name, val):
        new_dir = 1 if val == 'R' else -1
        if self.dirs[name][0] != new_dir:
            self.dirs[name][0] = new_dir
            self.angles[name] = self.normalize_angle(180 - self.angles[name])
            self.sliders[name].set_val(self.angles[name])
            self.textboxes[name].set_val(str(int(self.angles[name])))
            self.update_display()
    
    def update_dir_v(self, name, val):
        new_dir = 1 if val == 'U' else -1
        if self.dirs[name][1] != new_dir:
            self.dirs[name][1] = new_dir
            self.angles[name] = self.normalize_angle(-self.angles[name])
            self.sliders[name].set_val(self.angles[name])
            self.textboxes[name].set_val(str(int(self.angles[name])))
            self.update_display()
    
    def rotate(self, event):
        print("Rotating figure by 90 degrees")
        self.rotation = (self.rotation + 90) % 360
        for name in self.angles:
            # Compute effective angle
            eff_angle = self.angles[name]
            if self.dirs[name][0] == 1: eff_angle = self.angles[name]
            else: eff_angle = 180 - self.angles[name]
            if self.dirs[name][1] == -1: eff_angle = -eff_angle
            
            # Rotate and convert back
            new_eff = eff_angle + 90
            if self.dirs[name][1] == -1: new_eff = -new_eff
            if self.dirs[name][0] == -1: new_eff = 180 - new_eff
            
            self.angles[name] = self.normalize_angle(new_eff)
            self.sliders[name].set_val(self.angles[name])
            self.textboxes[name].set_val(str(int(self.angles[name])))
        self.update_display()
    
    def mirror(self, event):
        for name in self.angles:
            self.dirs[name][0] = -self.dirs[name][0]
            self.radios_h[name].set_active(0 if self.dirs[name][0] == 1 else 1)
        self.update_display()
    
    def toggle_axis(self, event):
        self.show_axis = not self.show_axis
        self.update_display()
        
    def compute_orientation(self, prox_pos, distal_pos, direction):
        """
        Compute segment orientation angle from joint positions, returning the same angle
        used in compute_distal_pos.
        Args:
            prox_pos: (x,y) proximal joint position
            distal_pos: (x,y) distal joint position  
            direction: [horizontal_dir, vertical_dir] where horizontal: 1 for right (R), -1 for left (L)
                      and vertical: 1 for up (U), -1 for down (D)
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
        return self.normalize_angle(angle_deg)
    
    def compute_joints(self):
        joints = {'hip': np.array([self.hip_base_x, self.hip_base_y])}
        
        for name in ['torso', 'neck', 'upper arm', 'forearm', 'hand', 'thigh', 'shank', 'toe']:
            if name in self.segments:
                prox_joint, distal_joint, _ = self.segments[name]
                joints[distal_joint] = self.compute_distal_pos(
                    joints[prox_joint], self.angles[name], self.dirs[name], self.lengths[name])
                computed_angle = self.compute_orientation(prox_pos=joints[prox_joint],distal_pos=joints[distal_joint],
                                                        direction=self.dirs[name]
                                        )
                expected_angle =  self.angles[name]
                if not np.isclose(computed_angle, expected_angle, atol=1e-2):
                    print(f"Warning: Computed angle {computed_angle:.2f} for segment '{name}' "
                          f"does not match expected angle {expected_angle:.2f}.")
        
        # Head
        if 'neck' in joints:
            joints['head'] = self.compute_distal_pos(
                joints['neck'], self.angles['neck'], self.dirs['neck'], self.lengths['head'])
        
        return joints
    
    def update_display(self):
        self.ax_main.clear()
        joints = self.compute_joints()
        
        self.ax_main.set_xlim(0, self.canvas_width)
        self.ax_main.set_ylim(0, self.canvas_height)
        
        # Draw segments
        for name, (prox, distal, _) in self.segments.items():
            if name != 'neck' and distal in joints and prox in joints:
                p1, p2 = joints[prox], joints[distal]
                self.ax_main.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-o', linewidth=2, markersize=4)
        
        # Draw neck
        if 'shoulder' in joints and 'neck' in joints:
            p1, p2 = joints['shoulder'], joints['neck']
            self.ax_main.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-o', linewidth=2, markersize=4)
        
        # Draw head circle
        if 'head' in joints:
            circle = plt.Circle(joints['head'], self.lengths['head'], fill=False, linewidth=2)
            self.ax_main.add_patch(circle)
        
        # Draw axis indicators
        if self.show_axis:
            for name, (_, distal, _) in self.segments.items():
                if distal in joints:
                    pos = joints[distal]
                    # Horizontal axis
                    dx = self.axis_length if self.dirs[name][0] == 1 else -self.axis_length
                    self.ax_main.plot([pos[0], pos[0] + dx], [pos[1], pos[1]], 'r-', linewidth=1, alpha=0.7)
                    # Vertical axis  
                    dy = self.axis_length if self.dirs[name][1] == 1 else -self.axis_length
                    self.ax_main.plot([pos[0], pos[0]], [pos[1], pos[1] + dy], 'g-', linewidth=1, alpha=0.7)
        
        # Coordinate system
        self.ax_main.plot(0, 0, 'ko', markersize=8)
        self.ax_main.plot([0, 20], [0, 0], 'k--', alpha=0.5, linewidth=1)
        self.ax_main.plot([0, 0], [0, 20], 'k--', alpha=0.5, linewidth=1)
        self.ax_main.text(22, -3, 'X+', fontsize=8)
        self.ax_main.text(-3, 22, 'Y+', fontsize=8)
        
        hip = joints['hip']
        self.ax_main.plot(hip[0], hip[1], 'ro', markersize=6, alpha=0.8)
        
        title = f'Bottom-Left Coordinate System (Rotation: {self.rotation}°) - Axis: {"ON" if self.show_axis else "OFF"}'
        self.ax_main.set_title(title)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_xlabel('X (Bottom-Left Origin)')
        self.ax_main.set_ylabel('Y (Bottom-Left Origin)')
        
        self.fig.canvas.draw()
    
    def show(self):
        plt.show()

if __name__ == '__main__':
    figure = SagittalStickFigure()
    figure.show()