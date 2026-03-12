import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from collections import deque


def compute_skeleton_angles(joints, joint_names, segments_map, kinematic_tree, sign_map=None):
    """
    General purpose biomechanical angle extractor.

    For each edge (parent -> child) in the kinematic tree, computes the
    relative rotation between the parent's LCS and the child's LCS,
    decomposed as intrinsic XZY Euler angles:
        channel 0 = Flexion/Extension  (rotation about local X axis)
        channel 1 = Abduction/Adduction (rotation about local Z axis)
        channel 2 = Internal/External Rotation (rotation about local Y / bone axis)

    Convention:  R_rel = R_parent^T @ R_child  = Rx(ch0) @ Rz(ch1) @ Ry(ch2)

    Each joint's LCS is built from the bone direction to a canonical child:
      - Y axis: unit vector from joint toward its canonical child (longitudinal)
      - Z axis: Y × global_lateral  (anterior / posterior)
      - X axis: Y × Z              (medial / lateral)

    Angles are stored at the **child** index.  Root joints get zeros.
    Leaf joints (no children) inherit their parent's LCS orientation.

    Args:
        joints: (N, K, 3) array of joint positions
        kinematic_tree: dict  {joint_name: parent_name}  (root has '' or None)
        joint_names: list of K joint name strings

    Returns:
        (N, K, 3) array of angles in **degrees**
    """
    N, K, _ = joints.shape
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    # ---- Global lateral axis (left hip -> right hip, i.e. +X ~ right) ----
    l_hip, r_hip = name_to_idx['left hip'], name_to_idx['right hip']
    global_lateral = joints[:, l_hip] - joints[:, r_hip]  # (N, 3)
    global_lateral /= (np.linalg.norm(global_lateral, axis=-1, keepdims=True) + 1e-8)

    # ---- Build LCS for every non-leaf joint ----
    lcs_frames = {}  # idx -> (N, 3, 3)  columns = [x, y, z]

    for name in joint_names:
        idx = name_to_idx[name]
        if name in segments_map:
            distal_name, c1 = segments_map[name]
            canon_idx = name_to_idx[distal_name]
            v_long = joints[:, canon_idx] - joints[:, idx]  # parent -> child
            v_long = v_long * c1  # Apply c1: flip Y-axis for limbs so straight = 0°
            y = v_long / (np.linalg.norm(v_long, axis=-1, keepdims=True) + 1e-8)
            z = np.cross(y, global_lateral)
            z /= (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
            x = np.cross(y, z)
            lcs_frames[idx] = np.stack([x, y, z], axis=-1)  # (N, 3, 3)

    # ---- Leaf joints inherit parent's LCS ----
    for name in joint_names:
        idx = name_to_idx[name]
        if idx not in lcs_frames:
            parent_name = kinematic_tree.get(name, '')
            if parent_name and name_to_idx.get(parent_name) in lcs_frames:
                lcs_frames[idx] = lcs_frames[name_to_idx[parent_name]].copy()
            else:
                lcs_frames[idx] = np.tile(np.eye(3), (N, 1, 1))

    # ---- Decompose relative rotation for every edge -> store at child idx ----
    angles = np.zeros((N, K, 3))

    for child_name, parent_name in kinematic_tree.items():
        if not parent_name:
            continue
        c_idx = name_to_idx[child_name]
        p_idx = name_to_idx[parent_name]
        R_child = lcs_frames[c_idx]  # (N, 3, 3)
        R_parent = lcs_frames[p_idx]  # (N, 3, 3)

        # R_rel = R_parent^T @ R_child
        R_rel = np.einsum('nij,nik->njk', R_parent, R_child)  # (N, 3, 3)

        # Intrinsic XZY decomposition of  R = Rx(a0) @ Rz(a1) @ Ry(a2)
        #   R[0,1] = -sin(a1)
        #   R[2,1] =  sin(a0)*cos(a1)    R[1,1] = cos(a0)*cos(a1)
        #   R[0,2] =  sin(a2)*cos(a1)    R[0,0] = cos(a1)*cos(a2)
        angles[:, c_idx, 0] = np.arctan2(R_rel[:, 2, 1], R_rel[:, 1, 1])  # flex
        angles[:, c_idx, 1] = np.arcsin(np.clip(-R_rel[:, 0, 1], -1, 1))  # abd
        angles[:, c_idx, 2] = np.arctan2(R_rel[:, 0, 2], R_rel[:, 0, 0])  # rot
    angles = np.degrees(angles)

    # Anatomical Convention Layer: per-joint sign presentation
    # This single layer handles both bilateral symmetry (abd/rot flip for
    # left-side joints) and joint-specific conventions (e.g. positive knee
    # flexion), all driven by the sign array in joint_defs.
    if sign_map is not None:
        for name, sign in sign_map.items():
            idx = name_to_idx.get(name)
            if idx is not None:
                angles[:, idx, :] *= sign[np.newaxis, :]

    return angles


# ---------------------------------------------------------------------------
#  FK Visualizer
# ---------------------------------------------------------------------------

class Biomech3DVisualizer:
    """Interactive 3D forward-kinematics visualizer.

    Given a kinematic tree and per-edge angles (stored at the child index),
    this class reconstructs the skeleton via FK and overlays the original
    coordinates (when provided) so you can verify correctness.
    """

    def __init__(self, kinematic_tree, joint_names, segments_map, initial_angles=None,
                 bone_lengths=None, original_coords=None, root_lcs=None,
                 bone_offsets=None, sign_map=None):
        self.tree = kinematic_tree
        self.joint_names = joint_names
        self.name_to_idx = {name: i for i, name in enumerate(joint_names)}
        self.original_coords = original_coords  # (K, 3) optional
        self.segments_map = segments_map
        self.sign_map = sign_map or {n: np.ones(3) for n in joint_names}

        # Root orientation (3x3) - needed so FK starts in the same frame as
        # the LCS that compute_skeleton_angles built for the root.
        self.root_lcs = root_lcs if root_lcs is not None else np.eye(3)

        # Bone lengths
        if bone_lengths is not None:
            self.lengths = bone_lengths
        else:
            self.lengths = {child: 0.3 for child in kinematic_tree if kinematic_tree[child]}

        # Bone offsets: the unit direction from parent -> child expressed in
        # the parent's LCS.  For canonical children this is [0,1,0]; for
        # non-canonical children (e.g. hips from pelvis) it differs.
        # If not provided, we assume [0,1,0] for all edges.
        if bone_offsets is not None:
            self.bone_offsets = bone_offsets
        else:
            self.bone_offsets = {child: np.array([0.0, 1.0, 0.0])
                                 for child in kinematic_tree if kinematic_tree[child]}

        # BFS-ordered edges (guarantees parent is processed before child)
        self.ordered_edges = self._bfs_edges()

        # Per-edge angles keyed by *child* name
        self.angles = {name: np.zeros(3) for name in joint_names}

        # --- matplotlib setup ---
        self.fig = plt.figure(figsize=(18, 9))
        self.ax = self.fig.add_subplot(121, projection='3d')
        self.ax_ctrl = self.fig.add_subplot(122)
        self.ax_ctrl.axis('off')
        self.sliders = {}
        self._build_sliders()

        if initial_angles is not None:
            self.set_pose(initial_angles)
        else:
            self._draw()

    def _bfs_edges(self):
        """Return edges in breadth-first order from the root."""
        root = [n for n, p in self.tree.items() if not p][0]
        visited = {root}
        queue = deque([root])
        ordered = []
        while queue:
            curr = queue.popleft()
            for child, parent in self.tree.items():
                if parent == curr and child not in visited:
                    ordered.append((child, parent))
                    visited.add(child)
                    queue.append(child)
        return ordered

    @staticmethod
    def _xzy_rotation(angles_deg):
        """Rx(flex) @ Rz(abd) @ Ry(rot)  -  intrinsic XZY."""
        flex, abd, rot = np.radians(angles_deg)
        cf, sf = np.cos(flex), np.sin(flex)
        ca, sa = np.cos(abd), np.sin(abd)
        cr, sr = np.cos(rot), np.sin(rot)
        Rx = np.array([[1, 0, 0],
                       [0, cf, -sf],
                       [0, sf, cf]])
        Rz = np.array([[ca, -sa, 0],
                       [sa, ca, 0],
                       [0, 0, 1]])
        Ry = np.array([[cr, 0, sr],
                       [0, 1, 0],
                       [-sr, 0, cr]])
        return Rx @ Rz @ Ry

    # ---- UI ----

    def _build_sliders(self):
        y_pos = 0.95
        for child, _parent in self.ordered_edges:
            if y_pos < 0.03:
                break
            for i, label in enumerate(['Flex', 'Abd', 'Rot']):
                ax_s = plt.axes([0.62, y_pos, 0.22, 0.012])
                s = Slider(ax_s, f"{child[:11]} {label}", -180, 180, valinit=0)
                s.on_changed(lambda val, n=child, ax=i: self._on_slider(n, ax, val))
                self.sliders[(child, i)] = s
                y_pos -= 0.019
            y_pos -= 0.012

    def _on_slider(self, name, axis, val):
        self.angles[name][axis] = val
        self._draw()

    def set_pose(self, k_3_angles):
        """Set all angles from an (K, 3) array (indexed by joint)."""
        for i, name in enumerate(self.joint_names):
            self.angles[name] = k_3_angles[i].copy()
            for ax in range(3):
                key = (name, ax)
                if key in self.sliders:
                    self.sliders[key].set_val(self.angles[name][ax])
        self._draw()

    # ---- FK + drawing ----

    def _draw(self):
        self.ax.clear()

        root_name = [n for n, p in self.tree.items() if not p][0]
        pos = {root_name: np.zeros(3)}
        orient = {root_name: self.root_lcs.copy()}

        for child, parent in self.ordered_edges:
            # Reverse the anatomical convention layer to get pure math angles
            fk_angles = self.angles[child].copy()
            fk_angles *= self.sign_map[child]
            R_rel = self._xzy_rotation(fk_angles)
            orient[child] = orient[parent] @ R_rel

            # Bone direction from parent->child, expressed in parent's LCS
            bone_dir_local = self.bone_offsets[child] * self.lengths[child]
            bone = orient[parent] @ bone_dir_local
            pos[child] = pos[parent] + bone

            # bone line
            self.ax.plot([pos[parent][0], pos[child][0]],
                         [pos[parent][1], pos[child][1]],
                         [pos[parent][2], pos[child][2]],
                         'b-o', lw=2, ms=4)

            # JCS arrows at parent
            for k, c in enumerate(['r', 'g', 'b']):
                v = orient[parent][:, k] * 0.06
                self.ax.quiver(pos[parent][0], pos[parent][1], pos[parent][2],
                               v[0], v[1], v[2], color=c, arrow_length_ratio=0.2)

        # Joint labels
        for name, p in pos.items():
            self.ax.text(p[0], p[1], p[2], f' {name}', fontsize=5, color='blue')

        # ---- overlay original coords ----
        if self.original_coords is not None:
            root_idx = self.name_to_idx[root_name]
            offset = pos[root_name] - self.original_coords[root_idx]
            shifted = self.original_coords + offset
            self.ax.scatter(shifted[:, 0], shifted[:, 1], shifted[:, 2],
                            c='red', s=25, alpha=0.8, zorder=5, label='Original')
            for child, parent in self.ordered_edges:
                ci, pi = self.name_to_idx[child], self.name_to_idx[parent]
                self.ax.plot([shifted[pi, 0], shifted[ci, 0]],
                             [shifted[pi, 1], shifted[ci, 1]],
                             [shifted[pi, 2], shifted[ci, 2]],
                             'r--', lw=1, alpha=0.5)
            for name in self.joint_names:
                idx = self.name_to_idx[name]
                self.ax.text(shifted[idx, 0], shifted[idx, 1], shifted[idx, 2],
                             f' {name}', fontsize=5, color='red')
            self.ax.legend(loc='upper left', fontsize=7)

        # equal-aspect auto-scale
        all_pts = np.array(list(pos.values()))
        if self.original_coords is not None:
            all_pts = np.vstack([all_pts, shifted])
        ctr = all_pts.mean(axis=0)
        span = np.abs(all_pts - ctr).max() * 1.4 + 0.05
        self.ax.set_xlim(ctr[0] - span, ctr[0] + span)
        self.ax.set_ylim(ctr[1] - span, ctr[1] + span)
        self.ax.set_zlim(ctr[2] - span, ctr[2] + span)

        self.ax.set_xlabel('X');
        self.ax.set_ylabel('Y');
        self.ax.set_zlabel('Z')
        self.ax.set_title('FK (blue) vs Original (red dashed)')
        self.ax.view_init(elev=15, azim=60)
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
#  Main - demo with raw coordinates
# ---------------------------------------------------------------------------

def main():
    # joint_defs: [parent, canonical_child, c1, sign]
    #   c1:   LCS Y-axis direction multiplier (-1 flips bone to point toward torso)
    #   sign: [flex, abd, rot] anatomical presentation multipliers
    #         Handles BOTH bilateral symmetry (abd/rot flip for left joints)
    #         AND joint-specific conventions (e.g. positive knee flexion).
    joint_defs = {
        "pelvis": ["", "spine", 1, [1, 1, 1]],
        "spine": ["pelvis", "neck", 1, [-1, 1, 1]],
        "neck": ["spine", "head", 1, [-1, 1, 1]],
        "head": ["neck", "head top", 1, [1, 1, 1]],
        "head top": ["head", "", 1, [1, 1, 1]],

        # Right arm/leg: c1=-1 (bone Y points toward torso for 0° straight limb)
        "right hip": ["pelvis", "right knee", -1, [1, 1, 1]],
        "right knee": ["right hip", "right ankle", -1, [-1, 1, 1]],
        "right ankle": ["right knee", "", -1, [1, 1, 1]],
        "right shoulder": ["neck", "right elbow", -1, [1, 1, 1]],
        "right elbow": ["right shoulder", "right wrist", -1, [1, 1, 1]],
        "right wrist": ["right elbow", "", -1, [1, 1, 1]],

        # Left arm/leg: c1=-1, sign flips abd & rot for bilateral symmetry
        "left hip": ["pelvis", "left knee", -1, [1, -1, -1]],
        "left knee": ["left hip", "left ankle", -1, [-1, -1, -1]],
        "left ankle": ["left knee", "", -1, [1, -1, -1]],
        "left shoulder": ["neck", "left elbow", -1, [1, -1, -1]],
        "left elbow": ["left shoulder", "left wrist", -1, [1, -1, -1]],
        "left wrist": ["left elbow", "", -1, [1, -1, -1]]
    }
    segments_map = {j: (distal, c1) for j, (proximal, distal, c1, sign) in joint_defs.items() if distal}
    sign_map = {j: np.array(sign, dtype=float) for j, (proximal, distal, c1, sign) in joint_defs.items()}
    ktree = {j: proximal for j, (proximal, distal, c1, sign) in joint_defs.items()}
    #c_jnames = list(joint_defs.keys())

    jnames = [
        'pelvis', 'right hip', 'right knee', 'right ankle',
        'left hip', 'left knee', 'left ankle',
        'spine', 'neck', 'head', 'head top',
        'left shoulder', 'left elbow', 'left wrist',
        'right shoulder', 'right elbow', 'right wrist',
    ]
    #print("Joint names:", jnames, c_jnames)
    #print("Kinematic tree:", ktree, c_ktree)
    #print("Segments map:", segments_map, c_segments_map)

    # A single frame of a mid-gait 3D pose (17 joints x 3)
    raw_coords = np.array([
        [0.0024984, 6.8794e-06, -0.0007963],  # pelvis
        [-0.0012018, 0.053859, 0.0014966],  # right hip
        [-0.066229, 0.0030878, -0.15618],  # right knee
        [-0.04598, -0.040396, -0.32004],  # right ankle
        [0.0047319, -0.051271, -0.0031012],  # left hip
        [-0.016447, -0.077047, -0.1662],  # left knee
        [0.14184, -0.02886, -0.24666],  # left ankle
        [0.0032881, 0.0044781, 0.10423],  # spine
        [0.00029384, 0.023843, 0.22248],  # neck
        [-0.051995, 0.018642, 0.2782],  # head
        [-0.1071, 0.024215, 0.33435],  # head top
        [0.00054328, -0.049722, 0.20746],  # left shoulder
        [-0.00057531, -0.10344, 0.097328],  # left elbow
        [-0.057034, -0.10887, 0.020448],  # left wrist
        [0.012189, 0.077975, 0.20765],  # right shoulder
        [0.018423, 0.12456, 0.096108],  # right elbow
        [-0.039036, 0.069698, 0.0088929],  # right wrist
    ])

    name_to_idx = {n: i for i, n in enumerate(jnames)}

    # ---- Compute actual bone lengths ----
    bone_lengths = {}
    for child, parent in ktree.items():
        if parent:
            ci, pi = name_to_idx[child], name_to_idx[parent]
            bone_lengths[child] = float(np.linalg.norm(raw_coords[ci] - raw_coords[pi]))

    # ---- Compute angles ----
    angles_deg = compute_skeleton_angles(raw_coords[np.newaxis], jnames, segments_map,
                                         kinematic_tree=ktree,
                                         sign_map=sign_map).squeeze()  # (K, 3)

    # ---- Recover root LCS so FK starts in the correct frame ----
    # (same construction as inside compute_skeleton_angles for the root)
    l_hip_idx, r_hip_idx = name_to_idx['left hip'], name_to_idx['right hip']
    g_lat = raw_coords[l_hip_idx] - raw_coords[r_hip_idx]
    g_lat /= np.linalg.norm(g_lat) + 1e-8
    spine_idx = name_to_idx['spine']
    pelvis_idx = name_to_idx['pelvis']
    v_long = raw_coords[spine_idx] - raw_coords[pelvis_idx]
    y = v_long / (np.linalg.norm(v_long) + 1e-8)
    z = np.cross(y, g_lat);
    z /= (np.linalg.norm(z) + 1e-8)
    x = np.cross(y, z)
    root_lcs = np.stack([x, y, z], axis=-1)  # (3, 3)

    # ---- Rebuild all LCS frames to compute bone offsets ----
    # We need each parent's LCS to express bone directions in local coords
    children_map = {}
    for child, parent in ktree.items():
        if parent:
            children_map.setdefault(parent, []).append(child)

    preferred = {'pelvis': 'spine', 'neck': 'head'}
    canonical_child = {}
    for parent_name, kids in children_map.items():
        if parent_name in preferred and preferred[parent_name] in kids:
            canonical_child[parent_name] = preferred[parent_name]
        else:
            canonical_child[parent_name] = kids[0]

    lcs_single = {}  # joint_name -> (3, 3) LCS for frame 0
    for name in jnames:
        if name in canonical_child:
            ci = name_to_idx[canonical_child[name]]
            idx = name_to_idx[name]
            vl = raw_coords[ci] - raw_coords[idx]

            c1 = 1
            if name in segments_map:
                _, c1 = segments_map[name]

            vl = vl * c1

            yy = vl / (np.linalg.norm(vl) + 1e-8)
            zz = np.cross(yy, g_lat);
            zz /= (np.linalg.norm(zz) + 1e-8)
            xx = np.cross(yy, zz)
            lcs_single[name] = np.stack([xx, yy, zz], axis=-1)
    # Leaf joints inherit parent's LCS
    for name in jnames:
        if name not in lcs_single:
            pn = ktree.get(name, '')
            if pn and pn in lcs_single:
                lcs_single[name] = lcs_single[pn].copy()
            else:
                lcs_single[name] = np.eye(3)

    # Compute bone offsets: unit direction from parent->child in parent's LCS
    bone_offsets = {}
    for child, parent in ktree.items():
        if parent:
            ci, pi = name_to_idx[child], name_to_idx[parent]
            bone_global = raw_coords[ci] - raw_coords[pi]
            bone_unit = bone_global / (np.linalg.norm(bone_global) + 1e-8)
            R_parent = lcs_single[parent]
            bone_offsets[child] = R_parent.T @ bone_unit  # parent-local direction

    # ---- Print computed angles ----
    print("Computed joint angles (degrees) - stored at child index:")
    for i, name in enumerate(jnames):
        a = angles_deg[i]
        print(f"  {name:>16s}:  Flex={a[0]:7.2f}   Abd={a[1]:7.2f}   Rot={a[2]:7.2f}")

    # ---- Quick numeric FK check (no GUI) ----
    # METHOD: directly use coordinate-derived LCS for position reconstruction
    # R_rel[child] = R_parent^T @ R_child (from compute_skeleton_angles)
    # FK: orient[child] = orient[parent] @ R_rel[child]
    # If orient[parent] == lcs_single[parent], then orient[child] == lcs_single[child]
    # pos[child] = pos[parent] + orient[parent] @ bone_dir_local
    # where bone_dir_local = lcs_single[parent].T @ (raw[child]-raw[parent])
    pos = {}
    orient = {}
    root_name = [n for n, p in ktree.items() if not p][0]
    pos[root_name] = np.zeros(3)
    orient[root_name] = root_lcs.copy()

    # BFS order
    queue = deque([root_name])
    visited = {root_name}
    bfs_edges = []
    while queue:
        cur = queue.popleft()
        for ch, pa in ktree.items():
            if pa == cur and ch not in visited:
                bfs_edges.append((ch, pa))
                visited.add(ch)
                queue.append(ch)

    # First verify orient propagation matches lcs_single
    print("\nLCS propagation check (FK orient vs coord-derived LCS):")
    for child, parent in bfs_edges:
        fk_angles = angles_deg[name_to_idx[child]].copy()
        # Reverse the anatomical convention layer to get pure math angles
        fk_angles *= sign_map[child]
        flex, abd, rot = np.radians(fk_angles)
        cf, sf = np.cos(flex), np.sin(flex)
        ca, sa = np.cos(abd), np.sin(abd)
        cr, sr = np.cos(rot), np.sin(rot)
        Rx = np.array([[1, 0, 0], [0, cf, -sf], [0, sf, cf]])
        Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        Ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
        R_rel = Rx @ Rz @ Ry
        orient[child] = orient[parent] @ R_rel

        # Compare with coordinate-derived LCS
        lcs_err = np.linalg.norm(orient[child] - lcs_single[child])
        print(f"  {child:>16s}: LCS_err={lcs_err:.8f}")

        bone_dir_local = bone_offsets[child] * bone_lengths[child]
        bone = orient[parent] @ bone_dir_local
        pos[child] = pos[parent] + bone

    root_offset = raw_coords[name_to_idx[root_name]]
    print("\nFK round-trip error per joint:")
    max_err = 0.0
    for name in jnames:
        fk = pos[name]
        orig = raw_coords[name_to_idx[name]] - root_offset
        err = np.linalg.norm(fk - orig)
        max_err = max(max_err, err)
        status = 'OK' if err < 1e-6 else 'FAIL'
        print(f"  [{status:>4s}] {name:>16s}:  {err:.8f}")
    print(f"\n  MAX ERROR = {max_err:.8f}")
    if max_err < 1e-4:
        print("  PASS - FK reconstruction matches original coordinates!")
    else:
        print("  FAIL - FK reconstruction does NOT match.")

    # ---- Launch interactive GUI ----
    if '--no-gui' not in sys.argv:
        viz = Biomech3DVisualizer(
            ktree, jnames, segments_map,
            initial_angles=angles_deg,
            bone_lengths=bone_lengths,
            original_coords=raw_coords,
            root_lcs=root_lcs,
            bone_offsets=bone_offsets,
            sign_map=sign_map,
        )
        plt.show()


if __name__ == '__main__':
    import sys
    main()
