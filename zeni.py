import numpy as np
from scipy.signal import find_peaks

class SkeletonGaitAnalysis:
    def __init__(self, fps=30):
        self.fps = fps

    def detect_events_zeni(self, heel_pos, toe_pos, sacrum_pos):
        """
        Implements Zeni et al. (2008) algorithm for HS and TO detection.
        Ref: 'Two simple methods for detecting gait events...'
        
        Args:
            heel_pos (np.array): (N, 3) Heel trajectory
            toe_pos (np.array): (N, 3) Toe trajectory
            sacrum_pos (np.array): (N, 3) Sacrum/Pelvis trajectory
        """
        # Calculate Anterior-Posterior (Y) distance relative to Sacrum
        # Note: Adjust index [1] if Y is up. Assuming Z is up, Y is walking direction.
        # If your data has Y as vertical, change to use the horizontal axis.
        
        # 1. Zeni "Velocity" or "Coordinate" method. 
        # The Coordinate method is robust: HS is Max(Heel_Y - Sacrum_Y), TO is Min(Toe_Y - Sacrum_Y)
        # assuming walking in Positive Y direction.
        
        # Adjust for walking direction. If the mean displacement is ~0 (standing/oscillating),
        # np.sign returns 0, which would zero-out the signals and suppress peaks entirely.
        direction = np.sign(np.mean(np.diff(sacrum_pos[:, 1])))  # Detect if walking +Y or -Y
        if direction == 0:
            direction = 1.0  # fallback so we still get peaks even if displacement is tiny
        
        heel_rel = direction * (heel_pos[:, 1] - sacrum_pos[:, 1])
        toe_rel = direction * (toe_pos[:, 1] - sacrum_pos[:, 1])

        # Heel Strikes: Local Maxima of Heel-Sacrum distance
        hs_indices, _ = find_peaks(heel_rel, distance=self.fps//2)
        
        # Toe Offs: Local Minima of Toe-Sacrum distance (or Maxima of negative)
        to_indices, _ = find_peaks(-toe_rel, distance=self.fps//2)
        
        return np.sort(hs_indices), np.sort(to_indices)

    def calculate_step_stride(self, l_heel, r_heel, l_hs_events, r_hs_events):
        """Calculates Step and Stride Lengths based on HS events."""
        step_lengths = []
        stride_lengths = []

        # Stride Length (Left)
        for i in range(len(l_hs_events) - 1):
            idx1 = l_hs_events[i]
            idx2 = l_hs_events[i+1]
            dist = np.linalg.norm(l_heel[idx2] - l_heel[idx1])
            stride_lengths.append(dist)

        # Step Length (Distance between heels at HS)
        # Find closest Right HS for every Left HS to define a 'step'
        for t_lhs in l_hs_events:
            # Step length is distance between L_Heel and R_Heel at moment of L_HS
            dist = np.linalg.norm(l_heel[t_lhs] - r_heel[t_lhs])
            step_lengths.append(dist)
            
        return np.mean(step_lengths), np.mean(stride_lengths)

    def calculate_gait_line(self, ankle_pos, hs_events, to_events):
        """
        Calculates Length of Gait Line: Trajectory length of Ankle during Stance.
        """
        gait_lines = []
        
        # Pair HS with subsequent TO
        for t_hs in hs_events:
            # Find first TO after this HS
            future_tos = to_events[to_events > t_hs]
            if len(future_tos) == 0: continue
            t_to = future_tos[0]
            
            # Extract trajectory during Stance
            stance_path = ankle_pos[t_hs:t_to]
            
            # Calculate cumulative distance (arc length)
            # diffs = sqrt((x2-x1)^2 + (y2-y1)^2 + ...)
            diffs = np.diff(stance_path, axis=0)
            segment_lengths = np.linalg.norm(diffs, axis=1)
            total_length = np.sum(segment_lengths)
            gait_lines.append(total_length)
            
        return np.mean(gait_lines)

    def calculate_single_support_line(self, ankle_pos, current_hs, contra_hs, contra_to):
        """
        Calculates Single Support Line: Trajectory length during Single Support.
        Start: Contra TO -> End: Contra HS
        """
        ss_lines = []
        
        # For every Stance phase of Current foot...
        for t_hs in current_hs:
            # Find the Single Support window inside this stance
            # SS starts when Opposite foot leaves ground (Contra TO)
            # SS ends when Opposite foot hits ground (Contra HS)
            
            # Find Contra TO that happens AFTER my HS
            relevant_c_to = contra_to[contra_to > t_hs]
            if len(relevant_c_to) == 0: continue
            t_start_ss = relevant_c_to[0]
            
            # Find Contra HS that happens AFTER that Start
            relevant_c_hs = contra_hs[contra_hs > t_start_ss]
            if len(relevant_c_hs) == 0: continue
            t_end_ss = relevant_c_hs[0]
            
            # Extract path
            ss_path = ankle_pos[t_start_ss:t_end_ss]
            diffs = np.diff(ss_path, axis=0)
            length = np.sum(np.linalg.norm(diffs, axis=1))
            ss_lines.append(length)
            
        return np.mean(ss_lines)

# Example Usage Mockup
# gait = SkeletonGaitAnalysis(fps=30)
# l_hs, l_to = gait.detect_events_zeni(l_heel, l_toe, sacrum)
# r_hs, r_to = gait.detect_events_zeni(r_heel, r_toe, sacrum)
# step_len, stride_len = gait.calculate_step_stride(l_heel, r_heel, l_hs, r_hs)
# l_gait_line = gait.calculate_gait_line(l_ankle, l_hs, l_to)
# l_ss_line = gait.calculate_single_support_line(l_ankle, l_hs, r_hs, r_to)
