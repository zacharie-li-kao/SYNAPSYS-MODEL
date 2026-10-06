"""
Phase 3: Scalable Synapse Network (Configurable Array Size)
============================================================

This module demonstrates emergent network behavior from device physics.

Features:
- Configurable N×M array of visual synapses with device-to-device variability
- Spatial light pattern stimulation (vertical bars, horizontal bars, diagonals, etc.)
- Real-time heatmap visualization of conductance states
- Weight matrix evolution tracking

This bridges single-device characterization to network-level computation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import copy


# =============================================================================
# SYNAPSE NETWORK CLASS
# =============================================================================

class SynapseNetwork:
    """
    A configurable N×M network of visual synapses with device variability.
    
    Each synapse can be independently addressed by spatial light patterns.
    """
    
    def __init__(self, base_params, shape=(3, 3), variability=0.1):
        """
        Create an N×M network of synapses.
        
        Args:
            base_params (dict): Base synapse parameters
            shape (tuple): Network dimensions (rows, cols), e.g., (3, 3), (5, 5), (10, 10)
            variability (float): Device-to-device variation (0.1 = 10%)
        """
        self.shape = shape
        self.n_synapses = shape[0] * shape[1]
        self.base_params = base_params
        self.variability = variability
        
        # Create synapse array with variability
        self.synapses = self._create_synapse_array()
        
        # Initialize conductance matrix
        self.G_matrix = np.zeros(self.shape)
        self.update_conductance_matrix()
        
        # History tracking
        self.history = {
            'G_matrices': [self.G_matrix.copy()],
            'patterns': [np.zeros(self.shape)],
            'time_points': [0],
            'pattern_names': ['Initial State']
        }
        
        self.current_time = 0
    
    @staticmethod
    def _variation_factor(variability):
        """A strictly positive multiplicative variation with unit mean.

        M14: this used `np.random.normal(1.0, variability)`, which is unbounded
        below. At the GUI-permitted maximum of 100% that produced, out of 25
        devices: 5 with negative G_min, 8 with negative A_peak (potentiation
        running backwards), 5 with negative decay_tau (runaway rather than
        relaxation), and 2 with G_min >= G_max. Seven of the 25 held NEGATIVE
        CONDUCTANCE after a single pulse — a physically meaningless state that
        the simulation nonetheless propagated.

        A lognormal is used instead: device-to-device spread in fabrication is
        multiplicative, so a lognormal is the physically appropriate
        distribution, and it cannot produce a negative factor at any spread.
        The shape parameter is chosen so the factor's mean is 1 and its
        relative standard deviation equals `variability`, matching what the GUI
        control claims to set.
        """
        if variability <= 0:
            return 1.0
        sigma = np.sqrt(np.log(1.0 + variability ** 2))
        mu = -0.5 * sigma ** 2          # so that E[exp(N(mu, sigma))] == 1
        return float(np.exp(np.random.normal(mu, sigma)))

    def _create_synapse_array(self):
        """Create N×M array of synapses with variability."""
        from Main import VisualSynapse  # Import from Main.py

        synapses = []

        for i in range(self.shape[0]):
            row = []
            for j in range(self.shape[1]):
                # Add device-to-device variability
                params = {}
                for key, value in self.base_params.items():
                    if isinstance(value, (int, float)):
                        params[key] = value * self._variation_factor(self.variability)
                    else:
                        params[key] = value

                # The bounds are varied independently, so a large spread can
                # still invert them. An inverted device is not a device: G_min
                # above G_max makes (G_max - G)^alpha take a negative base and
                # the clip pin the state to a bound immediately. Ordering is
                # restored rather than the draw being rejected, which preserves
                # the intended spread.
                if params.get('G_min') is not None and params.get('G_max') is not None:
                    if params['G_min'] > params['G_max']:
                        params['G_min'], params['G_max'] = (
                            params['G_max'], params['G_min'])
                    if params['G_min'] == params['G_max']:
                        # Degenerate device: no dynamic range at all. Give it
                        # the smallest range the base parameters imply rather
                        # than a zero-width one, which would divide by zero in
                        # every downstream normalisation.
                        params['G_max'] = params['G_min'] * 1.0001

                # A_peak carries units S^(1-alpha): rescale so the exponent spread
                # shapes the curve on the nominal window without moving the rate.
                ref = self.base_params
                W_ref = ref['G_max'] - ref['G_min']
                params['A_peak'] *= W_ref ** (ref['alpha'] - params['alpha'])
                params['B_peak'] *= W_ref ** (ref['beta'] - params['beta'])

                # Create synapse with varied parameters
                synapse = VisualSynapse(
                    G_min=params.get('G_min'),
                    G_max=params.get('G_max'),
                    alpha=params.get('alpha'),
                    beta=params.get('beta'),
                    lambda_peak=params.get('lambda_peak'),
                    lambda_width=params.get('lambda_width'),
                    decay_tau=params.get('decay_tau'),
                    A_peak=params.get('A_peak'),
                    B_peak=params.get('B_peak'),
                    wavelength_curve=params.get('wavelength_curve'),
                    wavelength_range=params.get('wavelength_range')
                )
                
                row.append(synapse)
            synapses.append(row)
        
        return synapses
    
    def update_conductance_matrix(self):
        """Update the conductance matrix from individual synapses."""
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.G_matrix[i, j] = self.synapses[i][j].G
    
    def apply_spatial_pattern(self, pattern, intensity_mW_cm2, wavelength_nm, 
                             duration_ms, mode='potentiation', stimulus_type='light',
                             pattern_name="Custom"):
        """
        Apply a spatial pattern to the network.
        
        Args:
            pattern (np.ndarray): N×M binary or continuous pattern (0 to 1)
            intensity_mW_cm2 (float): Base light intensity (mW/cm²) or voltage (V)
            wavelength_nm (float): Light wavelength (nm) - ignored for electrical
            duration_ms (float): Pulse duration
            mode (str): 'potentiation' or 'depression' - ONLY used for electrical stimuli.
                       For light stimuli, the wavelength response curve determines the effect.
            stimulus_type (str): 'light' or 'electrical'
            pattern_name (str): Name of the pattern for tracking
        """
        # Validate pattern shape
        if pattern.shape != self.shape:
            raise ValueError(f"Pattern shape {pattern.shape} does not match network shape {self.shape}")
        
        # Apply stimulus to each synapse based on pattern
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                local_intensity = intensity_mW_cm2 * pattern[i, j]
                self.synapses[i][j].apply_stimulus(
                    local_intensity, wavelength_nm, duration_ms, mode, stimulus_type
                )
        
        # Update matrix and history
        self.update_conductance_matrix()
        self.current_time += duration_ms / 1000.0
        
        self.history['G_matrices'].append(self.G_matrix.copy())
        self.history['patterns'].append(pattern.copy())
        self.history['time_points'].append(self.current_time)
        self.history['pattern_names'].append(pattern_name)
    
    
    def rest(self, duration_ms):
        """Allow all synapses to rest (decay)."""
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.synapses[i][j].rest(duration_ms)
        
        self.update_conductance_matrix()
        self.current_time += duration_ms / 1000.0
        
        self.history['G_matrices'].append(self.G_matrix.copy())
        self.history['patterns'].append(np.zeros(self.shape))
        self.history['time_points'].append(self.current_time)
        self.history['pattern_names'].append('Rest')
    
    def reset(self):
        """Reset all synapses to initial state."""
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.synapses[i][j].reset()
        
        self.update_conductance_matrix()
        self.current_time = 0
        
        self.history = {
            'G_matrices': [self.G_matrix.copy()],
            'patterns': [np.zeros(self.shape)],
            'time_points': [0],
            'pattern_names': ['Initial State']
        }
    
    def get_statistics(self):
        """Get network statistics."""
        G_flat = self.G_matrix.flatten()
        
        return {
            'mean_G': np.mean(G_flat),
            'std_G': np.std(G_flat),
            'min_G': np.min(G_flat),
            'max_G': np.max(G_flat),
            'total_weight': np.sum(G_flat)
        }


# =============================================================================
# SPATIAL PATTERNS (Scalable)
# =============================================================================

class SpatialPatterns:
    """Pre-defined spatial light patterns for network stimulation (scalable)."""
    
    @staticmethod
    def vertical_bar(shape, column):
        """
        Vertical bar at specified column.
        
        Args:
            shape (tuple): Network shape (rows, cols)
            column (int): Column index (0 to cols-1)
        """
        pattern = np.zeros(shape)
        if 0 <= column < shape[1]:
            pattern[:, column] = 1.0
        return pattern
    
    @staticmethod
    def horizontal_bar(shape, row):
        """
        Horizontal bar at specified row.
        
        Args:
            shape (tuple): Network shape (rows, cols)
            row (int): Row index (0 to rows-1)
        """
        pattern = np.zeros(shape)
        if 0 <= row < shape[0]:
            pattern[row, :] = 1.0
        return pattern
    
    @staticmethod
    def diagonal_main(shape):
        """Main diagonal (top-left to bottom-right)."""
        pattern = np.zeros(shape)
        n = min(shape[0], shape[1])
        for i in range(n):
            pattern[i, i] = 1.0
        return pattern
    
    @staticmethod
    def diagonal_anti(shape):
        """Anti-diagonal (top-right to bottom-left)."""
        pattern = np.zeros(shape)
        n = min(shape[0], shape[1])
        for i in range(n):
            pattern[i, shape[1] - 1 - i] = 1.0
        return pattern
    
    @staticmethod
    def cross(shape):
        """Cross pattern (center row + center column)."""
        pattern = np.zeros(shape)
        center_row = shape[0] // 2
        center_col = shape[1] // 2
        pattern[center_row, :] = 1.0  # Horizontal
        pattern[:, center_col] = 1.0  # Vertical
        return pattern
    
    @staticmethod
    def corners(shape):
        """Four corners."""
        pattern = np.zeros(shape)
        pattern[0, 0] = 1.0
        pattern[0, shape[1]-1] = 1.0
        pattern[shape[0]-1, 0] = 1.0
        pattern[shape[0]-1, shape[1]-1] = 1.0
        return pattern
    
    @staticmethod
    def center(shape):
        """Center pixel only."""
        pattern = np.zeros(shape)
        center_row = shape[0] // 2
        center_col = shape[1] // 2
        pattern[center_row, center_col] = 1.0
        return pattern
    
    @staticmethod
    def uniform(shape):
        """Uniform illumination (all pixels)."""
        return np.ones(shape)
    
    @staticmethod
    def checkerboard(shape):
        """Checkerboard pattern."""
        pattern = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (i + j) % 2 == 0:
                    pattern[i, j] = 1.0
        return pattern
    
    @staticmethod
    def left_half(shape):
        """Left half of the array."""
        pattern = np.zeros(shape)
        mid = shape[1] // 2
        pattern[:, :mid] = 1.0
        return pattern
    
    @staticmethod
    def right_half(shape):
        """Right half of the array."""
        pattern = np.zeros(shape)
        mid = shape[1] // 2
        pattern[:, mid:] = 1.0
        return pattern
    
    @staticmethod
    def top_half(shape):
        """Top half of the array."""
        pattern = np.zeros(shape)
        mid = shape[0] // 2
        pattern[:mid, :] = 1.0
        return pattern
    
    @staticmethod
    def bottom_half(shape):
        """Bottom half of the array."""
        pattern = np.zeros(shape)
        mid = shape[0] // 2
        pattern[mid:, :] = 1.0
        return pattern
    
    @staticmethod
    def border(shape, thickness=1):
        """Border pattern."""
        pattern = np.zeros(shape)
        pattern[:thickness, :] = 1.0  # Top
        pattern[-thickness:, :] = 1.0  # Bottom
        pattern[:, :thickness] = 1.0  # Left
        pattern[:, -thickness:] = 1.0  # Right
        return pattern
    
    @staticmethod
    def random(shape, density=0.5):
        """Random pattern with specified density."""
        return (np.random.rand(*shape) < density).astype(float)
    
    @staticmethod
    def circle(shape, radius_fraction=0.3):
        """Circular pattern centered in the network."""
        pattern = np.zeros(shape)
        center_row = shape[0] / 2
        center_col = shape[1] / 2
        radius = min(shape[0], shape[1]) * radius_fraction
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                dist = np.sqrt((i - center_row + 0.5)**2 + (j - center_col + 0.5)**2)
                if dist <= radius:
                    pattern[i, j] = 1.0
        return pattern
    
    @staticmethod
    def duck(shape):
        """
        Detailed duck profile silhouette (facing right).
        Fully vectorized for excellent scaling to high-resolution networks.
        Clear body, neck, head, eye, and beak structure.
        """
        rows, cols = shape
        r, c = rows, cols
        
        # Create coordinate grids
        Y, X = np.ogrid[:rows, :cols]
        
        # Main body (large back ellipse, slightly lower than center)
        body_center_y = 0.62 * r
        body_center_x = 0.35 * c
        body = (((X - body_center_x) / (0.28 * c))**2 + 
                ((Y - body_center_y) / (0.32 * r))**2) <= 1.0
        
        # Chest bulge (forward-facing ellipse)
        chest_center_y = 0.58 * r
        chest_center_x = 0.52 * c
        chest = (((X - chest_center_x) / (0.20 * c))**2 + 
                 ((Y - chest_center_y) / (0.24 * r))**2) <= 1.0
        
        # Neck (narrow connecting tube between body and head)
        neck_center_x = 0.62 * c
        neck_center_y = 0.45 * r
        neck = (((X - neck_center_x) / (0.10 * c))**2 + 
                ((Y - neck_center_y) / (0.18 * r))**2) <= 1.0
        
        # Head (rounded circle)
        head_center_x = 0.71 * c
        head_center_y = 0.32 * r
        head_radius = 0.12 * min(r, c)
        head = ((X - head_center_x)**2 + (Y - head_center_y)**2) <= head_radius**2
        
        # Eye (small dark accent)
        eye_center_x = 0.76 * c
        eye_center_y = 0.29 * r
        eye_radius = 0.025 * min(r, c)
        eye = ((X - eye_center_x)**2 + (Y - eye_center_y)**2) <= eye_radius**2
        
        # Beak (pointed, slightly curved downward)
        beak_base_x = 0.81 * c
        beak_base_y = 0.33 * r
        beak_tip_x = 0.93 * c
        beak_tip_y = 0.36 * r
        # Triangular beak region
        beak_dist = np.abs((beak_tip_y - beak_base_y) * (X - beak_base_x) - 
                           (beak_tip_x - beak_base_x) * (Y - beak_base_y))
        beak_length = np.sqrt((beak_tip_x - beak_base_x)**2 + (beak_tip_y - beak_base_y)**2)
        beak_on_line = beak_dist <= (0.04 * min(r, c) * beak_length)
        beak_x_valid = (X >= beak_base_x) & (X <= beak_tip_x)
        beak_y_valid = (Y >= np.minimum(beak_base_y, beak_tip_y) - 0.05*r) & \
                       (Y <= np.maximum(beak_base_y, beak_tip_y) + 0.05*r)
        beak = beak_on_line & beak_x_valid & beak_y_valid
        
        # Tail (trailing feather-like shape at back)
        tail_center_x = 0.15 * c
        tail_center_y = 0.60 * r
        tail = (((X - tail_center_x) / (0.12 * c))**2 + 
                ((Y - tail_center_y) / (0.16 * r))**2) <= 1.0
        
        # Water line (optional detail: small underside bulge)
        water_center_x = 0.45 * c
        water_center_y = 0.78 * r
        water = (((X - water_center_x) / (0.30 * c))**2 + 
                 ((Y - water_center_y) / (0.08 * r))**2) <= 1.0
        
        # Combine all parts
        pattern = np.zeros(shape)
        duck_mask = body | chest | neck | head | beak | tail | water
        pattern[duck_mask & ~eye] = 1.0  # Fill everything except eye (eye is negative space)
        
        return pattern

    @staticmethod
    def thumbs_up(shape):
        """
        Detailed thumbs-up silhouette with distinct fingers and thumb.
        Fully vectorized for excellent scaling to high-resolution networks.
        Clear hand structure with recognizable pose.
        """
        rows, cols = shape
        r, c = rows, cols
        
        # Create coordinate grids
        Y, X = np.ogrid[:rows, :cols]
        
        # Palm (main rectangular body)
        palm = (X >= 0.30 * c) & (X <= 0.70 * c) & \
               (Y >= 0.50 * r) & (Y <= 0.85 * r)
        
        # Four extended fingers (rounded rectangles above palm)
        finger_y_top = 0.15 * r
        finger_y_bottom = 0.50 * r
        finger_height = finger_y_bottom - finger_y_top
        
        # Finger 1 (pinky, left)
        f1_x_left, f1_x_right = 0.32 * c, 0.42 * c
        f1_rounded = (((X - f1_x_left) / (0.05 * c))**2 + 
                      ((Y - finger_y_top) / (0.06 * r))**2 <= 1.0) | \
                     (((X - f1_x_right) / (0.05 * c))**2 + 
                      ((Y - finger_y_top) / (0.06 * r))**2 <= 1.0) | \
                     ((X >= f1_x_left) & (X <= f1_x_right) & (Y >= finger_y_top) & (Y <= finger_y_bottom))
        finger1 = f1_rounded
        
        # Finger 2 (ring, middle-left)
        f2_x_left, f2_x_right = 0.44 * c, 0.54 * c
        f2_rounded = (((X - f2_x_left) / (0.05 * c))**2 + 
                      ((Y - finger_y_top) / (0.06 * r))**2 <= 1.0) | \
                     (((X - f2_x_right) / (0.05 * c))**2 + 
                      ((Y - finger_y_top) / (0.06 * r))**2 <= 1.0) | \
                     ((X >= f2_x_left) & (X <= f2_x_right) & (Y >= finger_y_top) & (Y <= finger_y_bottom))
        finger2 = f2_rounded
        
        # Finger 3 (middle, middle-right)
        f3_x_left, f3_x_right = 0.56 * c, 0.66 * c
        f3_rounded = (((X - f3_x_left) / (0.05 * c))**2 + 
                      ((Y - finger_y_top) / (0.06 * r))**2 <= 1.0) | \
                     (((X - f3_x_right) / (0.05 * c))**2 + 
                      ((Y - finger_y_top) / (0.06 * r))**2 <= 1.0) | \
                     ((X >= f3_x_left) & (X <= f3_x_right) & (Y >= finger_y_top) & (Y <= finger_y_bottom))
        finger3 = f3_rounded
        
        # Finger 4 (index, right)
        f4_x_left, f4_x_right = 0.68 * c, 0.78 * c
        f4_rounded = (((X - f4_x_left) / (0.05 * c))**2 + 
                      ((Y - finger_y_top) / (0.06 * r))**2 <= 1.0) | \
                     (((X - f4_x_right) / (0.05 * c))**2 + 
                      ((Y - finger_y_top) / (0.06 * r))**2 <= 1.0) | \
                     ((X >= f4_x_left) & (X <= f4_x_right) & (Y >= finger_y_top) & (Y <= finger_y_bottom))
        finger4 = f4_rounded
        
        all_fingers = finger1 | finger2 | finger3 | finger4
        
        # Thumb (organic ellipse pointing upward-left)
        thumb_center_x = 0.22 * c
        thumb_center_y = 0.55 * r
        thumb = (((X - thumb_center_x) / (0.09 * c))**2 + 
                 ((Y - thumb_center_y) / (0.22 * r))**2) <= 1.0
        
        # Wrist/forearm (narrow rectangle extending down from palm)
        wrist = (X >= 0.40 * c) & (X <= 0.60 * c) & \
                (Y >= 0.85 * r) & (Y <= 1.00 * r)
        
        # Combine all parts
        pattern = np.zeros(shape)
        hand_mask = palm | all_fingers | thumb | wrist
        pattern[hand_mask] = 1.0
        
        return pattern

    @staticmethod
    def raised_fist(shape):
        """
        Detailed raised fist silhouette (vertical orientation, similar to ✊ emoji).
        Fully vectorized for excellent scaling to high-resolution networks.
        Clear knuckles, thumb, and wrist structure.
        """
        rows, cols = shape
        r, c = rows, cols
        
        # Create coordinate grids
        Y, X = np.ogrid[:rows, :cols]
        
        # Main fist body (large rounded rectangle)
        fist_body_left = 0.30 * c
        fist_body_right = 0.70 * c
        fist_body_top = 0.15 * r
        fist_body_bottom = 0.70 * r
        fist_body_center_x = 0.50 * c
        fist_body_center_y = 0.50 * r
        
        # Main body as rounded rectangle (using ellipse corners)
        fist_rect = (X >= fist_body_left + 0.08*c) & (X <= fist_body_right - 0.08*c) & \
                    (Y >= fist_body_top) & (Y <= fist_body_bottom)
        fist_top_curve = (((X - fist_body_center_x) / (0.20 * c))**2 + 
                          ((Y - fist_body_top) / (0.15 * r))**2) <= 1.0
        fist_bottom_curve = (((X - fist_body_center_x) / (0.20 * c))**2 + 
                             ((Y - fist_body_bottom) / (0.15 * r))**2) <= 1.0
        fist_left_curve = (((X - fist_body_left) / (0.10 * c))**2 + 
                           ((Y - fist_body_center_y) / (0.28 * r))**2) <= 1.0
        fist_right_curve = (((X - fist_body_right) / (0.10 * c))**2 + 
                            ((Y - fist_body_center_y) / (0.28 * r))**2) <= 1.0
        fist_body = fist_rect | fist_top_curve | fist_bottom_curve | fist_left_curve | fist_right_curve
        
        # Four prominent knuckles (rounded bumps on top)
        knuckle_radius_sq = (0.08 * c)**2
        knuckle_y = 0.12 * r
        knuckle1 = ((X - 0.35 * c)**2 + (Y - knuckle_y)**2) <= knuckle_radius_sq
        knuckle2 = ((X - 0.45 * c)**2 + (Y - knuckle_y)**2) <= knuckle_radius_sq
        knuckle3 = ((X - 0.55 * c)**2 + (Y - knuckle_y)**2) <= knuckle_radius_sq
        knuckle4 = ((X - 0.65 * c)**2 + (Y - knuckle_y)**2) <= knuckle_radius_sq
        all_knuckles = knuckle1 | knuckle2 | knuckle3 | knuckle4
        
        # Thumb (side-protruding rounded shape)
        thumb_center_x = 0.18 * c
        thumb_center_y = 0.40 * r
        thumb_curve1 = (((X - thumb_center_x) / (0.10 * c))**2 + 
                        ((Y - thumb_center_y) / (0.16 * r))**2) <= 1.0
        thumb_center_x2 = 0.20 * c
        thumb_center_y2 = 0.55 * r
        thumb_curve2 = (((X - thumb_center_x2) / (0.11 * c))**2 + 
                        ((Y - thumb_center_y2) / (0.12 * r))**2) <= 1.0
        thumb = thumb_curve1 | thumb_curve2
        
        # Wrist/forearm (tapered rectangle below fist)
        wrist_top = 0.70 * r
        wrist_bottom = 1.0 * r
        wrist_left = 0.38 * c
        wrist_right = 0.62 * c
        wrist_taper = np.maximum(0, (Y - wrist_top) / (wrist_bottom - wrist_top)) * 0.08 * c
        wrist = (X >= wrist_left + wrist_taper) & (X <= wrist_right - wrist_taper) & \
                (Y >= wrist_top) & (Y <= wrist_bottom)
        
        # Combine all parts
        pattern = np.zeros(shape)
        fist_mask = fist_body | all_knuckles | thumb | wrist
        pattern[fist_mask] = 1.0
        
        return pattern
    
    @staticmethod
    def get_all_patterns():
        """
        Map every dropdown name to a builder that takes a shape.

        Every value used to be None, so the dict conveyed only the names and no
        caller could actually build a pattern from it. Returning real callables
        makes this usable as the single source of truth for which patterns
        exist — and lets a test verify that the GUI dispatcher covers all of
        them (audit M18: "Top Half" was offered in the dropdown with no branch
        in the dispatcher, and silently became a uniform pattern).

        'Custom (from Excel)' maps to None deliberately: it has no builder
        because its content comes from a user-supplied file.
        """
        P = SpatialPatterns
        return {
            'Vertical Bar (Left)': lambda s: P.vertical_bar(s, 0),
            'Vertical Bar (Center)': lambda s: P.vertical_bar(s, s[1] // 2),
            'Vertical Bar (Right)': lambda s: P.vertical_bar(s, s[1] - 1),
            'Horizontal Bar (Top)': lambda s: P.horizontal_bar(s, 0),
            'Horizontal Bar (Center)': lambda s: P.horizontal_bar(s, s[0] // 2),
            'Horizontal Bar (Bottom)': lambda s: P.horizontal_bar(s, s[0] - 1),
            'Diagonal (Main)': P.diagonal_main,
            'Diagonal (Anti)': P.diagonal_anti,
            'Cross': P.cross,
            'Corners': P.corners,
            'Center': P.center,
            'Uniform': P.uniform,
            'Checkerboard': P.checkerboard,
            'Left Half': P.left_half,
            'Right Half': P.right_half,
            'Top Half': P.top_half,
            'Bottom Half': P.bottom_half,
            'Border': P.border,
            'Random': P.random,
            'Circle': P.circle,
            'Duck': P.duck,
            'Thumbs Up': P.thumbs_up,
            'Raised Fist': P.raised_fist,
            'Custom (from Excel)': None,
        }


# =============================================================================
# VISUALIZATION FUNCTIONS (Scalable)
# =============================================================================

def plot_network_state(network, figsize=(10, 8)):
    """
    Plot current network conductance state as a heatmap.
    
    Works with any network size.
    
    Args:
        network (SynapseNetwork): The synapse network
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, num='Current Network State')
    fig.suptitle('Current Synapse Network State', fontsize=16, fontweight='bold')
    
    G_uS = network.G_matrix * 1e6
    
    im = ax.imshow(G_uS, cmap='viridis', aspect='auto', interpolation='nearest')
    ax.set_title(f'Conductance (µS) - t={network.current_time:.1f}s', fontsize=12)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Dynamic tick labels
    ax.set_xticks(range(network.shape[1]))
    ax.set_yticks(range(network.shape[0]))
    
    # Add text annotations (only for small networks to avoid clutter)
    if network.shape[0] <= 10 and network.shape[1] <= 10:
        for i in range(network.shape[0]):
            for j in range(network.shape[1]):
                text = ax.text(j, i, f'{G_uS[i, j]:.1f}',
                             ha="center", va="center", color="white", 
                             fontsize=max(6, 10 - network.shape[0] // 2))
    
    fig.colorbar(im, ax=ax, label='Conductance (µS)')
    
    # Add statistics
    stats = network.get_statistics()
    stats_text = (f"Mean: {stats['mean_G']*1e6:.2f} µS\n"
                  f"Std: {stats['std_G']*1e6:.2f} µS\n"
                  f"Range: {stats['min_G']*1e6:.2f} - {stats['max_G']*1e6:.2f} µS")
    ax.text(1.15, 0.5, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    return fig


def plot_network_evolution(network, figsize=(14, 10), max_plots_per_page=20):
    n_states = len(network.history['G_matrices'])
    if n_states == 0:
        return None
    
    all_G = [G * 1e6 for G in network.history['G_matrices']]
    vmin = min([np.min(G) for G in all_G])
    vmax = max([np.max(G) for G in all_G])
    
    show_text = network.shape[0] <= 5 and network.shape[1] <= 5
    show_tick_labels = network.shape[0] <= 10 and network.shape[1] <= 10 and n_states <= 20
    
    if n_states > max_plots_per_page:
        n_cols = 5
        n_rows = int(np.ceil(n_states / n_cols))
        fig_height = max(14, n_rows * 2.5)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, fig_height))
        fig.suptitle(f'Network Evolution Over Time ({n_states} states)', fontsize=14, fontweight='bold')
        axes = axes.reshape(n_rows, n_cols)
        show_tick_labels = False
    else:
        n_cols = min(5, n_states)
        n_rows = int(np.ceil(n_states / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle('Network Evolution Over Time', fontsize=16, fontweight='bold')
        if n_states == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(n_rows, n_cols)
    
    im = None
    for idx in range(n_states):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        G_uS = network.history['G_matrices'][idx] * 1e6
        time = network.history['time_points'][idx]
        pattern_name = network.history['pattern_names'][idx]
        im = ax.imshow(G_uS, cmap='viridis', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f't={time:.1f}s\n{pattern_name}', fontsize=9)
        if show_tick_labels:
            ax.set_xticks(range(network.shape[1]))
            ax.set_yticks(range(network.shape[0]))
            ax.tick_params(labelsize=7)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        if show_text:
            for i in range(network.shape[0]):
                for j in range(network.shape[1]):
                    ax.text(j, i, f'{G_uS[i, j]:.1f}', ha="center", va="center", color="white", fontsize=6)
    
    for row in range(n_rows):
        for col in range(n_cols):
            if row * n_cols + col >= n_states:
                axes[row, col].axis('off')
    
    if im is not None:
        cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Conductance (µS)')
    
    fig.tight_layout(rect=[0, 0, 0.92, 0.96], h_pad=2.0)
    return fig

def create_animation(network, save_path=None, interval=500):
    """
    Create an animation of network evolution.
    
    Args:
        network (SynapseNetwork): The synapse network with history
        save_path (str, optional): Path to save animation (e.g., 'network.gif')
        interval (int): Delay between frames in milliseconds
    
    Returns:
        matplotlib.animation.FuncAnimation: The animation object
    """
    plt.figure()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), num='Network Animation')
    
    # Find global min/max for consistent colormap
    all_G = [G * 1e6 for G in network.history['G_matrices']]
    vmin = min([np.min(G) for G in all_G])
    vmax = max([np.max(G) for G in all_G])
    
    # Determine if we should show text
    show_text = network.shape[0] <= 10 and network.shape[1] <= 10
    
    def update(frame):
        for ax in axes:
            ax.clear()
        
        G_uS = network.history['G_matrices'][frame] * 1e6
        pattern = network.history['patterns'][frame]
        time = network.history['time_points'][frame]
        pattern_name = network.history['pattern_names'][frame]
        
        # Conductance heatmap
        im1 = axes[0].imshow(G_uS, cmap='viridis', aspect='auto', 
                            interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Conductance (µS) - t={time:.1f}s')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')
        axes[0].set_xticks(range(network.shape[1]))
        axes[0].set_yticks(range(network.shape[0]))
        
        if show_text:
            for i in range(network.shape[0]):
                for j in range(network.shape[1]):
                    axes[0].text(j, i, f'{G_uS[i, j]:.1f}',
                               ha="center", va="center", color="white", fontsize=9)
        
        # Pattern
        im2 = axes[1].imshow(pattern, cmap='Greys', aspect='auto', 
                            interpolation='nearest', vmin=0, vmax=1)
        axes[1].set_title(f'Stimulus Pattern\n{pattern_name}')
        axes[1].set_xlabel('Column')
        axes[1].set_ylabel('Row')
        axes[1].set_xticks(range(network.shape[1]))
        axes[1].set_yticks(range(network.shape[0]))
        
        if show_text:
            for i in range(network.shape[0]):
                for j in range(network.shape[1]):
                    color = "white" if pattern[i, j] > 0.5 else "black"
                    axes[1].text(j, i, f'{pattern[i, j]:.1f}',
                               ha="center", va="center", color=color, fontsize=9)
        
        fig.suptitle(f'Synapse Network Evolution - Frame {frame+1}/{len(network.history["G_matrices"])}',
                    fontsize=12, fontweight='bold')
    
    anim = FuncAnimation(fig, update, frames=len(network.history['G_matrices']),
                        interval=interval, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow')
        print(f"Animation saved to {save_path}")
    
    return anim


# =============================================================================
# SPIKING NEURAL NETWORK WITH STDP
# =============================================================================

class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) neuron model.
    
    Membrane potential dynamics:
        τ_m * dV/dt = -(V - V_rest) + R*I_syn
    
    Spike when V >= V_thresh, then reset to V_reset.
    """
    
    def __init__(self, tau_m_ms=20.0, V_rest=-70.0, V_thresh=-50.0, 
                 V_reset=-70.0, R_membrane=10.0, refract_period_ms=2.0):
        """
        Initialize LIF neuron.
        
        Args:
            tau_m_ms: Membrane time constant (ms)
            V_rest: Resting potential (mV)
            V_thresh: Spike threshold (mV)
            V_reset: Reset potential after spike (mV)
            R_membrane: Membrane resistance (MΩ)
            refract_period_ms: Refractory period (ms)
        """
        self.tau_m = tau_m_ms
        self.V_rest = V_rest
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.R = R_membrane
        self.refract_period = refract_period_ms
        
        # State variables
        self.V = V_rest
        self.last_spike_time = -float('inf')
        self.spike_times = []
    
    def update(self, I_syn, dt_ms, current_time_ms):
        """
        Update neuron state for one time step.
        
        Args:
            I_syn: Synaptic current input (nA)
            dt_ms: Time step (ms)
            current_time_ms: Current simulation time (ms)
        
        Returns:
            bool: True if neuron spiked
        """
        # Check refractory period
        if current_time_ms - self.last_spike_time < self.refract_period:
            return False
        
        # Update membrane potential: Euler integration
        dV = (-(self.V - self.V_rest) + self.R * I_syn) / self.tau_m * dt_ms
        self.V += dV
        
        # Check for spike
        if self.V >= self.V_thresh:
            self.V = self.V_reset
            self.last_spike_time = current_time_ms
            self.spike_times.append(current_time_ms)
            return True
        
        return False
    
    def reset(self):
        """Reset neuron to resting state."""
        self.V = self.V_rest
        self.last_spike_time = -float('inf')
        self.spike_times = []


# Synaptic-drive calibration, shared by both spiking networks.
#
# One synapse at MID-RANGE conductance depolarises a resting neuron by this
# fraction of the threshold gap in a single timestep. At 0.10 that means ~10
# coincident mid-range inputs fire a neuron outright, while weaker synapses
# summate over several timesteps -- the regime in which input timing and
# synaptic weight both matter, which is the precondition for STDP to express
# anything.
#
# Module level (not a local) so consumers can check a proposed network size
# against it: a network with fewer than 1/SINGLE_SYNAPSE_DV_FRACTION inputs
# cannot reach threshold from a coincident volley at all.
SINGLE_SYNAPSE_DV_FRACTION = 0.10


class SpikingSynapseNetwork:
    """
    Spiking neural network with synaptic devices and STDP learning.

    Architecture:
        - Input layer: Poisson spike generators
        - Hidden layer: LIF neurons
        - Synapses: physical VisualSynapse devices

    What "physical device" means here, precisely (audit M16). The synapses are
    real `VisualSynapse` objects, and two parts of the device physics are
    genuinely in play:

      - the device's own G_min/G_max bounds clip every weight, so a fitted
        device's dynamic range constrains learning; and
      - the device's spontaneous relaxation (`decay_tau`) runs between spikes,
        so a fitted retention time causes learned weights to fade.

    The OPTICAL/ELECTRICAL drive terms of the ODE — alpha, beta, A_peak, B_peak
    and the wavelength curve — are deliberately NOT invoked: in a spiking
    network the weight is changed by the STDP rule, not by illuminating the
    device. Those parameters are carried on the synapse objects for
    inspection and for `rest()`, but they do not participate in `simulate()`.

    The docstring previously claimed the devices were driven by their ODE
    without qualification, while `apply_stdp` wrote `synapse.G` directly and
    nothing ever integrated anything.
    """
    
    def __init__(self, base_synapse_params, stdp_params, n_input=10, n_hidden=5, dt_ms=1.0):
        """
        Initialize spiking network.
        
        Args:
            base_synapse_params: Base parameters for synaptic devices
            stdp_params: STDP parameters {A_plus, A_minus, tau_plus_ms, tau_minus_ms}
            n_input: Number of input neurons
            n_hidden: Number of hidden neurons
            dt_ms: Simulation time step (ms)
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.dt_ms = dt_ms
        
        # STDP parameters
        self.A_plus = stdp_params.get('stdp_A_plus', 0.5)
        self.A_minus = stdp_params.get('stdp_A_minus', 0.3)
        self.tau_plus = stdp_params.get('stdp_tau_plus_ms', 20.0)
        self.tau_minus = stdp_params.get('stdp_tau_minus_ms', 20.0)
        
        # Create neurons
        self.hidden_neurons = [LIFNeuron() for _ in range(n_hidden)]

        # --- Synaptic coupling (audit C4) ---
        #
        # The drive used to be `I_syn += G * 1e9 * 0.1`, i.e. conductance in nS
        # times an unexplained 0.1. For a default device that is 3366 nA, and
        # through the LIF chain
        #     dV = (−(V − V_rest) + R·I_syn) / tau_m · dt
        # with R = 10 MΩ, tau_m = 20 ms, dt = 1 ms, that is
        #     dV = 0.05 · 10 · 3366 = 1683 mV
        # in a single timestep, against a threshold gap of only
        # V_thresh − V_rest = 20 mV. Every hidden neuron therefore fired on the
        # first timestep any single input fired. Verified: all five hidden
        # neurons emitted identical spike trains, so the post layer was a
        # deterministic OR of the input layer, with no neuron-specific timing
        # structure for STDP to read even in principle.
        #
        # A memristive device is not a biological synapse: at 10-100 µS it is
        # three to four orders of magnitude more conductive than the nS-scale
        # conductances a LIF neuron is parameterised for, so the device
        # conductance is MAPPED onto synaptic drive rather than used as a
        # literal membrane conductance. The mapping is fixed by an explicit
        # dynamic-range target rather than a magic constant:
        #
        #   one synapse at MID-RANGE conductance depolarises a resting neuron
        #   by SINGLE_SYNAPSE_DV_FRACTION of the threshold gap in one timestep.
        #
        # At 0.10 that means ~10 coincident mid-range inputs are needed to fire
        # a neuron outright, while weaker synapses summate over several
        # timesteps — a regime in which input timing and synaptic weight both
        # matter, which is the precondition for STDP to express anything.
        # Because the target is expressed relative to the device's own
        # G_min/G_max, a fitted model with a very different conductance scale
        # lands in the same dynamic regime automatically.
        neuron = self.hidden_neurons[0]
        threshold_gap_mV = neuron.V_thresh - neuron.V_rest
        G_mid = (base_synapse_params['G_min'] + base_synapse_params['G_max']) / 2.0

        target_dV_mV = SINGLE_SYNAPSE_DV_FRACTION * threshold_gap_mV
        # dV = (dt / tau_m) · R · I_syn  →  I_syn = dV · tau_m / (dt · R)
        target_I_nA = target_dV_mV * neuron.tau_m / (dt_ms * neuron.R)
        self.synaptic_gain_nA_per_S = target_I_nA / G_mid
        self.single_synapse_dv_fraction = SINGLE_SYNAPSE_DV_FRACTION

        # Create synaptic weight matrix (conductances)
        # Initialize with device variability
        try:
            from Main import VisualSynapse
        except (ImportError, ModuleNotFoundError):
            # Fallback: Define simple synapse class if Main.py unavailable
            class VisualSynapse:
                def __init__(self, G_min=1e-6, G_max=1e-4, **kwargs):
                    self.G_min = G_min
                    self.G_max = G_max
                    self.G = G_min * 1.5
        
        self.synapses = []
        # Mid-range, i.e. /2. This was /3 — undocumented, and indistinguishable
        # from a typo for /2 — which started every synapse a third of the way
        # up its range with no stated reason. Starting mid-range gives STDP
        # equal room to potentiate and depress, which is what the learning rule
        # assumes.
        G_init_mean = (base_synapse_params['G_min'] + base_synapse_params['G_max']) / 2

        # Device-to-device variability, from the caller rather than a hardcoded
        # 0.1 that ignored the GUI setting.
        variability = base_synapse_params.get('variability', 0.1)

        for i in range(n_input):
            synapse_row = []
            for j in range(n_hidden):
                # Add variability to initial conductances, clipped into the
                # physical bounds — an unclipped draw could start a synapse
                # outside [G_min, G_max].
                G_init = float(np.clip(
                    G_init_mean * (1 + np.random.normal(0, variability)),
                    base_synapse_params['G_min'],
                    base_synapse_params['G_max'],
                ))

                synapse = VisualSynapse(
                    G_min=base_synapse_params['G_min'],
                    G_max=base_synapse_params['G_max'],
                    alpha=base_synapse_params.get('alpha', 0.8),
                    beta=base_synapse_params.get('beta', 0.8),
                    lambda_peak=base_synapse_params.get('lambda_peak', 550),
                    lambda_width=base_synapse_params.get('lambda_width', 100),
                    decay_tau=base_synapse_params.get('decay_tau', 100),
                    A_peak=base_synapse_params.get('A_peak', 8e-4),
                    B_peak=base_synapse_params.get('B_peak', 6e-4)
                )
                synapse.G = G_init
                synapse_row.append(synapse)
            self.synapses.append(synapse_row)
        
        # Tracking
        self.input_spike_history = [[] for _ in range(n_input)]
        self.hidden_spike_history = [[] for _ in range(n_hidden)]
        self.weight_history = []
        self.time_points = []
        self.current_time = 0.0
    
    def get_weight_matrix(self):
        """Get current synaptic weight matrix (conductances in µS)."""
        W = np.zeros((self.n_input, self.n_hidden))
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                W[i, j] = self.synapses[i][j].G * 1e6  # Convert to µS
        return W

    # --- Parity with SynapseNetwork (audit C12) ---------------------------
    #
    # Main.py's Network tab reads n_synapses, get_statistics(), G_matrix and
    # rest() unconditionally right after constructing whichever network the
    # mode selector asked for. None of them existed here, so selecting
    # "spiking" and clicking Initialize Network raised AttributeError on the
    # first statement — the tab was dead on arrival in that mode. They are
    # implemented rather than special-cased in the GUI because each is a
    # meaningful property of a spiking network too.

    @property
    def n_synapses(self):
        return self.n_input * self.n_hidden

    @property
    def shape(self):
        return (self.n_input, self.n_hidden)

    @property
    def G_matrix(self):
        """Conductances in SIEMENS, matching SynapseNetwork.G_matrix.

        Note this is NOT get_weight_matrix(), which returns µS. Mixing the two
        is what made plot_weight_evolution draw values 1e6 too large.
        """
        G = np.zeros((self.n_input, self.n_hidden))
        for i in range(self.n_input):
            for j in range(self.n_hidden):
                G[i, j] = self.synapses[i][j].G
        return G

    def get_statistics(self):
        """Summary statistics over the synaptic conductances, in siemens."""
        G = self.G_matrix
        return {
            'mean_G': float(np.mean(G)),
            'std_G': float(np.std(G)),
            'min_G': float(np.min(G)),
            'max_G': float(np.max(G)),
            'n_synapses': self.n_synapses,
        }

    def rest(self, duration_ms):
        """Let every synapse relax with no stimulus.

        The devices are real VisualSynapse objects, so this exercises their
        spontaneous-decay term — the one part of the device ODE the spiking
        path does use.
        """
        for row in self.synapses:
            for synapse in row:
                if hasattr(synapse, 'rest'):
                    synapse.rest(duration_ms)
        self.current_time += duration_ms / 1000.0
    
    @property
    def stdp_window_ms(self):
        """Eligibility half-window, per CLAUDE.md: 5 × max(tau)."""
        return 5.0 * max(self.tau_plus, self.tau_minus)

    def apply_stdp(self, pre_idx, post_idx, delta_t_ms):
        """
        Apply the STDP learning rule to one synapse.

        delta_t_ms is t_post − t_pre:
            > 0  pre before post  → LTP
            < 0  post before pre  → LTD
            = 0  both in the same discretisation bin. Treated as the dt → 0+
                 limit of the LTP branch: the pre-spike is what drove the
                 neuron to threshold within that bin, so the pair is causal.
                 This case used to fall through BOTH branches and be silently
                 discarded — and since the input spike was recorded before the
                 neuron update, every genuinely causal pair landed here.
                 Measured on a live run: 465 zero-Δt pairs against 455
                 post-spikes, i.e. all of them, while the 4370 weight updates
                 that did happen came from acausal older spikes.

        A_plus and A_minus are PERCENTAGES of the dynamic range: the 0.01 below
        is part of the rule as CLAUDE.md specifies it, so A_plus = 0.5 means
        0.5% of (G_max − G_min). Do not add a second /100 anywhere.
        """
        synapse = self.synapses[pre_idx][post_idx]
        G_range = synapse.G_max - synapse.G_min

        if delta_t_ms >= 0:
            delta_G = self.A_plus * np.exp(-delta_t_ms / self.tau_plus) * G_range * 0.01
        else:
            delta_G = -self.A_minus * np.exp(delta_t_ms / self.tau_minus) * G_range * 0.01

        synapse.G = float(np.clip(synapse.G + delta_G, synapse.G_min, synapse.G_max))
        return delta_G
    
    def simulate(self, input_spike_trains, duration_ms, learning_enabled=True,
                 apply_device_decay=True):
        """
        Simulate network with spike trains and STDP learning.

        Args:
            input_spike_trains: List of lists, each containing spike times (ms) for one input neuron
            duration_ms: Simulation duration (ms)
            learning_enabled: Whether to apply STDP learning
            apply_device_decay: Whether the synapses relax toward G_min with
                their own decay_tau between spikes. On by default because these
                are physical devices and they do — see the class docstring.
                Switch off only to isolate the STDP rule from retention.

        Returns:
            dict: Simulation results
        """
        n_steps = int(duration_ms / self.dt_ms)
        
        # Reset neurons
        for neuron in self.hidden_neurons:
            neuron.reset()
        
        # Clear spike history for this simulation
        self.input_spike_history = [[] for _ in range(self.n_input)]
        self.hidden_spike_history = [[] for _ in range(self.n_hidden)]
        
        # Record initial weights
        self.weight_history = [self.get_weight_matrix().copy()]
        self.time_points = [0]
        
        # Simulation loop
        for step in range(n_steps):
            t = step * self.dt_ms
            
            # Check for input spikes at this time
            input_spikes = [False] * self.n_input
            for i in range(self.n_input):
                if input_spike_trains[i]:
                    # Check if any spike occurs in this time window
                    for spike_time in input_spike_trains[i]:
                        if abs(spike_time - t) < self.dt_ms / 2:
                            input_spikes[i] = True
                            self.input_spike_history[i].append(t)
                            break
            
            # --- LTD: a pre-spike now, paired against PAST post-spikes ---
            #
            # C2: STDP used to be applied only inside the post-synaptic
            # `if spiked:` block, pairing each new post-spike against past
            # pre-spikes as delta_t = t − t_pre. Since t_pre always came from
            # history, delta_t was never negative, the `elif delta_t < 0`
            # branch was dead code, and A_minus was never used at all.
            # Verified on a live run: 4370 positive pairs, 0 negative, and no
            # weight ever decreased. The eligibility window was effectively the
            # half-window [0, +100) rather than the spec's (−100, +100).
            #
            # This is the missing symmetric half. It runs BEFORE the neuron
            # update, so hidden_spike_history contains only spikes strictly
            # earlier than t — which is exactly the LTD pairing, and means the
            # same-bin causal pair cannot also be counted here. It is counted
            # once, as LTP, in the post-synaptic block below.
            if learning_enabled:
                for i in range(self.n_input):
                    if not input_spikes[i]:
                        continue
                    for j in range(self.n_hidden):
                        for t_post in self.hidden_spike_history[j]:
                            delta_t = t_post - t          # strictly negative
                            if -self.stdp_window_ms < delta_t < 0:
                                self.apply_stdp(i, j, delta_t)

            # Update hidden neurons
            hidden_spikes = [False] * self.n_hidden
            for j in range(self.n_hidden):
                # Synaptic drive from the inputs that spiked this step.
                I_syn = 0.0
                for i in range(self.n_input):
                    if input_spikes[i]:
                        I_syn += self.synapses[i][j].G * self.synaptic_gain_nA_per_S

                # Update neuron
                spiked = self.hidden_neurons[j].update(I_syn, self.dt_ms, t)
                if spiked:
                    hidden_spikes[j] = True
                    self.hidden_spike_history[j].append(t)

                    # --- LTP: this post-spike against recent pre-spikes ---
                    # delta_t >= 0 here by construction; the == 0 case is the
                    # same-bin causal pair (see apply_stdp).
                    if learning_enabled:
                        for i in range(self.n_input):
                            for t_pre in self.input_spike_history[i]:
                                delta_t = t - t_pre
                                if 0 <= delta_t < self.stdp_window_ms:
                                    self.apply_stdp(i, j, delta_t)


            # --- Device relaxation between spikes (audit M16) ---
            #
            # The synapses are real devices, so a learned weight decays toward
            # G_min with the device's own fitted decay_tau. Without this the
            # "physical device" claim was empty: a fitted retention time had no
            # effect on the network whatsoever, and weights persisted forever.
            #
            # Applied analytically over one timestep rather than by calling
            # rest(), which would append to each synapse's history array on
            # every step of every simulation.
            if apply_device_decay:
                dt_s = self.dt_ms / 1000.0
                for i in range(self.n_input):
                    for j in range(self.n_hidden):
                        syn = self.synapses[i][j]
                        tau = getattr(syn, 'decay_tau', None)
                        if not tau or tau <= 0:
                            continue
                        syn.G -= (syn.G - syn.G_min) / tau * dt_s

            # Record weights periodically
            if step % 10 == 0:
                self.weight_history.append(self.get_weight_matrix().copy())
                self.time_points.append(t)
        
        # Final weight recording
        self.weight_history.append(self.get_weight_matrix().copy())
        self.time_points.append(duration_ms)
        
        return {
            'input_spikes': self.input_spike_history,
            'hidden_spikes': self.hidden_spike_history,
            'weight_history': self.weight_history,
            'time_points': self.time_points,
            'final_weights': self.get_weight_matrix()
        }
    
    def generate_poisson_spikes(self, rate_hz, duration_ms, seed=None):
        """
        Generate Poisson spike train.
        
        Args:
            rate_hz: Firing rate (Hz)
            duration_ms: Duration (ms)
            seed: Random seed
        
        Returns:
            list: Spike times (ms)
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_expected = rate_hz * duration_ms / 1000.0
        n_spikes = np.random.poisson(n_expected)
        spike_times = sorted(np.random.uniform(0, duration_ms, n_spikes))
        
        return list(spike_times)


# =============================================================================
# MULTI-LAYER SPIKING NEURAL NETWORK
# =============================================================================

class MultiLayerSpikingNetwork:
    """
    Flexible multi-layer spiking neural network with configurable architecture.
    Supports feedforward and recurrent connections with STDP learning.
    """
    
    def __init__(self, layer_sizes, base_synapse_params, stdp_params, 
                 recurrent_layers=None, dt_ms=1.0):
        """
        Initialize multi-layer SNN.
        
        Args:
            layer_sizes: List of neurons per layer, e.g. [784, 100, 10] for 3 layers
            base_synapse_params: Base parameters for synaptic devices
            stdp_params: STDP parameters dict
            recurrent_layers: List of layer indices to add recurrent connections (e.g. [1, 2])
            dt_ms: Simulation time step (ms)
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.dt_ms = dt_ms
        self.recurrent_layers = recurrent_layers or []

        # STDP parameters
        self.stdp_params = stdp_params
        self.tau_plus = stdp_params.get('stdp_tau_plus_ms', 20.0)
        self.tau_minus = stdp_params.get('stdp_tau_minus_ms', 20.0)

        # Create neurons for each layer (except input layer 0)
        self.neurons = []
        for size in layer_sizes[1:]:
            self.neurons.append([LIFNeuron() for _ in range(size)])

        # Synaptic coupling, derived exactly as in SpikingSynapseNetwork — see
        # the long note there for why `G * 1e9 * 0.1` over-drove the neurons by
        # ~84x and made every post-synaptic layer a deterministic OR of its
        # input.
        neuron = self.neurons[0][0] if self.neurons else LIFNeuron()
        threshold_gap_mV = neuron.V_thresh - neuron.V_rest
        G_mid = (base_synapse_params['G_min'] + base_synapse_params['G_max']) / 2.0
        target_dV_mV = SINGLE_SYNAPSE_DV_FRACTION * threshold_gap_mV
        target_I_nA = target_dV_mV * neuron.tau_m / (dt_ms * neuron.R)
        self.synaptic_gain_nA_per_S = target_I_nA / G_mid
        self.single_synapse_dv_fraction = SINGLE_SYNAPSE_DV_FRACTION

        # Create feedforward synapse matrices
        self.ff_synapses = []
        for i in range(self.n_layers - 1):
            synapses = self._create_synapse_layer(
                layer_sizes[i], layer_sizes[i+1], base_synapse_params
            )
            self.ff_synapses.append(synapses)
        
        # Create recurrent synapse matrices for specified layers
        self.rec_synapses = {}
        for layer_idx in self.recurrent_layers:
            if 1 <= layer_idx < self.n_layers:  # Can't be input layer
                size = layer_sizes[layer_idx]
                synapses = self._create_synapse_layer(size, size, base_synapse_params)
                self.rec_synapses[layer_idx] = synapses
        
        # Spike history tracking
        self.spike_history = [[] for _ in range(sum(layer_sizes))]
        self.weight_histories = []
        self.current_time = 0.0
    
    def _create_synapse_layer(self, n_pre, n_post, params):
        """Create synapse matrix between two layers."""
        try:
            from Main import VisualSynapse
        except ImportError:
            # Fallback simple synapse
            class VisualSynapse:
                def __init__(self, G_min=1e-6, G_max=1e-4, **kwargs):
                    self.G_min = G_min
                    self.G_max = G_max
                    self.G = (G_min + G_max) / 3
        
        # Mid-range (/2), and variability from the caller — see the matching
        # note in SpikingSynapseNetwork.__init__.
        G_init_mean = (params['G_min'] + params['G_max']) / 2
        variability = params.get('variability', 0.1)
        synapses = []

        for i in range(n_pre):
            row = []
            for j in range(n_post):
                G_init = float(np.clip(
                    G_init_mean * (1 + np.random.normal(0, variability)),
                    params['G_min'], params['G_max'],
                ))
                s = VisualSynapse(
                    G_min=params['G_min'],
                    G_max=params['G_max'],
                    alpha=params.get('alpha', 0.8),
                    beta=params.get('beta', 0.8),
                    lambda_peak=params.get('lambda_peak', 550),
                    lambda_width=params.get('lambda_width', 100),
                    decay_tau=params.get('decay_tau', 100),
                    A_peak=params.get('A_peak', 8e-4),
                    B_peak=params.get('B_peak', 6e-4)
                )
                s.G = G_init
                row.append(s)
            synapses.append(row)
        
        return synapses
    
    def get_weight_matrix(self, layer_idx, recurrent=False):
        """
        Get synaptic weight matrix for specified layer.
        
        Args:
            layer_idx: Layer index (0 = input→hidden, 1 = hidden→output, etc.)
            recurrent: If True, get recurrent weights for this layer
        
        Returns:
            2D numpy array of weights in µS
        """
        if recurrent:
            if layer_idx not in self.rec_synapses:
                return None
            synapses = self.rec_synapses[layer_idx]
        else:
            synapses = self.ff_synapses[layer_idx]
        
        n_pre = len(synapses)
        n_post = len(synapses[0]) if synapses else 0
        W = np.zeros((n_pre, n_post))
        
        for i in range(n_pre):
            for j in range(n_post):
                W[i, j] = synapses[i][j].G * 1e6
        
        return W
    
    @property
    def stdp_window_ms(self):
        """Eligibility half-window, per CLAUDE.md: 5 x max(tau).

        This was hardcoded as 100 ms in three places, which is correct only
        while both time constants happen to be 20 ms; a fitted model with
        tau = 50 ms would have had its window silently truncated to two time
        constants.
        """
        return 5.0 * max(self.tau_plus, self.tau_minus)

    def apply_stdp(self, pre_idx, post_idx, delta_t_ms, synapses):
        """Apply STDP to a synapse. See SpikingSynapseNetwork.apply_stdp."""
        synapse = synapses[pre_idx][post_idx]
        G_range = synapse.G_max - synapse.G_min

        A_plus = self.stdp_params.get('stdp_A_plus', 0.5)
        A_minus = self.stdp_params.get('stdp_A_minus', 0.3)

        # delta_t == 0 is the causal same-bin pair and takes the LTP limit,
        # rather than falling through to a no-op.
        if delta_t_ms >= 0:
            delta_G = A_plus * np.exp(-delta_t_ms / self.tau_plus) * G_range * 0.01
        else:
            delta_G = -A_minus * np.exp(delta_t_ms / self.tau_minus) * G_range * 0.01

        synapse.G = float(np.clip(synapse.G + delta_G, synapse.G_min, synapse.G_max))
        return delta_G
    
    def simulate(self, input_spike_trains, duration_ms, learning_enabled=True):
        """
        Simulate multi-layer network with STDP.
        
        Args:
            input_spike_trains: List of spike trains for input layer
            duration_ms: Simulation duration (ms)
            learning_enabled: Enable STDP learning
        
        Returns:
            dict: {layer_spikes, weight_matrices, time_points}
        """
        n_steps = int(duration_ms / self.dt_ms)
        
        # Reset all neurons
        for layer in self.neurons:
            for neuron in layer:
                neuron.reset()
        
        # Initialize spike histories for all layers
        layer_spike_histories = [
            [[] for _ in range(size)] for size in self.layer_sizes
        ]
        
        # Copy input spikes to layer 0 history
        for i, spikes in enumerate(input_spike_trains):
            layer_spike_histories[0][i] = list(spikes)
        
        # Track weights
        weight_history = []
        time_points = []
        
        # Record initial weights
        weight_snapshot = {
            'feedforward': [self.get_weight_matrix(i) for i in range(self.n_layers - 1)],
            'recurrent': {k: self.get_weight_matrix(k, recurrent=True) 
                         for k in self.recurrent_layers}
        }
        weight_history.append(weight_snapshot)
        time_points.append(0)
        
        # Simulation loop
        for step in range(n_steps):
            t = step * self.dt_ms
            
            # Check which input neurons spike at this time
            input_active = [False] * self.layer_sizes[0]
            for i in range(self.layer_sizes[0]):
                for spike_t in input_spike_trains[i]:
                    if abs(spike_t - t) < self.dt_ms / 2:
                        input_active[i] = True
                        break
            
            # Forward propagate through layers
            for layer_idx in range(1, self.n_layers):
                neuron_layer = self.neurons[layer_idx - 1]
                
                for j in range(self.layer_sizes[layer_idx]):
                    # Feedforward current
                    I_syn = 0
                    
                    # From previous layer
                    if layer_idx == 1:
                        pre_active = input_active
                    else:
                        # Check for spikes in previous layer at this timestep
                        pre_active = [False] * self.layer_sizes[layer_idx - 1]
                        for k in range(self.layer_sizes[layer_idx - 1]):
                            recent_spikes = [s for s in layer_spike_histories[layer_idx - 1][k]
                                           if abs(s - t) < self.dt_ms / 2]
                            pre_active[k] = len(recent_spikes) > 0
                    
                    # Feedforward synaptic current
                    for i, active in enumerate(pre_active):
                        if active:
                            I_syn += (self.ff_synapses[layer_idx - 1][i][j].G
                                      * self.synaptic_gain_nA_per_S)

                    # Recurrent current (if this layer has recurrent connections)
                    if layer_idx in self.recurrent_layers:
                        for i in range(self.layer_sizes[layer_idx]):
                            if i != j:  # No self-connections
                                recent_spikes = [s for s in layer_spike_histories[layer_idx][i]
                                               if abs(s - t) < self.dt_ms / 2]
                                if recent_spikes:
                                    I_syn += (self.rec_synapses[layer_idx][i][j].G
                                              * self.synaptic_gain_nA_per_S)

                    # --- LTD: a pre-spike now against PAST post-spikes ---
                    # The symmetric half of the rule, missing here exactly as it
                    # was in SpikingSynapseNetwork. Runs before this neuron's
                    # update, so its history holds only strictly earlier spikes.
                    if learning_enabled:
                        window = self.stdp_window_ms
                        for i, active in enumerate(pre_active):
                            if not active:
                                continue
                            for t_post in layer_spike_histories[layer_idx][j]:
                                delta_t = t_post - t
                                if -window < delta_t < 0:
                                    self.apply_stdp(i, j, delta_t,
                                                    self.ff_synapses[layer_idx - 1])

                    # Update neuron
                    spiked = neuron_layer[j].update(I_syn, self.dt_ms, t)
                    
                    if spiked:
                        layer_spike_histories[layer_idx][j].append(t)
                        
                        # Apply STDP. The window is 5*max(tau), not a hardcoded
                        # 100 ms — see stdp_window_ms.
                        if learning_enabled:
                            window = self.stdp_window_ms

                            # Feedforward LTP: this post-spike against recent
                            # pre-spikes (delta_t >= 0 by construction).
                            for i in range(self.layer_sizes[layer_idx - 1]):
                                for t_pre in layer_spike_histories[layer_idx - 1][i]:
                                    delta_t = t - t_pre
                                    if 0 <= delta_t < window:
                                        self.apply_stdp(i, j, delta_t,
                                                       self.ff_synapses[layer_idx - 1])

                            # Recurrent STDP, both signs: a recurrent partner's
                            # spike may precede or follow this one.
                            if layer_idx in self.recurrent_layers:
                                for i in range(self.layer_sizes[layer_idx]):
                                    if i == j:
                                        continue
                                    for t_pre in layer_spike_histories[layer_idx][i]:
                                        delta_t = t - t_pre
                                        if abs(delta_t) < window:
                                            self.apply_stdp(i, j, delta_t,
                                                          self.rec_synapses[layer_idx])
            
            # Record weights periodically
            if step % 50 == 0:
                weight_snapshot = {
                    'feedforward': [self.get_weight_matrix(i).copy() 
                                   for i in range(self.n_layers - 1)],
                    'recurrent': {k: self.get_weight_matrix(k, recurrent=True).copy()
                                 for k in self.recurrent_layers}
                }
                weight_history.append(weight_snapshot)
                time_points.append(t)
        
        # Final weight recording
        weight_snapshot = {
            'feedforward': [self.get_weight_matrix(i).copy() for i in range(self.n_layers - 1)],
            'recurrent': {k: self.get_weight_matrix(k, recurrent=True).copy() 
                         for k in self.recurrent_layers}
        }
        weight_history.append(weight_snapshot)
        time_points.append(duration_ms)
        
        return {
            'layer_spikes': layer_spike_histories,
            'weight_history': weight_history,
            'time_points': time_points,
            'final_weights_ff': [self.get_weight_matrix(i) for i in range(self.n_layers - 1)],
            'final_weights_rec': {k: self.get_weight_matrix(k, recurrent=True) 
                                 for k in self.recurrent_layers}
        }


# =============================================================================
# DEMONSTRATION SEQUENCES
# =============================================================================

def demo_spatial_learning(base_params, shape=(3, 3)):
    """
    Demonstration: Show how spatial patterns create specific weight distributions.
    
    Args:
        base_params (dict): Base synapse parameters
        shape (tuple): Network shape (rows, cols)
    
    Returns:
        SynapseNetwork: The network after demo
    """
    print("\n" + "="*60)
    print(f"PHASE 3 DEMO: Spatial Pattern Learning ({shape[0]}×{shape[1]} Network)")
    print("="*60)
    
    # Create network
    print(f"\n1. Creating {shape[0]}×{shape[1]} synapse network...")
    network = SynapseNetwork(base_params, shape=shape, variability=0.15)
    print(f"   ✓ Network created with {network.n_synapses} synapses (15% device variability)")
    
    # Demo sequence - adapted to network size
    patterns_sequence = [
        ('Vertical Bar (Center)', SpatialPatterns.vertical_bar(shape, shape[1]//2), 5),
        ('Rest', None, 1),
        ('Horizontal Bar (Center)', SpatialPatterns.horizontal_bar(shape, shape[0]//2), 5),
        ('Rest', None, 1),
        ('Diagonal (Main)', SpatialPatterns.diagonal_main(shape), 5),
        ('Rest', None, 1),
        ('Cross', SpatialPatterns.cross(shape), 5),
    ]
    
    print("\n2. Applying spatial pattern sequence...")
    for pattern_name, pattern, n_pulses in patterns_sequence:
        if pattern is None:
            print(f"   → Rest period (500ms)")
            network.rest(500)
        else:
            print(f"   → {pattern_name} × {n_pulses} pulses")
            for _ in range(n_pulses):
                network.apply_spatial_pattern(
                    pattern, 
                    intensity_mW_cm2=20, 
                    wavelength_nm=550, 
                    duration_ms=100,
                    mode='potentiation',
                    pattern_name=pattern_name
                )
                network.rest(100)
    
    print("\n3. Final network statistics:")
    stats = network.get_statistics()
    print(f"   Mean conductance: {stats['mean_G']*1e6:.2f} µS")
    print(f"   Std deviation:    {stats['std_G']*1e6:.2f} µS")
    print(f"   Dynamic range:    {stats['min_G']*1e6:.2f} - {stats['max_G']*1e6:.2f} µS")
    
    print("\n" + "="*60)
    print("✓ Demo Complete! Plotting results...")
    print("="*60)
    
    return network


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test the network module with different sizes
    print("Testing Phase 3: Scalable Synapse Network Module")
    
    # Base synapse parameters
    base_params = {
        'G_min': 1e-6,
        'G_max': 1e-4,
        'alpha': 0.8,
        'beta': 0.8,
        'lambda_peak': 550,
        'lambda_width': 100,
        'decay_tau': 100
    }
    
    # Test with different network sizes
    test_shapes = [(3, 3), (5, 5), (7, 7)]
    
    for shape in test_shapes:
        print(f"\n{'='*60}")
        print(f"Testing with {shape[0]}×{shape[1]} network")
        print(f"{'='*60}")
        
        # Run demo
        network = demo_spatial_learning(base_params, shape=shape)
        
        # Visualize
        fig1 = plot_network_state(network)
        fig2 = plot_network_evolution(network)
        
        # Save figures with shape-specific names
        fig1.savefig(f'network_state_{shape[0]}x{shape[1]}.png', dpi=150, bbox_inches='tight')
        fig2.savefig(f'network_evolution_{shape[0]}x{shape[1]}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualizations for {shape[0]}×{shape[1]} network")
    
    plt.show()