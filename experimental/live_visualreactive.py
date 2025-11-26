# Disable warnings
import warnings
warnings.filterwarnings("ignore")

import os
from typing import List, Union, Optional, Tuple, Type
import click
import time
import copy

try:
    import dnnlib
except ImportError as e:
    # Add the module to the path
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_utils import gen_utils
# Import discriminator features for v8 mode
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np

import cv2
import PIL
from PIL import Image

import random
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
from torchvision import transforms
from torch.autograd import Variable

import legacy

from network_features import VGG16FeaturesNVIDIA

import mediapipe as mp


# ----------------------------------------------------------------------------


def parse_height(s: str = None) -> Union[int, Type[None]]:
    """Parse height argument."""
    if s is not None:
        if s == 'max':
            return s
        else:
            return int(s)
    return None


def parse_mix_layers(s: str, max_layers: int = 18) -> List[int]:
    """
    Parse mix layers argument for v6 mode.

    Args:
        s: Layer specification ('coarse', 'middle', 'fine', 'all', or range like '0-4')
        max_layers: Maximum number of layers in the model

    Returns:
        List of layer indices to mix
    """
    layer_groups = {
        'coarse': list(range(0, 4)),
        'middle': list(range(4, 8)),
        'fine': list(range(8, max_layers)),
        'all': list(range(0, max_layers))
    }

    if s in layer_groups:
        return layer_groups[s]
    else:
        # Parse as range (e.g., "0-4" or "2,5,7-9")
        layers = []
        for part in s.split(','):
            if '-' in part:
                start, end = part.split('-')
                layers.extend(range(int(start), int(end) + 1))
            else:
                layers.append(int(part))
        return [max(0, min(l, max_layers - 1)) for l in layers]


def setup_generator(network_pkl: str, device: str, cfg: Optional[str], anchor_latent_space: bool):
    """Set up the generator."""
    if cfg:
        try:
            network_pkl = gen_utils.resume_specs[cfg][network_pkl]
        except KeyError:
            pass  # Assume it's a local file or URL
    print('Loading Generator...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False).to(device)
    if anchor_latent_space:
        gen_utils.anchor_latent_space(G)
    return G


# TODO: change to setup_backbone, let's test with EfficientNet-B0, for example
def setup_vgg16(device: str):
    """Set up VGG16 feature extractor."""
    print('Loading VGG16 and its features...')
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    return VGG16FeaturesNVIDIA(vgg16).requires_grad_(False).to(device)

def setup_camera(demo_height: int, demo_width: Optional[int]):
    """Set up the camera and video dimensions."""
    height = demo_height
    width = int(4.0/3*demo_height) if demo_width is None else demo_width
    cam = cv2.VideoCapture(0)
    return cam, height, width

def setup_mediapipe():
    """Set up MediaPipe for hand tracking."""
    mp_hands = mp.solutions.hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    return mp_hands, mp_drawing, mp_drawing_styles


def setup_selfie_segmentation():
    """Set up MediaPipe for selfie segmentation (multi-class body parts)."""
    try:
        # MediaPipe SelfieSegmentation with multiclass model
        mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        return mp_selfie
    except Exception as e:
        print(f'Warning: Could not initialize selfie segmentation: {e}')
        return None


# ----------------------------------------------------------------------------
# Hand Tracking Utilities
# ----------------------------------------------------------------------------


class EMAFilter:
    """Exponential Moving Average filter for smoothing values."""

    def __init__(self, alpha: float = 0.15):
        """
        Initialize EMA filter.

        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive, lower = smoother.
        """
        self.alpha = alpha
        self.values = {}
        self.initialized = {}

    def update(self, name: str, new_value: float, decay_to_zero: bool = False) -> float:
        """
        Update and return smoothed value.

        Args:
            name: Name/key for the value being tracked
            new_value: New measurement
            decay_to_zero: If True and new_value is None/0, decay towards zero

        Returns:
            Smoothed value
        """
        if name not in self.initialized:
            self.values[name] = new_value if new_value is not None else 0.0
            self.initialized[name] = True
            return self.values[name]

        if new_value is None or (decay_to_zero and new_value == 0.0):
            # Decay towards zero
            self.values[name] = (1 - self.alpha) * self.values[name]
        else:
            # Normal EMA update
            self.values[name] = self.alpha * new_value + (1 - self.alpha) * self.values[name]

        return self.values[name]

    def reset(self):
        """Reset all tracked values."""
        self.values = {}
        self.initialized = {}


def get_hand_center(hand_landmarks) -> Tuple[float, float, float]:
    """
    Calculate the center of a hand from MediaPipe landmarks.

    Args:
        hand_landmarks: MediaPipe hand landmarks

    Returns:
        Tuple of (x, y, z) coordinates
    """
    return np.mean([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], axis=0)


def get_hand_angle(hand_landmarks) -> float:
    """
    Calculate hand orientation angle from base to middle finger.

    Args:
        hand_landmarks: MediaPipe hand landmarks

    Returns:
        Angle in radians
    """
    base = hand_landmarks.landmark[0]
    middle = hand_landmarks.landmark[9]
    dx = middle.x - base.x
    dy = middle.y - base.y
    return np.pi / 2 + np.arctan2(dy, dx)


def get_hand_distance_from_center(hand_center: Tuple[float, float],
                                   normalize: bool = True) -> float:
    """
    Calculate distance of hand center from screen center.

    Args:
        hand_center: (x, y) coordinates of hand center
        normalize: If True, normalize to 0-1 range

    Returns:
        Distance value
    """
    x = hand_center[0] - 0.5
    y = hand_center[1] - 0.5
    dist = np.sqrt(x ** 2 + y ** 2)

    if normalize:
        dist = dist * 4 * 2 ** 0.5  # Scale up for better range
        dist = max(0.0, min(dist, 1.0))  # Clamp to [0, 1]

    return dist


def calculate_hand_area(hand_landmarks) -> float:
    """
    Calculate hand area using trapezoidal rule on key points.

    Args:
        hand_landmarks: MediaPipe hand landmarks

    Returns:
        Hand area (0-1 range)
    """
    area_points = [hand_landmarks.landmark[i] for i in range(0, 21, 4)]
    hand_area = 0.0
    for i in range(len(area_points) - 1):
        hand_area += (area_points[i].x - area_points[i + 1].x) * (
                area_points[i].y + area_points[i + 1].y)
    hand_area += (area_points[-1].x - area_points[0].x) * (area_points[-1].y + area_points[0].y)
    hand_area = abs(hand_area) / 2
    return max(0.0, min(hand_area, 1.0))


def calculate_hand_openness(hand_landmarks) -> float:
    """
    Calculate how open/spread the hand is based on finger distances.

    Args:
        hand_landmarks: MediaPipe hand landmarks

    Returns:
        Openness value (0-1 range, 0=closed fist, 1=open palm)
    """
    # Get fingertip landmarks
    fingertips = [
        hand_landmarks.landmark[4],   # Thumb
        hand_landmarks.landmark[8],   # Index
        hand_landmarks.landmark[12],  # Middle
        hand_landmarks.landmark[16],  # Ring
        hand_landmarks.landmark[20]   # Pinky
    ]

    # Calculate average distance between consecutive fingertips
    distances = []
    for i in range(len(fingertips) - 1):
        dist = np.sqrt(
            (fingertips[i].x - fingertips[i+1].x) ** 2 +
            (fingertips[i].y - fingertips[i+1].y) ** 2
        )
        distances.append(dist)

    avg_distance = np.mean(distances)
    # Normalize: typical open hand has ~0.3 distance, closed has ~0.05
    openness = (avg_distance - 0.05) / (0.3 - 0.05)
    return max(0.0, min(openness, 1.0))


def draw_hand_landmarks(image, results, show_center: bool = False,
                       show_distance_line: bool = False):
    """
    Draw hand landmarks and optional visualization aids.

    Args:
        image: Image to draw on (will be modified)
        results: MediaPipe hand tracking results
        show_center: If True, show screen center and hand center
        show_distance_line: If True, draw line from screen to hand center
    """
    if not results.multi_hand_landmarks:
        return

    image.flags.writeable = True

    # Draw hand landmarks
    for hand_landmarks in results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS)

    # Draw center visualization if requested
    if show_center and results.multi_hand_landmarks:
        center_x, center_y = int(image.shape[1] / 2), int(image.shape[0] / 2)
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)  # Blue screen center

        hand_center = get_hand_center(results.multi_hand_landmarks[0])
        hand_center_x = int(hand_center[0] * image.shape[1])
        hand_center_y = int(hand_center[1] * image.shape[0])
        cv2.circle(image, (hand_center_x, hand_center_y), 5, (0, 255, 255), -1)  # Yellow hand center

        if show_distance_line:
            cv2.line(image, (center_x, center_y), (hand_center_x, hand_center_y),
                    (0, 255, 255), 2)


# ----------------------------------------------------------------------------


class CircleObject:
    MU: float = 0.995  # Friction factor; 1 is a "frictionless surface"
    RHO: float = 0.05  # Density of the circle to be used to calculate the mass

    def __init__(self, position: Union[list, tuple, np.ndarray], radius: int,
                 initial_velocity: Union[list, tuple, np.ndarray], screen_width: int = 1280,
                 screen_height: int = 720, color: Union[list, tuple] = (255, 0, 0)):
        # Utility function to convert input to numpy array
        def to_numpy_array(input_value, dtype=np.float32):
            if isinstance(input_value, (tuple, list)):
                return np.array(input_value, dtype=dtype)
            return input_value

        # Store the initial state
        self.initial_position = to_numpy_array(position, dtype=np.int32)  # Position vector (x, y)
        self.initial_velocity = to_numpy_array(initial_velocity)  # Velocity vector (x, y)

        # Set current position, velocity, and other attributes
        self.position = self.initial_position.copy()
        self.velocity = self.initial_velocity.copy()
        self.radius = radius
        self.color = tuple(color) if isinstance(color, np.ndarray) else color
        self.mass = self.RHO * np.pi * self.radius ** 2  # m = rho * A := density * area

        # Save the screen dimensions for spawning purposes
        self.screen_width = screen_width
        self.screen_height = screen_height

    def reset(self):
        # Reset the state of the circle to its initial state
        self.position = self.initial_position.copy()
        self.velocity = self.initial_velocity.copy()

    def update(self):
        self.velocity = self.velocity * self.MU  # Apply friction to reduce velocity
        self.position = self.position + self.velocity

        # Check for collisions with the edges of the screen and reflect velocity
        edge_bounce = 1.01  # Slight bounce factor
        if self.position[0] - self.radius <= 1e-2:
            self.velocity[0] = abs(self.velocity[0]) * edge_bounce
            self.position[0] = self.radius + 1e-2
        elif self.position[0] + self.radius >= self.screen_width - 1e-2:
            self.velocity[0] = -abs(self.velocity[0]) * edge_bounce
            self.position[0] = self.screen_width - self.radius - 1e-2

        if self.position[1] - self.radius <= 1e-2:
            self.velocity[1] = abs(self.velocity[1]) * edge_bounce
            self.position[1] = self.radius + 1e-2
        elif self.position[1] + self.radius >= self.screen_height - 1e-2:
            self.velocity[1] = -abs(self.velocity[1]) * edge_bounce
            self.position[1] = self.screen_height - self.radius - 1e-2

    def draw(self, image):
        cv2.circle(image, tuple([int(p) for p in self.position]), self.radius, self.color, -1)

    def check_collision(self, hand_position):
        distance = np.linalg.norm(self.position - hand_position)
        return 1e-2 < self.radius - distance

    # Method to check collision with another circle
    def collides_with(self, other_circle):
        distance = np.linalg.norm(self.position - other_circle.position)
        return 1e-2 < (self.radius + other_circle.radius) - distance

    # Method to handle collision response
    def handle_collision(self, other_circle, is_elastic: bool = True) -> None:
        # Save the original velocities
        original_self_velocity = self.velocity.copy()
        original_other_velocity = other_circle.velocity.copy()

        if is_elastic:
            # Simple elastic collision physics; note that if the masses are equal (i.e., equal areas), then this reduces to
            # self.velocity, other_circle.velocity = other_circle.velocity, self.velocity, but let's make it more general
            # See: https://en.wikipedia.org/wiki/Elastic_collision#Equations

            # Update velocities using the original values
            self.velocity = ((self.mass - other_circle.mass) / (
                        self.mass + other_circle.mass)) * original_self_velocity + \
                            ((2 * other_circle.mass) / (self.mass + other_circle.mass)) * original_other_velocity
            other_circle.velocity = ((2 * self.mass) / (self.mass + other_circle.mass)) * original_self_velocity + \
                                    ((other_circle.mass - self.mass) / (
                                                self.mass + other_circle.mass)) * original_other_velocity
        else:
            # Inelastic collision: https://en.wikipedia.org/wiki/Inelastic_collision#Perfectly_inelastic_collision
            self.velocity = other_circle.velocity = (self.mass * original_self_velocity + other_circle.mass * original_other_velocity) / (self.mass + other_circle.mass)

    def apply_separation_force(self, other_circle, separation_factor=0.1):
        direction = self.position - other_circle.position
        distance = np.linalg.norm(direction)
        if distance < self.radius + other_circle.radius:
            force = direction / distance * separation_factor
            self.velocity += force
            other_circle.velocity -= force

    def kinetic_energy(self):
        return 0.5 * self.mass * np.linalg.norm(self.velocity) ** 2


def is_overlapping(new_position, new_radius, existing_circles):
    for circle in existing_circles:
        distance = np.linalg.norm(np.array(new_position) - circle.position)
        if distance + 1e-2 < (new_radius + circle.radius):
            return True  # Overlap found
    return False  # No overlap


# Function to create a new circle
def create_circle(existing_circles, screen_width, screen_height, max_attempts=100):
    for _ in range(max_attempts):
        new_radius = random.randint(50, 125)  # Or some other logic to determine the radius
        new_position = np.array([random.randint(new_radius, screen_width - new_radius),
                                 random.randint(new_radius, screen_height - new_radius)])
        if not is_overlapping(new_position, new_radius, existing_circles):
            new_circle = CircleObject(
                position=new_position,
                radius=new_radius,
                initial_velocity=np.array([random.randint(-5, 5), random.randint(-5, 5)]),
                screen_width=screen_width, screen_height=screen_height,
                color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            existing_circles.append(new_circle)
            break


# ----------------------------------------------------------------------------


# Main processing functions (to be implemented)
def process_v0(frame, vgg16_features, G, static_w, layer, label):
    """
    Base visual-reactive interpolation: encode image w/VGG16 (for now),
    which yields a "fake" dlatent that we will use with the Generator. To
    enhance variability of the output, we will use style mixing with a 
    static latent.
    """
    fake_z = vgg16_features.get_layers_features(frame, layers=[layer])[0]
    fake_z = fake_z.view(1, 512, -1).mean(2)
    fake_w = gen_utils.z_to_dlatent(G, fake_z, label, 1.0)
    fake_w[:, 4:] = static_w[:, 4:]
    return fake_w


def process_v1(frame, vgg16_features, G, layer, label, device):
    """
    Same as v0, except now we separate the top half of the image to control
    the "coarse" latent features, the bottom left to control the "middle", and
    the bottom right to control the "fine" features of the fake latent vector
    prior to do the style mixing.
    """
    fake_z = vgg16_features.get_layers_features(frame, layers=[layer])[0]
    _n, _c, h, w = fake_z.shape
    coarse_fake_z = fake_z[:, :, :h // 2, :]
    middle_fake_z = fake_z[:, :, h // 2:, :w // 2]
    fine_fake_z = fake_z[:, :, h // 2:, w // 2:]
    coarse_fake_z = coarse_fake_z.reshape(1, G.z_dim, -1).mean(2)
    middle_fake_z = middle_fake_z.reshape(1, G.z_dim, -1).mean(2)
    fine_fake_z = fine_fake_z.reshape(1, G.z_dim, -1).mean(2)
    coarse_fake_w = gen_utils.z_to_dlatent(G, coarse_fake_z, label, 1.0)
    middle_fake_w = gen_utils.z_to_dlatent(G, middle_fake_z, label, 1.0)
    fine_fake_w = gen_utils.z_to_dlatent(G, fine_fake_z, label, 1.0)
    fake_w = torch.cat([coarse_fake_w[:, :4], middle_fake_w[:, 4:8], fine_fake_w[:, 8:]], dim=1)
    return fake_w


# State filters for different modes
_ema_v2 = EMAFilter(alpha=0.15)


def process_v2(G, latent, mp_hands, image, label, const_input: torch.Tensor = None,
               const_input_interpolation: torch.Tensor = None, show_landmarks: bool = False):
    """
    Corrupt the learned constants. For StyleGAN2, corrupt the constant input vector towards a random one.
    For StyleGAN3, change the learned affine transformation (translate, rotate, ...). One hand only.
    """
    image.flags.writeable = False
    results = mp_hands.process(image)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # Calculate hand features using helper functions
        angle = get_hand_angle(hand)
        x, y, z = get_hand_center(hand)
        x, y = x - 0.5, y - 0.5
        dist = get_hand_distance_from_center((x + 0.5, y + 0.5))
        hand_area = calculate_hand_area(hand)

        # Apply EMA filtering
        angle = _ema_v2.update('angle', angle)
        x = _ema_v2.update('x', x)
        y = _ema_v2.update('y', y)
        z = _ema_v2.update('z', z)
        dist = _ema_v2.update('dist', dist)
        hand_area = _ema_v2.update('hand_area', hand_area)
    else:
        # Decay to zero when no hand detected
        angle = _ema_v2.update('angle', 0.0, decay_to_zero=True)
        x = _ema_v2.update('x', 0.0, decay_to_zero=True)
        y = _ema_v2.update('y', 0.0, decay_to_zero=True)
        z = _ema_v2.update('z', 0.0, decay_to_zero=True)
        dist = _ema_v2.update('dist', 0.0, decay_to_zero=True)
        hand_area = _ema_v2.update('hand_area', 0.0, decay_to_zero=True)

    # Apply transformations to generator
    if hasattr(G.synthesis, 'input'):
        m = gen_utils.make_affine_transform(m=None, angle=angle, translate_x=x, translate_y=-y,
                                            scale_x=1/(1 + 3*hand_area), scale_y=1/(1 + 3*hand_area))
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))
    elif hasattr(G.synthesis, 'b4'):
        G.synthesis.b4.const.copy_(torch.from_numpy((1 - dist) * const_input + const_input_interpolation * dist))

    # Draw hand landmarks if requested
    if show_landmarks:
        draw_hand_landmarks(image, results)

    generated_image = gen_utils.z_to_img(G, latent, label, truncation_psi=0.7, noise_mode='const')[0]

    return generated_image, image

_ema_v3 = EMAFilter(alpha=0.15)


def process_v3(G, latent, mp_hands, image, label, components: torch.Tensor, show_landmarks: bool = False):
    """
    Let each finger position/distance to the hand center dictate how much to move in the
    Principal Component (PC) of the latent space of the Generator.
    """
    image.flags.writeable = False
    results = mp_hands.process(image)

    # Finger tip landmark indices
    FINGERTIPS = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}

    finger_distances = {}
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        hand_center = get_hand_center(hand)

        # Calculate distance from each fingertip to hand center
        for finger_name, landmark_idx in FINGERTIPS.items():
            fingertip = hand.landmark[landmark_idx]
            dist = np.sqrt((fingertip.x - hand_center[0]) ** 2 + (fingertip.y - hand_center[1]) ** 2)
            finger_distances[finger_name] = _ema_v3.update(finger_name, dist)
    else:
        # Decay to zero when no hand detected
        for finger_name in FINGERTIPS.keys():
            finger_distances[finger_name] = _ema_v3.update(finger_name, 0.0, decay_to_zero=True)

    # Create a copy of the latent to manipulate
    latent_manipulated = latent.clone()
    latent_pc = latent_manipulated @ components.float()

    # Use fingertip distances to move along principal components
    scale_factor = 2.0
    pc_adjustments = torch.zeros_like(latent_pc)
    for i, finger_name in enumerate(FINGERTIPS.keys()):
        pc_adjustments[0, i] = finger_distances[finger_name] * scale_factor

    # Apply the adjustments and project back to W space
    latent_manipulated = latent_manipulated + (pc_adjustments @ components.float().T)

    # Draw hand landmarks if requested
    if show_landmarks:
        draw_hand_landmarks(image, results)

    generated_image = gen_utils.w_to_img(G, latent_manipulated, truncation_psi=0.7)[0]

    return generated_image, image


def process_v4(G, latent, mp_hands, image, label, circles, show_landmarks: bool = False):
    """
    "Kinetic" interpolation: let the total momentum of the bouncing circles dictate the
    truncation psi that the Generator will use. In other words, its expressivity.
    """
    results = mp_hands.process(image)

    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks:
            image.flags.writeable = True
            index_fingertip = landmark.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            hand_pos = np.array([int(index_fingertip.x * image.shape[1]),
                                 int(index_fingertip.y * image.shape[0])], dtype=np.float32)
            cv2.circle(image, (
                int(index_fingertip.x * image.shape[1]), int(index_fingertip.y * image.shape[0])), 5, (0, 255, 0), -1)  # Green circle with radius 5

            for circle in circles:
                if circle.check_collision(hand_pos):
                    circle.velocity = - (hand_pos - circle.position) * 0.3

    # Update circles and check for collisions
    for i in range(len(circles)):
        circles[i].update()
        for j in range(i + 1, len(circles)):
            if circles[i].collides_with(circles[j]):
                circles[i].handle_collision(circles[j], False)

    # Draw circles
    for circle in circles:
        circle.draw(image)

    # Calculate total kinetic energy
    total_kinetic_energy = sum(circle.kinetic_energy() for circle in circles)
    # print(f"Total Kinetic Energy: {total_kinetic_energy:.2f} x 10^6")
    # Normalize kinetic energy to use as truncation psi
    # Assuming we want truncation_psi between 0.5 and 1.0
    max_energy = sum(0.5 * circle.mass * 25 for circle in circles)  # Assuming max velocity of 5 in each direction
    truncation_psi = total_kinetic_energy / max_energy
    # truncation_psi = min(max(truncation_psi, 0.5), 1.0)  # Clamp between 0.5 and 1.0

    # Generate image using the calculated truncation_psi
    w = G.mapping(latent, label, truncation_psi=truncation_psi)
    img = G.synthesis(w, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img[0].cpu().numpy()

    # Draw hand landmarks if requested
    if show_landmarks:
        draw_hand_landmarks(image, results)

    return img, image


_ema_v5 = EMAFilter(alpha=0.15)


def process_v5(G, w_base, w_coarse, w_fine, mp_hands, image, label, truncation_psi: float = 0.7,
               show_landmarks: bool = False):
    """
    Hand-based latent mixing: mix coarse and fine features from two static latent vectors into a base latent.
    - Hand distance from center (0-1) controls mixing strength of coarse features (layers 0-3)
    - Hand orientation (angle) controls mixing strength of fine features (layers 8+)
    """
    image.flags.writeable = False
    results = mp_hands.process(image)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # Calculate hand features using helper functions
        angle = get_hand_angle(hand)
        hand_center = get_hand_center(hand)
        dist = get_hand_distance_from_center(hand_center[:2])

        # Apply EMA filtering
        angle = _ema_v5.update('angle', angle)
        dist = _ema_v5.update('dist', dist)
    else:
        # Decay to zero when no hand detected
        angle = _ema_v5.update('angle', 0.0, decay_to_zero=True)
        dist = _ema_v5.update('dist', 0.0, decay_to_zero=True)

    # Create mixed latent
    w_mixed = w_base.clone()

    # Mix coarse features (layers 0-3) based on distance
    coarse_mix_strength = dist
    w_mixed[:, 0:4] = (1 - coarse_mix_strength) * w_base[:, 0:4] + coarse_mix_strength * w_coarse[:, 0:4]

    # Mix fine features (layers 8+) based on orientation (normalize angle to 0-1)
    fine_mix_strength = (angle % (2 * np.pi)) / (2 * np.pi)
    w_mixed[:, 8:] = (1 - fine_mix_strength) * w_base[:, 8:] + fine_mix_strength * w_fine[:, 8:]

    # Draw hand landmarks if requested
    if show_landmarks:
        draw_hand_landmarks(image, results, show_center=True, show_distance_line=True)

    # Generate the image
    generated_image = gen_utils.w_to_img(G, w_mixed, truncation_psi=truncation_psi)[0]

    return generated_image, image


_ema_v6 = EMAFilter(alpha=0.15)
_ema_v7 = EMAFilter(alpha=0.15)


def process_v6(G1, G2, latent, mp_hands, image, label, mix_layers: List[int],
               truncation_psi: float = 0.7, show_landmarks: bool = False):
    """
    Model Forging: Mix two models in real-time using two hands.
    - Two hands tracked independently
    - Distance between hands controls mixing strength
    - When hands close → models merge, when far → models separate
    - Mix specified layers from both models
    """
    image.flags.writeable = False
    results = mp_hands.process(image)

    # Default mixing strength (no hands = use model 1)
    mix_strength = 0.0
    hand1_center = None
    hand2_center = None

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)

        if num_hands >= 2:
            # Track both hands
            hand1 = results.multi_hand_landmarks[0]
            hand2 = results.multi_hand_landmarks[1]

            # Get hand centers
            h1_center = get_hand_center(hand1)
            h2_center = get_hand_center(hand2)

            # Calculate distance between hands (normalized)
            hand_distance = np.sqrt(
                (h1_center[0] - h2_center[0]) ** 2 +
                (h1_center[1] - h2_center[1]) ** 2
            )

            # Normalize distance to mixing strength
            # Close hands (distance ~ 0) = high mixing (1.0)
            # Far hands (distance ~ 1.4 diagonal) = low mixing (0.0)
            max_distance = np.sqrt(2)  # Diagonal of unit square
            mix_strength = 1.0 - min(hand_distance / max_distance, 1.0)

            # Apply EMA filtering for smooth transitions
            mix_strength = _ema_v6.update('mix_strength', mix_strength)

            hand1_center = h1_center
            hand2_center = h2_center

        elif num_hands == 1:
            # Only one hand: decay to zero mixing
            mix_strength = _ema_v6.update('mix_strength', 0.0, decay_to_zero=True)
            hand1_center = get_hand_center(results.multi_hand_landmarks[0])
    else:
        # No hands: decay to zero mixing
        mix_strength = _ema_v6.update('mix_strength', 0.0, decay_to_zero=True)

    # Generate w latents from both models
    w1 = G1.mapping(latent, label, truncation_psi=truncation_psi)
    w2 = G2.mapping(latent, label, truncation_psi=truncation_psi)

    # Mix the specified layers
    w_mixed = w1.clone()
    for layer_idx in mix_layers:
        if layer_idx < w1.shape[1]:
            w_mixed[:, layer_idx] = (1 - mix_strength) * w1[:, layer_idx] + mix_strength * w2[:, layer_idx]

    # Generate image using the base model's synthesis network with mixed latent
    img = G1.synthesis(w_mixed, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    generated_image = img[0].cpu().numpy()

    # Draw visualization
    if show_landmarks:
        image.flags.writeable = True

        # Draw hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS)

        # If two hands, draw connection line and mixing visualization
        if hand1_center is not None and hand2_center is not None:
            h1_x = int(hand1_center[0] * image.shape[1])
            h1_y = int(hand1_center[1] * image.shape[0])
            h2_x = int(hand2_center[0] * image.shape[1])
            h2_y = int(hand2_center[1] * image.shape[0])

            # Draw centers
            cv2.circle(image, (h1_x, h1_y), 8, (255, 0, 0), -1)  # Blue for model 1
            cv2.circle(image, (h2_x, h2_y), 8, (0, 0, 255), -1)  # Red for model 2

            # Draw connection line with thickness based on mixing strength
            thickness = max(1, int(mix_strength * 10))
            # Color gradient: blue to purple to red based on mix_strength
            b = int(255 * (1 - mix_strength))
            r = int(255 * mix_strength)
            cv2.line(image, (h1_x, h1_y), (h2_x, h2_y), (b, 0, r), thickness)

            # Draw mixing percentage text
            mix_percentage = int(mix_strength * 100)
            text = f"Mix: {mix_percentage}%"
            text_x = (h1_x + h2_x) // 2
            text_y = (h1_y + h2_y) // 2 - 20
            cv2.putText(image, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        elif hand1_center is not None:
            # Draw single hand center
            h1_x = int(hand1_center[0] * image.shape[1])
            h1_y = int(hand1_center[1] * image.shape[0])
            cv2.circle(image, (h1_x, h1_y), 8, (255, 0, 0), -1)

    return generated_image, image


def process_v7(G, G2, base_latent, mp_hands, image, label, truncation_psi: float = 0.7,
               show_landmarks: bool = False):
    """
    🎪 Latent Playground - A discovery-based multi-hand interaction mode.

    No instructions given - users must explore and discover what each gesture does.
    Multiple hands create emergent interactions. Mix everything. Play. Discover.

    Hint: Try different numbers of hands, positions, rotations, and openness...
    """
    image.flags.writeable = False
    results = mp_hands.process(image)

    # Default parameters
    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    # Base latent to manipulate
    w = G.mapping(base_latent, label, truncation_psi=truncation_psi)

    # Extract features from all detected hands
    hand_features = []
    if results.multi_hand_landmarks:
        for hand_idx, hand in enumerate(results.multi_hand_landmarks):
            center = get_hand_center(hand)
            angle = get_hand_angle(hand)
            dist_from_center = get_hand_distance_from_center(center[:2])
            openness = calculate_hand_openness(hand)

            # Apply EMA for smooth transitions
            center_x = _ema_v7.update(f'hand{hand_idx}_cx', center[0])
            center_y = _ema_v7.update(f'hand{hand_idx}_cy', center[1])
            angle = _ema_v7.update(f'hand{hand_idx}_angle', angle)
            dist = _ema_v7.update(f'hand{hand_idx}_dist', dist_from_center)
            openness = _ema_v7.update(f'hand{hand_idx}_open', openness)

            hand_features.append({
                'center': (center_x, center_y, center[2]),
                'angle': angle,
                'distance': dist,
                'openness': openness,
                'height': center_y  # Y position (0=top, 1=bottom)
            })
    else:
        # Decay all values when no hands detected
        for hand_idx in range(2):  # Support up to 2 hands
            _ema_v7.update(f'hand{hand_idx}_cx', 0.5, decay_to_zero=False)
            _ema_v7.update(f'hand{hand_idx}_cy', 0.5, decay_to_zero=False)
            _ema_v7.update(f'hand{hand_idx}_angle', 0.0, decay_to_zero=True)
            _ema_v7.update(f'hand{hand_idx}_dist', 0.0, decay_to_zero=True)
            _ema_v7.update(f'hand{hand_idx}_open', 0.5, decay_to_zero=False)

    # === EMERGENT INTERACTIONS (Users discover these) ===

    w_modified = w.clone()
    dynamic_psi = truncation_psi

    if num_hands == 0:
        # No hands: gentle drift (decay to base state)
        pass

    elif num_hands == 1:
        # Single hand: Navigate latent space
        h = hand_features[0]

        # Hand position affects latent direction (coarse layers)
        # Center position creates a directional vector
        dx = (h['center'][0] - 0.5) * 2  # -1 to 1
        dy = (h['center'][1] - 0.5) * 2

        # Distance from center affects intensity
        intensity = h['distance'] * 3.0

        # Angle affects which dimensions are modulated
        angle_factor = h['angle'] / (2 * np.pi)

        # Openness affects truncation psi (detail level)
        # Open hand = more details, closed = smoother
        dynamic_psi = truncation_psi * (0.5 + h['openness'])

        # Modify coarse layers based on position
        direction = torch.randn_like(w_modified[:, :4]) * intensity * 0.1
        w_modified[:, :4] += direction * dx

        # Modify middle layers based on angle
        middle_shift = torch.randn_like(w_modified[:, 4:8]) * intensity * 0.05
        w_modified[:, 4:8] += middle_shift * angle_factor

        # Modify fine layers based on distance
        fine_shift = torch.randn_like(w_modified[:, 8:]) * h['distance'] * 0.03
        w_modified[:, 8:] += fine_shift

    elif num_hands >= 2:
        # Two hands: Model/latent mixing mode
        h1, h2 = hand_features[0], hand_features[1]

        # Distance between hands controls mixing strength
        hand_dist = np.sqrt(
            (h1['center'][0] - h2['center'][0]) ** 2 +
            (h1['center'][1] - h2['center'][1]) ** 2
        )
        mix_strength = 1.0 - min(hand_dist / np.sqrt(2), 1.0)
        mix_strength = _ema_v7.update('two_hand_mix', mix_strength)

        # Average openness affects overall detail level
        avg_openness = (h1['openness'] + h2['openness']) / 2
        dynamic_psi = truncation_psi * (0.5 + avg_openness)

        # If we have G2, mix between models
        if G2 is not None:
            w2 = G2.mapping(base_latent, label, truncation_psi=dynamic_psi)

            # Mix based on hand distance
            # Also use hand heights to determine which layers to mix
            h1_layer_start = int(h1['height'] * G.mapping.num_ws)
            h2_layer_end = int(h2['height'] * G.mapping.num_ws)

            mix_start = min(h1_layer_start, h2_layer_end)
            mix_end = max(h1_layer_start, h2_layer_end)

            # Mix the determined layers
            for i in range(mix_start, min(mix_end, w.shape[1])):
                w_modified[:, i] = (1 - mix_strength) * w[:, i] + mix_strength * w2[:, i]
        else:
            # Without G2, create latent interpolation
            # Generate a second latent vector influenced by second hand
            offset = torch.randn_like(base_latent) * (h2['distance'] * 2.0)
            latent2 = base_latent + offset
            w2 = G.mapping(latent2, label, truncation_psi=dynamic_psi)

            # Mix based on angles and distances
            angle_diff = abs(h1['angle'] - h2['angle'])
            angle_mix = angle_diff / (2 * np.pi)

            # Selective layer mixing based on hand positions
            for i in range(w.shape[1]):
                layer_factor = i / w.shape[1]
                # Use hand heights to create zones of influence
                if h1['height'] < 0.5 and layer_factor < 0.5:
                    # Top hand affects early layers
                    w_modified[:, i] = (1 - mix_strength) * w[:, i] + mix_strength * w2[:, i]
                elif h2['height'] > 0.5 and layer_factor > 0.5:
                    # Bottom hand affects late layers
                    w_modified[:, i] = (1 - mix_strength * angle_mix) * w[:, i] + (mix_strength * angle_mix) * w2[:, i]

    # Regenerate with modified latent and dynamic truncation
    img = G.synthesis(w_modified, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    generated_image = img[0].cpu().numpy()

    # Visualization: Show cryptic, colorful feedback
    if show_landmarks:
        image.flags.writeable = True

        # Draw hand landmarks with custom colors per hand
        colors = [(255, 100, 100), (100, 100, 255), (100, 255, 100), (255, 255, 100)]

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS)

                # Draw hand center with unique color
                h = hand_features[hand_idx]
                cx = int(h['center'][0] * image.shape[1])
                cy = int(h['center'][1] * image.shape[0])
                color = colors[hand_idx % len(colors)]

                # Openness affects circle size
                radius = int(5 + h['openness'] * 15)
                cv2.circle(image, (cx, cy), radius, color, -1)

                # Draw distance trail (visual feedback for distance from center)
                center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
                alpha = int(h['distance'] * 255)
                cv2.line(image, (center_x, center_y), (cx, cy), color, 1)

            # If two hands, show connection
            if num_hands >= 2:
                h1, h2 = hand_features[0], hand_features[1]
                cx1 = int(h1['center'][0] * image.shape[1])
                cy1 = int(h1['center'][1] * image.shape[0])
                cx2 = int(h2['center'][0] * image.shape[1])
                cy2 = int(h2['center'][1] * image.shape[0])

                # Connection line with gradient
                hand_dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                max_dist = np.sqrt(image.shape[1] ** 2 + image.shape[0] ** 2)
                thickness = max(1, int((1 - hand_dist / max_dist) * 10))
                cv2.line(image, (cx1, cy1), (cx2, cy2), (200, 200, 0), thickness)

        # Show cryptic hint text
        hints = [
            "∞ hands shape reality ∞",
            "⟲ rotate for color ⟳",
            "⇄ distance is strength ⇆",
            "✋ open for detail 👊",
            "⚡ two hands unlock fusion ⚡"
        ]
        hint = hints[num_hands % len(hints)]
        cv2.putText(image, hint, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return generated_image, image


# Preprocessing for discriminator (v8 mode)
mean_disc = np.array([0.485, 0.456, 0.406])
std_disc = np.array([0.229, 0.224, 0.225])
preprocess_disc = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_disc, std_disc)])


def deprocess_disc(image_np: torch.Tensor) -> np.ndarray:
    """Deprocess discriminator output."""
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std_disc.reshape((1, 1, 3)) + mean_disc.reshape((1, 1, 3))
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (255 * image_np).astype('uint8')
    return image_np


def clip_disc(image_tensor: torch.Tensor) -> torch.Tensor:
    """Clamp per channel for discriminator."""
    for c in range(3):
        m, s = mean_disc[c], std_disc[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor


def process_v8(D_model, mp_selfie, image, body_parts: List[int],
               iterations: int = 5, lr: float = 1e-2,
               layers: List[str] = None, show_mask: bool = False):
    """
    🎭 Body Dream Mask - Real-time discriminator dreaming on segmented body parts.

    Uses MediaPipe selfie segmentation to mask specific body parts, then applies
    discriminator-based DeepDream to those regions in real-time.

    Args:
        D_model: Discriminator feature extractor
        mp_selfie: MediaPipe selfie segmentation model
        image: Camera frame (BGR)
        body_parts: List of segmentation class IDs to dream on
            0 - background, 1 - hair, 2 - body-skin, 3 - face-skin, 4 - clothes, 5 - others
        iterations: Number of gradient ascent steps (low for real-time)
        lr: Learning rate
        layers: Discriminator layers to use
        show_mask: Show segmentation mask visualization

    Returns:
        (dreamed_image, visualization_image)
    """
    if mp_selfie is None:
        # Fallback: return original image if segmentation not available
        return image, image

    # Default layers if not specified
    if layers is None:
        layers = ['b16_conv0']

    # Get segmentation
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_selfie.process(image_rgb)

    if results.segmentation_mask is None:
        return image, image

    # Get segmentation mask (256x256)
    seg_mask = results.segmentation_mask

    # Create binary mask for selected body parts
    # Note: MediaPipe's multiclass segmentation returns class IDs directly
    mask = np.zeros_like(seg_mask, dtype=np.uint8)
    for part_id in body_parts:
        mask[seg_mask == part_id] = 1

    # Upscale mask to match image resolution
    mask_upscaled = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_upscaled = mask_upscaled[:, :, np.newaxis]  # Add channel dimension

    # Convert image to PIL for preprocessing
    pil_image = Image.fromarray(image_rgb)

    # Preprocess for discriminator
    preprocessed = preprocess_disc(pil_image).unsqueeze(0).to(D_model.device)
    preprocessed = Variable(preprocessed, requires_grad=True)

    # Perform gradient ascent (lightweight version of DeepDream)
    for i in range(iterations):
        D_model.zero_grad()

        # Extract features
        features = D_model.get_layers_features(preprocessed, layers=layers)

        # Maximize activation (simple dream loss)
        loss = sum(layer.norm() for layer in features)
        loss.backward()

        # Gradient ascent
        avg_grad = np.abs(preprocessed.grad.data.cpu().numpy()).mean()
        norm_lr = lr / (avg_grad + 1e-7)
        preprocessed.data += norm_lr * preprocessed.grad.data
        preprocessed.data = clip_disc(preprocessed.data)
        preprocessed.grad.data.zero_()

    # Deprocess
    dreamed_np = deprocess_disc(preprocessed.cpu().data.numpy())
    dreamed_bgr = cv2.cvtColor(dreamed_np, cv2.COLOR_RGB2BGR)

    # Composite: blend dreamed region with original using mask
    output = image.copy().astype(np.float32)
    dreamed_float = dreamed_bgr.astype(np.float32)
    output = mask_upscaled * dreamed_float + (1 - mask_upscaled) * output
    output = output.astype(np.uint8)

    # Visualization
    if show_mask:
        # Show mask overlay
        vis_image = image.copy()
        # Colorize mask based on body parts
        colors = {
            0: (0, 0, 0),       # background - black
            1: (255, 0, 255),   # hair - magenta
            2: (0, 255, 0),     # body-skin - green
            3: (255, 255, 0),   # face-skin - cyan
            4: (0, 0, 255),     # clothes - red
            5: (255, 128, 0)    # others - orange
        }

        mask_colored = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
        for part_id in body_parts:
            mask_colored[seg_mask == part_id] = colors.get(part_id, (255, 255, 255))

        mask_colored_upscaled = cv2.resize(mask_colored, (image.shape[1], image.shape[0]))
        vis_image = cv2.addWeighted(vis_image, 0.7, mask_colored_upscaled, 0.3, 0)

        return output, vis_image

    return output, image


# ----------------------------------------------------------------------------


# Main loop function
def main_loop(G, vgg16_features, mp_hands, cam, height, width, display_height, device, layer, static_w,
              label, all_latents, const_input, const_input_interpolation, mode, verbose, show_landmarks, fps, mirror,
              w_base=None, w_coarse=None, w_fine=None, truncation_psi=0.7, G2=None, mix_layer_indices=None,
              D_model=None, mp_selfie=None, body_parts_list=None, dream_iterations=5, dream_layers_list=None):

    if mode == 'v3':
        # Get the principal components, if we use mode 'v3'
        z = torch.randn(10000, G.z_dim, device=device)
        w = G.mapping(z, label, truncation_psi=1.0)[:, 0].detach().cpu()
        scaler = StandardScaler()
        w_scaled = scaler.fit_transform(w)

        # Get the components
        pca = PCA(n_components=20)
        pca.fit(w_scaled)
        components = pca.components_

        # Scale back the components
        components = scaler.inverse_transform(components)
        components = torch.from_numpy(components).to(device).T

    if mode == 'v4':
        num_circles = 3
        circles = []
        for i in range(num_circles):
            # Spawn a circle at a time to (best) avoid collision
            create_circle(circles, int(4 / 3 * display_height), display_height)

    # Preprocess the image
    preprocess = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
    counter = 0
    c = 0
    start_time = time.time()
    recording_flag = False

    while cam.isOpened():
        ret_val, img = cam.read()
        if not ret_val:
            break

        # Resize the image, keeping it in BGR
        img = cv2.resize(img, (width, height))

        if mode == 'v0' or mode == 'v1':
            # Convert to RGB for processing in v0 and v1 modes
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
            frame = preprocess(img_tensor / 255.0)

        # Pass the arguments to the selected mode
        if mode == 'v0':
            fake_w = process_v0(frame, vgg16_features, G, static_w, layer, label)
            simg = gen_utils.w_to_img(G, fake_w, noise_mode='const')[0]
        elif mode == 'v1':
            fake_w = process_v1(frame, vgg16_features, G, layer, label, device)
            simg = gen_utils.w_to_img(G, fake_w, noise_mode='const')[0]
        elif mode == 'v2':
            latent = all_latents[c % len(all_latents)]
            simg, img = process_v2(
                G, latent, mp_hands, img, label, const_input,
                const_input_interpolation[c % len(const_input_interpolation)] if const_input_interpolation is not None else None,
                show_landmarks)
        elif mode == 'v3':
            dlatent = gen_utils.get_w_from_seed(G, device, 0, 1.0)
            simg, img = process_v3(
                G, dlatent, mp_hands, img, label, components, show_landmarks)
        elif mode == 'v4':
            simg, img = process_v4(G, all_latents[c % len(all_latents)], mp_hands, img, label, circles, show_landmarks)
        elif mode == 'v5':
            simg, img = process_v5(G, w_base, w_coarse, w_fine, mp_hands, img, label, truncation_psi, show_landmarks)
        elif mode == 'v6':
            latent = all_latents[c % len(all_latents)]
            simg, img = process_v6(G, G2, latent, mp_hands, img, label, mix_layer_indices, truncation_psi, show_landmarks)
        elif mode == 'v7':
            latent = all_latents[c % len(all_latents)]
            simg, img = process_v7(G, G2, latent, mp_hands, img, label, truncation_psi, show_landmarks)
        elif mode == 'v8':
            simg, img = process_v8(D_model, mp_selfie, img, body_parts_list,
                                   iterations=dream_iterations, layers=dream_layers_list,
                                   show_mask=show_landmarks)
        else:
            raise ValueError(f"Mode {mode} not recognized.")

        simg = cv2.cvtColor(simg, cv2.COLOR_BGR2RGB)

        img = cv2.flip(img, 1)

        # Ensure simg is in the right format (HWC) and dtype
        if isinstance(simg, torch.Tensor):
            simg = simg.cpu().numpy()
        if simg.dtype != np.uint8:
            simg = (simg * 255).clip(0, 255).astype(np.uint8)
        if simg.shape[0] == 3:
            simg = np.transpose(simg, (1, 2, 0))

        # Resize images to have the same height
        display_width = int(4 / 3 * display_height)
        img_display = cv2.resize(img, (display_width, display_height))
        simg_display = cv2.resize(simg, (display_height, display_height))

        # Concatenate images (both are now in BGR format)
        display_img = np.concatenate((img_display, simg_display), axis=1)

        cv2.imshow('Visuorreactive Demo', display_img)

        counter += 1
        c += 1
        if (time.time() - start_time) > 1 and verbose:
            print(f"FPS: {counter / (time.time() - start_time):0.2f}")
            counter = 0
            start_time = time.time()

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            if not recording_flag:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, fps, (display_width + display_height, display_height))
                recording_flag = True
            else:
                recording_flag = False
                out.release()

        if recording_flag:
            out.write(display_img)

    cam.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------


# TODO: Analyze latent space/variant to the proposed PCA https://openreview.net/pdf?id=SlzEll3EsKv
# TODO: Add hand tracking/normalization here: https://github.com/caillonantoine/hand_osc/blob/master/detect.py

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename: can be URL, local file, or the name of the model in torch_utils.gen_utils.resume_specs', required=True)
@click.option('--network2', 'network_pkl2', help='Second network for model mixing (v6 mode only)', default=None)
@click.option('--device', help='Device to use for image generation; using the CPU is slower than the GPU', type=click.Choice(['cpu', 'cuda']), default='cuda', show_default=True)
@click.option('--cfg', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of the network, used only if you want to use the pretrained models in torch_utils.gen_utils.resume_specs')
@click.option('--cfg2', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of second network (v6 mode only)', default=None)
# Synthesis options (feed a list of seeds or give the projected w to synthesize)
@click.option('--seed', type=click.INT, help='Random seed to use for static synthesized image', default=0, show_default=True)
@click.option('--coarse-seed', type=click.INT, help='Random seed for coarse features source (v5 mode only)', default=1, show_default=True)
@click.option('--fine-seed', type=click.INT, help='Random seed for fine features source (v5 mode only)', default=2, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.6, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)', default=None, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--new-center', type=gen_utils.parse_new_center, help='New center for the W latent space; a seed (int) or a path to a projected dlatent (.npy/.npz)', default=None)
@click.option('--mirror', is_flag=True, help='Mirror the synthesized image')
@click.option('--demo-height', type=int, help='Height of the demo window', default=360, show_default=True)
@click.option('--demo-width', type=int, help='Width of the demo window', default=None, show_default=True)
@click.option('--layer', type=str, help='Layer of the pre-trained VGG16 to use as the feature extractor', default='conv4_1', show_default=True)
# Mediapipe options
@click.option('--hands', 'hand_tracking', type=bool, help='Use hand tracking', default=True, show_default=True)
@click.option('--face', 'face_tracking', type=bool, help='Use face tracking', default=False, show_default=True)
@click.option('--body', 'body_tracking', type=bool, help='Use body tracking', default=False, show_default=True)
# How to set the fake dlatent
@click.option('--mode', type=click.Choice(['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8']), required=True)
@click.option('--body-parts', type=str, help='Body parts to dream on for v8 mode (comma-separated: 0=bg,1=hair,2=body-skin,3=face-skin,4=clothes,5=others)', default='4', show_default=True)
@click.option('--dream-iterations', type=int, help='Number of dream iterations for v8 mode (low for real-time)', default=5, show_default=True)
@click.option('--dream-layers', type=str, help='Discriminator layers for v8 dreaming (comma-separated)', default='b16_conv0', show_default=True)
@click.option('--mix-layers', type=str, help='Layers to mix for v6 mode (e.g., "coarse", "middle", "fine", "all", "0-4")', default='all', show_default=True)
# TODO: intermediate layers?
# Video options
@click.option('--display-height', type=parse_height, help="Height of the display window; if 'max', will use G.img_resolution", default=None, show_default=True)
@click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
@click.option('--fps', type=click.IntRange(min=1), help='Save the video with this framerate.', default=30, show_default=True)
@click.option('--compress', is_flag=True, help='Add flag to compress the final mp4 file with `ffmpeg-python` (same resolution, lower file size)')
# Extra parameters
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(),
                                                                                                                            '../out', 'videos'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='live_visual-reactive', show_default=True)
@click.option('--verbose', is_flag=True, help='Print FPS of the live interpolation ever second; plot the detected hands for `--v2`')
@click.option('--show-landmarks', is_flag=True, help='Show the detected hand landmarks for `--v2`')
def live_visual_reactive(
        ctx,
        network_pkl: str,
        network_pkl2: Optional[str],
        device: Optional[str],
        cfg: str,
        cfg2: Optional[str],
        seed: int,
        coarse_seed: int,
        fine_seed: int,
        truncation_psi: float,
        class_idx: int,
        noise_mode: str,
        new_center: Union[int, str],
        mirror: bool,
        demo_height: int,
        demo_width: int,
        layer: str,
        hand_tracking: bool,
        face_tracking: bool,
        body_tracking: bool,
        mode: str,
        body_parts: str,
        dream_iterations: int,
        dream_layers: str,
        mix_layers: str,
        display_height: Optional[int],
        anchor_latent_space: bool,
        fps: int,
        compress: bool,
        outdir: str,
        description: str,
        verbose: Optional[bool],
        show_landmarks: Optional[bool]):
    """Live Visual-Reactive interpolation. A camera/webcamera is needed to be accessed by OpenCV."""

    G = setup_generator(network_pkl, device, cfg, anchor_latent_space)

    # Load second generator for v6 or v7 mode
    if mode in ['v6', 'v7']:
        if network_pkl2 is not None:
            print('Loading second generator for model mixing...')
            G2 = setup_generator(network_pkl2, device, cfg2 if cfg2 else cfg, anchor_latent_space)
            # Check compatibility
            if G.img_resolution != G2.img_resolution:
                raise ValueError(f"Models must have same resolution. G1: {G.img_resolution}, G2: {G2.img_resolution}")
        else:
            if mode == 'v6':
                raise ValueError("v6 mode requires --network2 parameter")
            print('v7 mode: Running with single model (latent mixing only)')
            G2 = None
    else:
        G2 = None

    # Label, in case it's a class-conditional model
    class_idx = gen_utils.parse_class(G, class_idx, ctx)
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    vgg16_features = setup_vgg16(device) if mode in ['v0', 'v1'] else None
    cam, height, width = setup_camera(demo_height, demo_width)
    mp_hands, mp_drawing, mp_drawing_styles = setup_mediapipe() if mode in ['v2', 'v3', 'v4', 'v5', 'v6', 'v7'] else (None, None, None)

    # Setup discriminator and selfie segmentation for v8 mode
    if mode == 'v8':
        print('Loading discriminator for body dream masking...')
        D = gen_utils.load_network('D', network_pkl, cfg, device)
        from network_features import DiscriminatorFeatures
        D_model = DiscriminatorFeatures(D).requires_grad_(False).to(device)
        mp_selfie = setup_selfie_segmentation()
        body_parts_list = [int(x.strip()) for x in body_parts.split(',')]
        dream_layers_list = [x.strip() for x in dream_layers.split(',')]
        print(f'Body parts to dream on: {body_parts_list}')
        print(f'Dream layers: {dream_layers_list}')
    else:
        D_model = None
        mp_selfie = None
        body_parts_list = None
        dream_layers_list = None

    display_height = G.img_resolution if display_height is None or display_height == 'max' else display_height

    static_w = gen_utils.get_w_from_seed(G, device, seed, truncation_psi) if mode == 'v0' else None

    # Generate latent vectors for v5 mode
    if mode == 'v5':
        w_base = gen_utils.get_w_from_seed(G, device, seed, truncation_psi)
        w_coarse = gen_utils.get_w_from_seed(G, device, coarse_seed, truncation_psi)
        w_fine = gen_utils.get_w_from_seed(G, device, fine_seed, truncation_psi)
    else:
        w_base = None
        w_coarse = None
        w_fine = None

    # Setup for v6 mode (model mixing)
    if mode == 'v6':
        # Parse which layers to mix
        mix_layer_indices = parse_mix_layers(mix_layers, max_layers=G.mapping.num_ws)
        print(f'Mixing layers: {mix_layer_indices}')

        # Create noise loop for continuous variation
        num_frames = 900
        shape = [num_frames, 1, G.z_dim]
        all_latents = np.random.RandomState(seed).randn(*shape).astype(np.float32)
        all_latents = scipy.ndimage.gaussian_filter(all_latents, sigma=[3.0 * 30, 0, 0], mode='wrap')
        all_latents /= np.sqrt(np.mean(np.square(all_latents)))
        all_latents = torch.from_numpy(all_latents).to(device)
    else:
        mix_layer_indices = None

    # Setup for v7 mode (latent playground)
    if mode == 'v7':
        print('🎪 Entering the Latent Playground... Explore and discover!')
        # Create noise loop for continuous variation
        num_frames = 900
        shape = [num_frames, 1, G.z_dim]
        all_latents = np.random.RandomState(seed).randn(*shape).astype(np.float32)
        all_latents = scipy.ndimage.gaussian_filter(all_latents, sigma=[3.0 * 30, 0, 0], mode='wrap')
        all_latents /= np.sqrt(np.mean(np.square(all_latents)))
        all_latents = torch.from_numpy(all_latents).to(device)

    if mode in ['v2', 'v4']:
        num_frames = 900
        shape = [num_frames, 1, G.z_dim]
        all_latents = np.random.RandomState(seed).randn(*shape).astype(np.float32)
        all_latents = scipy.ndimage.gaussian_filter(all_latents, sigma=[3.0 * 30, 0, 0], mode='wrap')
        all_latents /= np.sqrt(np.mean(np.square(all_latents)))
        all_latents = torch.from_numpy(all_latents).to(device)

        if hasattr(G.synthesis, 'b4'):
            const_input = copy.deepcopy(G.synthesis.b4.const).cpu().numpy()
            const_input_interpolation = np.random.randn(num_frames, *const_input.shape).astype(
                np.float32)  # [num_frames, G.w_dim, 4, 4]
            const_input_interpolation = scipy.ndimage.gaussian_filter(const_input_interpolation,
                                                                      sigma=[fps, 0, 0, 0],
                                                                      mode='wrap')
            const_input_interpolation /= np.sqrt(np.mean(np.square(const_input_interpolation))) / 2
        else:
            const_input = None
            const_input_interpolation = None

    else:
        all_latents = None
        const_input = None
        const_input_interpolation = None

    main_loop(G, vgg16_features, mp_hands, cam, height, width, display_height, device,
              layer, static_w, label, all_latents, const_input, const_input_interpolation, mode, verbose, show_landmarks, fps, mirror,
              w_base, w_coarse, w_fine, truncation_psi, G2, mix_layer_indices,
              D_model, mp_selfie, body_parts_list, dream_iterations, dream_layers_list)


# ----------------------------------------------------------------------------


if __name__ == '__main__':
    live_visual_reactive()


# ----------------------------------------------------------------------------
