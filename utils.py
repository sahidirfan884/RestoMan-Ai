"""
utils.py
--------
Utility module for the Restaurant CCTV Monitoring System.

This module holds:
    1. Config          - central configuration (paths, thresholds, video source)
    2. TableZone        - dataclass describing a single table's rectangle zone
    3. Geometry helpers - point-in-rectangle checks, centroid extraction from
                          YOLO-Pose keypoints
    4. Time helpers     - formatting durations and timestamps consistently
      across the whole project

Keeping these in one place means every other module (detector, tracker,
dashboard, database, main) can import from here instead of repeating logic.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional
import os
import math
import json


# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
class Config:
    """
    Central configuration class for the whole application.

    Having a single Config class means all tunable parameters (model path,
    video source, thresholds, colors, etc.) live in one place instead of
    being scattered as magic numbers across multiple files.
    """

    # ---- Paths ----
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, "restaurant_monitor.db")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    # JSON file where user-defined table zones are saved (via select_zones.py
    # or the Streamlit "Zone Setup" tab). Both entry points read/write this
    # SAME file, so zones defined in either place apply everywhere.
    ZONES_CONFIG_PATH = os.path.join(BASE_DIR, "table_zones.json")

    # ---- Video source settings ----
    # VIDEO_SOURCE can be:
    #   0 (int)              -> default webcam
    #   "path/to/video.mp4"  -> a recorded CCTV file / demo clip
    #
    # A bundled demo clip is included at demo_videos/restaurant_demo.webm so
    # the app can be run and shown immediately without a webcam or CCTV feed.
    # Switch to 0 to use your live webcam instead.
    VIDEO_SOURCE = os.path.join(BASE_DIR, "demo_videos", "restaurant_demo.webm")
    # VIDEO_SOURCE = 0  # <- uncomment this (and comment the line above) for webcam

    # If VIDEO_SOURCE is a file (demo/CCTV recording) rather than a live
    # webcam, loop it back to the start automatically when it ends - handy
    # for continuous demos/presentations instead of the app just closing.
    LOOP_VIDEO_FILE = True

    # ---- YOLOv8 model settings ----
    MODEL_PATH = "yolov8n-pose.pt"   # nano pose model - fast, good for CPU/low-end GPU
    CONFIDENCE_THRESHOLD = 0.45      # minimum detection confidence to accept a person
    DEVICE = "cpu"                  # change to "cuda" if a compatible GPU is available

    # ---- Occupancy detection thresholds ----
    OCCUPIED_DEBOUNCE_SECONDS = 2.0  # person must be present this long before "Occupied"
    EMPTY_GRACE_SECONDS = 5.0        # table must be empty this long before logging exit

    # ---- Sitting-posture detection thresholds ----
    # Only SITTING people count as "customers occupying a table" - a waiter or
    # passerby standing briefly near/inside a zone should NOT count.
    SITTING_KNEE_ANGLE_THRESHOLD = 140.0   # degrees; below this = bent knee = sitting
    SITTING_ASPECT_RATIO_THRESHOLD = 1.6   # bbox height/width fallback when pose is unclear
    MIN_KEYPOINT_CONFIDENCE = 0.3          # minimum confidence to trust a keypoint

    # ---- Display colors (BGR format, since we use OpenCV) ----
    COLOR_OCCUPIED = (0, 0, 255)     # red
    COLOR_EMPTY = (0, 200, 0)        # green
    COLOR_SKELETON = (255, 200, 0)   # light blue for pose skeleton lines
    COLOR_BOX = (0, 255, 255)        # yellow for person bounding boxes
    COLOR_TEXT = (255, 255, 255)     # white text
    COLOR_PANEL_BG = (30, 30, 30)    # dark gray dashboard panel background

    # ---- Window / display settings ----
    WINDOW_NAME = "Restaurant CCTV Monitoring System"
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720

    @staticmethod
    def ensure_directories():
        """
        Make sure required folders (e.g., logs/) exist before the app runs.
        Called once at startup from main.py.
        """
        os.makedirs(Config.LOGS_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# TABLE ZONE DEFINITION
# --------------------------------------------------------------------------
@dataclass
class TableZone:
    """
    Represents a single restaurant table's monitored zone as a rectangle
    drawn over the camera frame.

    Attributes:
        table_id (int): Unique identifier for the table (e.g., 1, 2, 3...).
        x1, y1 (int):   Top-left corner of the rectangle in pixel coordinates.
        x2, y2 (int):   Bottom-right corner of the rectangle in pixel coordinates.
    """
    table_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    def contains_point(self, x: float, y: float) -> bool:
        """
        Check whether a given (x, y) point falls inside this table's zone.
        Used to test if a detected person's reference point (centroid or
        hip-midpoint) is standing/sitting within this table's area.
        """
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def top_left(self) -> Tuple[int, int]:
        """Return the (x1, y1) top-left corner as a tuple, for drawing."""
        return (self.x1, self.y1)

    def bottom_right(self) -> Tuple[int, int]:
        """Return the (x2, y2) bottom-right corner as a tuple, for drawing."""
        return (self.x2, self.y2)


# --------------------------------------------------------------------------
# PREDEFINED TABLE ZONES
# --------------------------------------------------------------------------
# NOTE: These coordinates are examples. Adjust them to match your actual
# camera frame by pausing on a frame and noting pixel coordinates of each
# table area (a helper script for this can be added later if needed).
TABLE_ZONES: List[TableZone] = [
    TableZone(table_id=1, x1=40,  y1=80,  x2=280, y2=340),
    TableZone(table_id=2, x1=320, y1=80,  x2=560, y2=340),
    TableZone(table_id=3, x1=600, y1=80,  x2=840, y2=340),
    TableZone(table_id=4, x1=880, y1=80,  x2=1120, y2=340),
]


def load_table_zones(path: str = None) -> List[TableZone]:
    """
    Load table zones from a JSON config file if it exists (created by
    select_zones.py or the Streamlit "Zone Setup" tab), otherwise fall
    back to the hardcoded TABLE_ZONES default above.

    Using a shared JSON file (rather than editing utils.py by hand) means
    main.py, streamlit_app.py, and select_zones.py all stay in sync.

    Args:
        path: filesystem path to the zones JSON file. Defaults to
            Config.ZONES_CONFIG_PATH.

    Returns:
        List[TableZone] - either the loaded custom zones, or the defaults.
    """
    if path is None:
        path = Config.ZONES_CONFIG_PATH

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            zones = [TableZone(**item) for item in data]
            if zones:
                return zones
        except (json.JSONDecodeError, TypeError, KeyError) as error:
            print(f"[Utils] Warning: failed to load zones from '{path}' ({error}). Using defaults.")

    return TABLE_ZONES


def save_table_zones(zones: List[TableZone], path: str = None) -> None:
    """
    Save a list of TableZone objects to a JSON config file, so other entry
    points (main.py, streamlit_app.py) pick them up automatically next time
    they call load_table_zones().

    Args:
        zones: list of TableZone objects to persist.
        path: filesystem path to write to. Defaults to Config.ZONES_CONFIG_PATH.
    """
    if path is None:
        path = Config.ZONES_CONFIG_PATH

    data = [
        {"table_id": z.table_id, "x1": z.x1, "y1": z.y1, "x2": z.x2, "y2": z.y2}
        for z in zones
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --------------------------------------------------------------------------
# GEOMETRY HELPERS
# --------------------------------------------------------------------------
def get_person_reference_point(
    bbox: Tuple[float, float, float, float],
    keypoints: Optional[List[Tuple[float, float, float]]] = None,
) -> Tuple[float, float]:
    """
    Determine the single (x, y) reference point that represents "where a
    person is standing/sitting", used to test against table zones.

    Preference order:
        1. Hip-midpoint from pose keypoints (COCO format: left_hip=11, right_hip=12)
           -> more accurate since it represents body position, not head or feet.
        2. Bounding box centroid (fallback if keypoints are missing/low confidence).

    Args:
        bbox: (x1, y1, x2, y2) bounding box of the detected person.
        keypoints: list of (x, y, confidence) tuples in COCO 17-keypoint order.

    Returns:
        (x, y) reference point in pixel coordinates.
    """
    if keypoints and len(keypoints) >= 13:
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        # Only use hip keypoints if both are confidently detected
        if left_hip[2] > 0.3 and right_hip[2] > 0.3:
            mid_x = (left_hip[0] + right_hip[0]) / 2
            mid_y = (left_hip[1] + right_hip[1]) / 2
            return (mid_x, mid_y)

    # Fallback: bounding box centroid
    x1, y1, x2, y2 = bbox
    centroid_x = (x1 + x2) / 2
    centroid_y = (y1 + y2) / 2
    return (centroid_x, centroid_y)


def find_zone_for_point(x: float, y: float, zones: List[TableZone]) -> Optional[int]:
    """
    Given a point and a list of table zones, return the table_id of the
    first zone that contains this point, or None if the point isn't inside
    any defined table zone (e.g., person walking in an aisle).
    """
    for zone in zones:
        if zone.contains_point(x, y):
            return zone.table_id
    return None


def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Calculate the angle ABC (in degrees) formed at vertex B by the two
    line segments B->A and B->C.

    Used to measure the knee-bend angle (hip -> knee -> ankle) to classify
    a person's posture as sitting (bent knee, smaller angle) or standing
    (straight leg, angle close to 180 degrees).

    Args:
        a, b, c: (x, y) points, where b is the vertex of the angle.

    Returns:
        Angle in degrees (0-180). Returns 180.0 (treated as "straight/standing")
        if the points are degenerate (zero-length vectors).
    """
    vec_ba = (a[0] - b[0], a[1] - b[1])
    vec_bc = (c[0] - b[0], c[1] - b[1])

    magnitude_ba = math.hypot(*vec_ba)
    magnitude_bc = math.hypot(*vec_bc)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 180.0

    dot_product = vec_ba[0] * vec_bc[0] + vec_ba[1] * vec_bc[1]
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    # Clamp to valid range for acos to avoid floating-point domain errors
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.degrees(math.acos(cos_angle))


def is_sitting_posture(
    bbox: Tuple[float, float, float, float],
    keypoints: Optional[List[Tuple[float, float, float]]] = None,
    knee_angle_threshold: float = Config.SITTING_KNEE_ANGLE_THRESHOLD,
    aspect_ratio_threshold: float = Config.SITTING_ASPECT_RATIO_THRESHOLD,
    min_confidence: float = Config.MIN_KEYPOINT_CONFIDENCE,
) -> bool:
    """
    Classify whether a detected person is SITTING (as opposed to standing).

    This matters because only seated people should count as "customers
    occupying a table" - a waiter or passerby briefly standing inside a
    table's zone should NOT be counted as a customer.

    Method (in priority order):
        1. POSE-BASED: Compute the knee-bend angle using hip->knee->ankle
           keypoints. A bent knee (angle < knee_angle_threshold) indicates
           a sitting posture. Checks the left leg first, then the right leg,
           using whichever has confident keypoints.
        2. FALLBACK (bounding-box aspect ratio): If pose keypoints for the
           legs aren't confidently detected (e.g., legs hidden behind a
           table), fall back to the bounding box's height/width ratio.
           A standing person's box is tall and narrow; a seated person's
           visible box (often just torso+head above a table) is comparatively
           shorter/wider.

    Args:
        bbox: (x1, y1, x2, y2) bounding box of the detected person.
        keypoints: 17 COCO keypoints as (x, y, confidence) tuples.
        knee_angle_threshold: degrees; below this = bent knee = sitting.
        aspect_ratio_threshold: fallback height/width ratio cutoff.
        min_confidence: minimum keypoint confidence to trust it.

    Returns:
        True if the person is classified as sitting, False if standing.
    """
    if keypoints and len(keypoints) >= 17:
        left_hip, right_hip = keypoints[11], keypoints[12]
        left_knee, right_knee = keypoints[13], keypoints[14]
        left_ankle, right_ankle = keypoints[15], keypoints[16]

        # Try the left leg first if all three joints are confidently detected.
        if (left_hip[2] > min_confidence and left_knee[2] > min_confidence
                and left_ankle[2] > min_confidence):
            angle = calculate_angle(
                (left_hip[0], left_hip[1]),
                (left_knee[0], left_knee[1]),
                (left_ankle[0], left_ankle[1]),
            )
            return angle < knee_angle_threshold

        # Otherwise try the right leg.
        if (right_hip[2] > min_confidence and right_knee[2] > min_confidence
                and right_ankle[2] > min_confidence):
            angle = calculate_angle(
                (right_hip[0], right_hip[1]),
                (right_knee[0], right_knee[1]),
                (right_ankle[0], right_ankle[1]),
            )
            return angle < knee_angle_threshold

    # FALLBACK: neither leg's keypoints were confident enough (common when
    # a person is seated behind a table and their legs are occluded).
    # Use bounding-box aspect ratio as a rough secondary signal.
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1e-6)
    height = y2 - y1
    aspect_ratio = height / width

    return aspect_ratio < aspect_ratio_threshold


# --------------------------------------------------------------------------
# TIME HELPERS
# --------------------------------------------------------------------------
def format_duration(seconds: float) -> str:
    """
    Convert a duration in seconds into a clean human-readable string,
    e.g., 125.3 -> "2m 5s", used for "time since occupied" display.
    """
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def current_timestamp() -> str:
    """
    Return the current time as a standardized string, used consistently
    for database entries and CSV reports.
    Format: YYYY-MM-DD HH:MM:SS
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def current_date_str() -> str:
    """
    Return today's date as YYYY-MM-DD, used for naming daily CSV report
    files (e.g., daily_report_2026-07-24.csv).
    """
    return datetime.now().strftime("%Y-%m-%d")
