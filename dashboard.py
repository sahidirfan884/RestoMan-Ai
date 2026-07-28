"""
dashboard.py
------------
Handles all visual rendering onto video frames for the Restaurant CCTV
Monitoring System:
    - Table zone rectangles (red = occupied, green = empty)
    - Table info labels (Table ID, status, people count, occupied duration)
    - Person bounding boxes and pose skeletons
    - A summary dashboard panel (total/occupied/empty tables, total customers)

Kept separate from detection/tracking logic so all "how things look on
screen" decisions live in one place - if you want to restyle the UI, you
only touch this file.
"""

from typing import List
import cv2
import numpy as np

from utils import Config, TableZone
from tracker import TableStatus
from detector import PersonDetection, SKELETON_CONNECTIONS


class Dashboard:
    """
    Renders all visual overlays (table zones, people, skeletons, and the
    summary stats panel) onto a video frame using OpenCV drawing functions.
    """

    def __init__(self, table_zones: List[TableZone]):
        """
        Args:
            table_zones: the list of TableZone objects, used to know where
                to draw each table's rectangle and label.
        """
        self.table_zones = table_zones

    # ----------------------------------------------------------------
    # TABLE ZONE DRAWING
    # ----------------------------------------------------------------
    def draw_table_zones(self, frame: np.ndarray, statuses: List[TableStatus]) -> np.ndarray:
        """
        Draw every table's rectangle zone onto the frame, colored red if
        occupied or green if empty, along with a text label showing:
        Table ID, Occupied/Empty, number of people, and time since occupied.

        Args:
            frame: the video frame to draw onto (modified in place & returned).
            statuses: list of TableStatus objects (one per table) from tracker.py.

        Returns:
            The same frame, with table zones drawn on it.
        """
        # Build a quick lookup from table_id -> zone, so we can match each
        # status to its rectangle coordinates.
        zone_lookup = {zone.table_id: zone for zone in self.table_zones}

        for status in statuses:
            zone = zone_lookup.get(status.table_id)
            if zone is None:
                continue

            color = Config.COLOR_OCCUPIED if status.occupied else Config.COLOR_EMPTY

            # Draw the rectangle for this table's zone.
            cv2.rectangle(frame, zone.top_left(), zone.bottom_right(), color, thickness=2)

            # Build the label lines to display above/inside the rectangle.
            state_text = "OCCUPIED" if status.occupied else "EMPTY"
            label_lines = [
                f"Table {status.table_id}: {state_text}",
                f"People: {status.num_people}",
            ]
            if status.occupied and status.duration_str:
                label_lines.append(f"Time: {status.duration_str}")

            self._draw_label_block(frame, zone.x1, zone.y1, label_lines, color)

        return frame

    def _draw_label_block(self, frame: np.ndarray, x: int, y: int, lines: List[str], color: tuple):
        """
        Draw a small semi-opaque background box with multiple lines of text,
        anchored above a given (x, y) point. Used for per-table labels.

        Args:
            frame: frame to draw on.
            x, y: anchor point (top-left corner of the table zone).
            lines: list of text strings, one per line.
            color: BGR color tuple used for the background box border.
        """
        line_height = 20
        padding = 6
        box_width = 190
        box_height = line_height * len(lines) + padding * 2

        # Position the label box just above the zone's top edge; if there's
        # no room above (near top of frame), draw it just below instead.
        label_y = y - box_height - 4
        if label_y < 0:
            label_y = y + 4

        # Semi-transparent dark background rectangle for readability.
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, label_y),
            (x + box_width, label_y + box_height),
            (20, 20, 20),
            thickness=-1,
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (x, label_y), (x + box_width, label_y + box_height), color, thickness=1)

        # Draw each line of text.
        for i, line in enumerate(lines):
            text_y = label_y + padding + (i + 1) * line_height - 6
            cv2.putText(
                frame, line, (x + padding, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_TEXT, 1, cv2.LINE_AA
            )

    # ----------------------------------------------------------------
    # PERSON / SKELETON DRAWING
    # ----------------------------------------------------------------
    def draw_people(self, frame: np.ndarray, detections: List[PersonDetection]) -> np.ndarray:
        """
        Draw bounding boxes and pose skeletons for every detected person.

        Args:
            frame: the video frame to draw onto.
            detections: list of PersonDetection objects from detector.py.

        Returns:
            The same frame, with people drawn on it.
        """
        for detection in detections:
            x1, y1, x2, y2 = [int(v) for v in detection.bbox]

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), Config.COLOR_BOX, thickness=2)

            # Confidence label above the box
            conf_text = f"{detection.confidence * 100:.0f}%"
            cv2.putText(
                frame, conf_text, (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, Config.COLOR_BOX, 1, cv2.LINE_AA
            )

            # Skeleton: draw a line between each connected keypoint pair,
            # but only if both keypoints have acceptable confidence.
            keypoints = detection.keypoints
            if keypoints:
                for (idx_a, idx_b) in SKELETON_CONNECTIONS:
                    if idx_a < len(keypoints) and idx_b < len(keypoints):
                        xa, ya, ca = keypoints[idx_a]
                        xb, yb, cb = keypoints[idx_b]
                        if ca > Config.MIN_KEYPOINT_CONFIDENCE and cb > Config.MIN_KEYPOINT_CONFIDENCE:
                            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), Config.COLOR_SKELETON, 2)

                # Draw a small dot at each confident keypoint.
                for (kx, ky, kc) in keypoints:
                    if kc > Config.MIN_KEYPOINT_CONFIDENCE:
                        cv2.circle(frame, (int(kx), int(ky)), 3, Config.COLOR_SKELETON, -1)

        return frame

    # ----------------------------------------------------------------
    # SUMMARY DASHBOARD PANEL
    # ----------------------------------------------------------------
    def draw_summary_panel(self, frame: np.ndarray, summary: dict) -> np.ndarray:
        """
        Draw a clean summary panel in the top-left corner of the frame
        showing: total tables, occupied tables, empty tables, total customers.

        Args:
            frame: the video frame to draw onto.
            summary: dict from TableTracker.get_summary(), with keys
                total_tables, occupied_tables, empty_tables, total_customers.

        Returns:
            The same frame, with the summary panel drawn on it.
        """
        panel_x, panel_y = 10, 10
        panel_width, panel_height = 260, 130

        # Semi-transparent background panel.
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            Config.COLOR_PANEL_BG,
            thickness=-1,
        )
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(
            frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
            (255, 255, 255), thickness=1
        )

        # Title
        cv2.putText(
            frame, "RESTAURANT DASHBOARD", (panel_x + 12, panel_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.line(frame, (panel_x + 10, panel_y + 32), (panel_x + panel_width - 10, panel_y + 32), (100, 100, 100), 1)

        # Stat lines
        stats = [
            ("Total Tables", summary["total_tables"], (255, 255, 255)),
            ("Occupied", summary["occupied_tables"], Config.COLOR_OCCUPIED),
            ("Empty", summary["empty_tables"], Config.COLOR_EMPTY),
            ("Total Customers", summary["total_customers"], (0, 255, 255)),
        ]

        for i, (label, value, color) in enumerate(stats):
            line_y = panel_y + 55 + i * 22
            text = f"{label}: {value}"
            cv2.putText(
                frame, text, (panel_x + 14, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

        return frame

    # ----------------------------------------------------------------
    # MASTER RENDER FUNCTION
    # ----------------------------------------------------------------
    def render(
        self,
        frame: np.ndarray,
        detections: List[PersonDetection],
        statuses: List[TableStatus],
        summary: dict,
    ) -> np.ndarray:
        """
        Convenience method that applies all drawing steps in the correct
        order for a single frame: people/skeletons -> table zones -> summary
        panel on top. Called once per frame from main.py.

        Args:
            frame: the raw video frame (BGR, from OpenCV).
            detections: list of PersonDetection objects for this frame.
            statuses: list of TableStatus objects for this frame.
            summary: dashboard summary dict for this frame.

        Returns:
            The fully annotated frame, ready to display or save.
        """
        frame = self.draw_people(frame, detections)
        frame = self.draw_table_zones(frame, statuses)
        frame = self.draw_summary_panel(frame, summary)
        return frame
