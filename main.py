"""
main.py
-------
Entry point for the Restaurant CCTV Monitoring System.

This module wires together every other module into a single application:
    detector.py   -> detects people + poses per frame
    tracker.py     -> maps detections to table zones, manages occupancy state
    database.py    -> logs entry/exit events to SQLite, exports CSV reports
    dashboard.py   -> draws all overlays onto the video frame
    utils.py       -> shared config, table zones, helper functions

Run this file directly to start the monitoring application:
    python main.py

Press 'q' to quit, or 'e' to export today's CSV report on demand.
"""

import sys
import time
import cv2

from utils import Config, load_table_zones, current_date_str
from detector import PersonDetector
from tracker import TableTracker
from database import DatabaseManager
from dashboard import Dashboard


class RestaurantMonitorApp:
    """
    Top-level application class. Owns the video capture loop and
    coordinates the detector, tracker, database, and dashboard components.

    Wrapping everything in a class (rather than free-floating script code)
    keeps state (e.g., open video capture, DB connection, pending log IDs)
    organized and makes the app easier to extend or test.
    """

    def __init__(self, video_source=Config.VIDEO_SOURCE):
        """
        Initialize every component of the application.

        Args:
            video_source: 0 for webcam, or a filepath string for a CCTV
                video file. Defaults to Config.VIDEO_SOURCE.
        """
        Config.ensure_directories()

        print("[App] Initializing Restaurant CCTV Monitoring System...")

        # Core components
        self.detector = PersonDetector()
        self.database = DatabaseManager()
        # Load table zones from the shared JSON config (written by
        # select_zones.py or the Streamlit "Zone Setup" tab), falling back
        # to the hardcoded defaults in utils.py if no config file exists yet.
        self.table_zones = load_table_zones(Config.ZONES_CONFIG_PATH)
        self.dashboard = Dashboard(self.table_zones)

        # Keep track of the open DB log-row id for each table that is
        # currently occupied, so we know which row to update on exit.
        self._open_log_ids = {}

        # Wire up the tracker's callbacks to the database, using small
        # wrapper methods (see _handle_table_occupied / _handle_table_vacated
        # below). This is dependency injection - tracker.py doesn't need
        # to know database.py exists at all.
        self.tracker = TableTracker(
            table_zones=self.table_zones,
            occupied_debounce_seconds=Config.OCCUPIED_DEBOUNCE_SECONDS,
            empty_grace_seconds=Config.EMPTY_GRACE_SECONDS,
            on_occupied=self._handle_table_occupied,
            on_vacated=self._handle_table_vacated,
        )

        # Video capture setup
        self.video_source = video_source
        self.capture = cv2.VideoCapture(video_source)
        if not self.capture.isOpened():
            raise RuntimeError(
                f"[App] ERROR: Could not open video source '{video_source}'. "
                f"Check your webcam index or video file path in utils.Config."
            )

        # Try to set a consistent frame size (works for webcams; ignored
        # gracefully for video files with a fixed native resolution).
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

        print("[App] Initialization complete. Starting video stream...")

    # ----------------------------------------------------------------
    # DATABASE CALLBACKS (wired into TableTracker)
    # ----------------------------------------------------------------
    def _handle_table_occupied(self, table_id: int, entry_timestamp: float):
        """
        Called by TableTracker the moment a table is confirmed Occupied.
        Inserts a new open log row (exit_time still NULL) and remembers
        its row id so we can close it out later.
        """
        log_id = self.database.log_entry(table_id, entry_timestamp)
        self._open_log_ids[table_id] = log_id
        print(f"[DB] Table {table_id} occupied at "
              f"{time.strftime('%H:%M:%S', time.localtime(entry_timestamp))} (log id {log_id})")

    def _handle_table_vacated(self, table_id: int, entry_timestamp: float, exit_timestamp: float):
        """
        Called by TableTracker the moment a table is confirmed Empty again.
        Updates the previously opened log row with exit_time and duration.
        """
        log_id = self._open_log_ids.pop(table_id, None)
        if log_id is not None:
            self.database.log_exit(log_id, entry_timestamp, exit_timestamp)
            duration = exit_timestamp - entry_timestamp
            print(f"[DB] Table {table_id} vacated after {duration:.1f}s (log id {log_id})")

    # ----------------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------------
    def run(self):
        """
        Main video processing loop:
            1. Read a frame from the video source.
            2. Run person + pose detection.
            3. Update table occupancy states via the tracker.
            4. Draw all overlays via the dashboard.
            5. Display the frame and handle keyboard input.

        Runs until the video ends, the source disconnects, or the user
        presses 'q' to quit.
        """
        try:
            while True:
                success, frame = self.capture.read()

                if not success:
                    # If we're playing a demo/CCTV video FILE (not a live webcam),
                    # loop back to the start instead of quitting - much nicer for
                    # continuous demos/presentations.
                    if isinstance(self.video_source, str) and Config.LOOP_VIDEO_FILE:
                        print("[App] Demo video ended - looping back to start.")
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("[App] Video stream ended or frame could not be read.")
                    break

                # Normalize every frame to the configured canvas size, so the
                # TABLE_ZONES pixel coordinates line up consistently regardless
                # of the source video's native resolution (webcam vs. CCTV file
                # vs. this demo clip, which may all differ in size).
                frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))

                # Step 1: detect people + poses in this frame.
                detections = self.detector.detect(frame)

                # Step 2: update occupancy state machine for all tables.
                statuses = self.tracker.update(detections)

                # Step 3: compute dashboard summary stats.
                summary = self.tracker.get_summary(statuses)

                # Step 4: draw everything onto the frame.
                annotated_frame = self.dashboard.render(frame, detections, statuses, summary)

                # Step 5: display the frame.
                cv2.imshow(Config.WINDOW_NAME, annotated_frame)

                # Handle keyboard input (1ms wait keeps the video responsive).
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[App] 'q' pressed - shutting down.")
                    break
                elif key == ord('e'):
                    csv_path = self.database.export_daily_csv()
                    print(f"[App] Daily report exported to: {csv_path}")

        except KeyboardInterrupt:
            print("[App] Interrupted by user (Ctrl+C).")
        finally:
            self.cleanup()

    # ----------------------------------------------------------------
    # CLEANUP
    # ----------------------------------------------------------------
    def cleanup(self):
        """
        Release all resources cleanly on shutdown: close the video capture,
        destroy OpenCV windows, export a final CSV report, and close the DB.
        """
        print("[App] Cleaning up resources...")

        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()

        # Export a final CSV snapshot of today's data on shutdown, so
        # nothing is lost even if the app wasn't manually exported via 'e'.
        try:
            csv_path = self.database.export_daily_csv(current_date_str())
            print(f"[App] Final daily report exported to: {csv_path}")
        except Exception as export_error:
            print(f"[App] Warning: could not export final CSV report: {export_error}")

        self.database.close()
        print("[App] Shutdown complete.")


def main():
    """
    Script entry point. Creates and runs the RestaurantMonitorApp.
    Wrapped in a function (rather than bare module-level code) so this
    file can also be imported elsewhere (e.g., for unit tests) without
    automatically starting the video loop.
    """
    app = RestaurantMonitorApp(video_source=Config.VIDEO_SOURCE)
    app.run()


if __name__ == "__main__":
    main()
