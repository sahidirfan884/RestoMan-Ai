"""
tracker.py
----------
Core occupancy-tracking logic for the Restaurant CCTV Monitoring System.

This module takes:
    - a list of PersonDetection objects (from detector.py) for the current frame
    - the predefined TableZone list (from utils.py)

...and produces, per table:
    - current occupancy state (Occupied / Empty)
    - number of people currently in that zone
    - how long the table has been continuously occupied

It uses a simple debounce state machine per table so that momentary noise
(e.g., a waiter walking past, or a brief mis-detection) doesn't cause the
table status to flicker between Occupied/Empty every frame.

This module is intentionally decoupled from database.py: instead of
importing DatabaseManager directly, TableTracker accepts optional callback
functions (on_occupied, on_vacated) that main.py wires up. This keeps the
tracker reusable/testable on its own (a good OOP/software design practice -
"dependency injection" instead of tight coupling).
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

from utils import (
    TableZone,
    get_person_reference_point,
    find_zone_for_point,
    format_duration,
    is_sitting_posture,
)
from detector import PersonDetection


# --------------------------------------------------------------------------
# STATUS SNAPSHOT RETURNED TO THE CALLER (main.py / dashboard.py) EACH FRAME
# --------------------------------------------------------------------------
@dataclass
class TableStatus:
    """
    Represents the current status of one table for a single frame,
    ready to be displayed on the dashboard or drawn on the video.

    Attributes:
        table_id: which table this status belongs to.
        occupied: True if the table is currently considered Occupied.
        num_people: how many people are currently detected inside this zone.
        occupied_since: unix timestamp (float) when this table became
            occupied (None if currently empty).
        duration_str: human-readable "time since occupied" (e.g., "2m 5s").
    """
    table_id: int
    occupied: bool
    num_people: int
    occupied_since: Optional[float]
    duration_str: str


# --------------------------------------------------------------------------
# INTERNAL STATE HELD PER TABLE (not exposed outside the tracker)
# --------------------------------------------------------------------------
@dataclass
class _TableState:
    """
    Internal bookkeeping for one table's debounce state machine.
    Prefixed with underscore since this is an implementation detail of
    TableTracker, not meant to be used directly by other modules.
    """
    is_occupied: bool = False              # confirmed current state
    occupied_since: Optional[float] = None  # timestamp when confirmed occupied

    # debounce bookkeeping
    pending_occupied_since: Optional[float] = None  # when "people appeared" started
    pending_empty_since: Optional[float] = None      # when "zone went empty" started


# --------------------------------------------------------------------------
# TABLE TRACKER CLASS
# --------------------------------------------------------------------------
class TableTracker:
    """
    Tracks occupancy of every defined table zone across video frames,
    applying debounce thresholds to avoid flickering status changes.
    """

    def __init__(
        self,
        table_zones: List[TableZone],
        occupied_debounce_seconds: float,
        empty_grace_seconds: float,
        on_occupied: Optional[Callable[[int, float], None]] = None,
        on_vacated: Optional[Callable[[int, float, float], None]] = None,
    ):
        """
        Args:
            table_zones: list of TableZone objects defining each table's area.
            occupied_debounce_seconds: how long people must be continuously
                present before a table flips from Empty -> Occupied.
            empty_grace_seconds: how long a zone must stay empty before a
                table flips from Occupied -> Empty (and gets logged).
            on_occupied: optional callback fired when a table transitions to
                Occupied. Signature: on_occupied(table_id, entry_timestamp).
            on_vacated: optional callback fired when a table transitions to
                Empty. Signature: on_vacated(table_id, entry_timestamp, exit_timestamp).
        """
        self.table_zones = table_zones
        self.occupied_debounce_seconds = occupied_debounce_seconds
        self.empty_grace_seconds = empty_grace_seconds
        self.on_occupied = on_occupied
        self.on_vacated = on_vacated

        # One _TableState per table, keyed by table_id.
        self._states: Dict[int, _TableState] = {
            zone.table_id: _TableState() for zone in table_zones
        }

    def update(self, detections: List[PersonDetection]) -> List[TableStatus]:
        """
        Process one frame's worth of detections and update every table's
        occupancy state machine accordingly.

        Args:
            detections: list of PersonDetection objects for the current frame.

        Returns:
            List[TableStatus]: the current status of every table, in the
            same order as self.table_zones - ready for the dashboard to draw.
        """
        now = time.time()

        # Step 1: count how many SITTING people are currently inside each
        # table zone. Standing people (e.g., waiters, passersby) inside a
        # zone are deliberately ignored - only seated customers count as
        # "occupying" a table.
        people_count_per_table: Dict[int, int] = {zone.table_id: 0 for zone in self.table_zones}

        for detection in detections:
            if not is_sitting_posture(detection.bbox, detection.keypoints):
                continue  # skip standing people - not counted as customers

            ref_point = get_person_reference_point(detection.bbox, detection.keypoints)
            table_id = find_zone_for_point(ref_point[0], ref_point[1], self.table_zones)
            if table_id is not None:
                people_count_per_table[table_id] += 1

        # Step 2: run the debounce state machine for each table.
        statuses: List[TableStatus] = []
        for zone in self.table_zones:
            table_id = zone.table_id
            state = self._states[table_id]
            people_present = people_count_per_table[table_id] > 0

            if not state.is_occupied:
                # Currently EMPTY - check if it should transition to OCCUPIED.
                if people_present:
                    if state.pending_occupied_since is None:
                        # People just appeared - start the debounce timer.
                        state.pending_occupied_since = now
                    elif now - state.pending_occupied_since >= self.occupied_debounce_seconds:
                        # People have been present long enough - confirm occupied.
                        state.is_occupied = True
                        state.occupied_since = state.pending_occupied_since
                        state.pending_occupied_since = None
                        if self.on_occupied:
                            self.on_occupied(table_id, state.occupied_since)
                else:
                    # No one present - reset any pending occupied timer.
                    state.pending_occupied_since = None

            else:
                # Currently OCCUPIED - check if it should transition to EMPTY.
                if not people_present:
                    if state.pending_empty_since is None:
                        # Zone just became empty - start the grace timer.
                        state.pending_empty_since = now
                    elif now - state.pending_empty_since >= self.empty_grace_seconds:
                        # Zone has been empty long enough - confirm vacated.
                        entry_time = state.occupied_since
                        exit_time = state.pending_empty_since
                        if self.on_vacated:
                            self.on_vacated(table_id, entry_time, exit_time)
                        state.is_occupied = False
                        state.occupied_since = None
                        state.pending_empty_since = None
                else:
                    # Someone is still present - reset the empty grace timer.
                    state.pending_empty_since = None

            # Build the status snapshot for this table for the current frame.
            duration_str = ""
            if state.is_occupied and state.occupied_since is not None:
                duration_str = format_duration(now - state.occupied_since)

            statuses.append(TableStatus(
                table_id=table_id,
                occupied=state.is_occupied,
                num_people=people_count_per_table[table_id],
                occupied_since=state.occupied_since,
                duration_str=duration_str,
            ))

        return statuses

    def get_summary(self, statuses: List[TableStatus]) -> Dict[str, int]:
        """
        Compute dashboard-level summary statistics from a list of TableStatus
        objects: total tables, occupied count, empty count, total customers.

        Args:
            statuses: the list returned by update() for the current frame.

        Returns:
            dict with keys: total_tables, occupied_tables, empty_tables, total_customers.
        """
        total_tables = len(statuses)
        occupied_tables = sum(1 for s in statuses if s.occupied)
        empty_tables = total_tables - occupied_tables
        total_customers = sum(s.num_people for s in statuses)

        return {
            "total_tables": total_tables,
            "occupied_tables": occupied_tables,
            "empty_tables": empty_tables,
            "total_customers": total_customers,
        }
