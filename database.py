"""
database.py
------------
Handles all persistence for the Restaurant CCTV Monitoring System:
    - SQLite database for storing table occupancy logs
      (table number, entry time, exit time, duration)
    - Exporting daily reports to CSV
    - Staff accounts + role-based authentication (Manager / Captain / Waiter)
    - Live table status (shared "source of truth" so every logged-in staff
      member sees consistent data, even though only the Manager's session
      runs the camera/YOLO pipeline)
    - Waiter-to-table assignments
    - Food-service events (tracks how long after seating food was served)
    - Notifications (occupancy alerts, assignment alerts) targeted by role

Kept as its own module so the storage mechanism (SQLite) can be swapped
out later (e.g., for MySQL/PostgreSQL) without touching detection,
tracking, or dashboard logic - another example of separation of concerns.

SECURITY NOTE: Authentication here is intentionally simple (SHA-256 hashed
passwords, no salting/rate-limiting/session tokens) since this is a demo
mini-project, not a production system. A real deployment should use a
proper auth library (e.g., bcrypt/argon2 with per-user salts) and HTTPS.
"""

import sqlite3
import csv
import os
import hashlib
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from utils import Config, current_timestamp, current_date_str, format_duration


class DatabaseManager:
    """
    Manages all SQLite database operations for occupancy logs, plus
    exporting those logs to CSV daily reports.
    """

    def __init__(self, db_path: str = Config.DB_PATH):
        """
        Connect to (or create) the SQLite database file and ensure the
        required table schema exists.

        Args:
            db_path: filesystem path to the .db file.
        """
        self.db_path = db_path
        # check_same_thread=False allows the connection to be used safely
        # from the main video-processing loop without extra thread issues
        # in this single-threaded application.
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.execute("PRAGMA foreign_keys = ON;")
        self._create_schema()

    def _create_schema(self):
        """
        Create every required table if it doesn't already exist, and seed
        demo staff accounts on first run.

        Schema overview:
            table_logs         - one row per table visit (entry/exit/duration)
            users               - staff accounts (username, hashed password, role)
            live_status          - CURRENT status of every table (shared "source
                                    of truth" read by Manager/Captain/Waiter alike)
            table_assignments     - which waiter is currently servicing each table
            service_events         - food-served events, with time-to-serve tracking
            notifications           - role/user-targeted alerts (occupancy, assignment)
        """
        create_statements = [
            """
            CREATE TABLE IF NOT EXISTS table_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_number INTEGER NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                duration_sec REAL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('manager', 'captain', 'waiter')),
                full_name TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS live_status (
                table_number INTEGER PRIMARY KEY,
                is_occupied INTEGER NOT NULL DEFAULT 0,
                num_people INTEGER NOT NULL DEFAULT 0,
                occupied_since TEXT,
                current_log_id INTEGER,
                last_updated TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS table_assignments (
                table_number INTEGER PRIMARY KEY,
                waiter_username TEXT,
                assigned_by TEXT,
                assigned_at TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS service_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_number INTEGER NOT NULL,
                log_id INTEGER,
                waiter_username TEXT NOT NULL,
                served_at TEXT NOT NULL,
                seconds_to_serve REAL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_number INTEGER,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                target_roles TEXT NOT NULL
            );
            """,
        ]
        with self.connection:
            for statement in create_statements:
                self.connection.execute(statement)

        self._seed_demo_users()

    def _seed_demo_users(self):
        """
        Insert demo staff accounts on first run only (INSERT OR IGNORE means
        this is safe to call every startup without duplicating or resetting
        existing accounts/passwords).

        Demo credentials (for the project report / grading demo):
            manager  / manager123   (role: manager)
            captain  / captain123   (role: captain)
            waiter1  / waiter123    (role: waiter)
            waiter2  / waiter123    (role: waiter)
        """
        demo_users = [
            ("manager", "manager123", "manager", "Manager Demo"),
            ("captain", "captain123", "captain", "Captain Demo"),
            ("waiter1", "waiter123", "waiter", "Waiter One"),
            ("waiter2", "waiter123", "waiter", "Waiter Two"),
        ]
        with self.connection:
            for username, plain_password, role, full_name in demo_users:
                self.connection.execute(
                    "INSERT OR IGNORE INTO users (username, password_hash, role, full_name) VALUES (?, ?, ?, ?);",
                    (username, self._hash_password(plain_password), role, full_name),
                )

    # ------------------------------------------------------------
    # AUTHENTICATION
    # ------------------------------------------------------------
    @staticmethod
    def _hash_password(plain_password: str) -> str:
        """
        Hash a plaintext password with SHA-256. Simple and dependency-free,
        appropriate for this demo project (see the SECURITY NOTE at the top
        of this file for production-readiness caveats).
        """
        return hashlib.sha256(plain_password.encode("utf-8")).hexdigest()

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Verify a username/password pair against the users table.

        Args:
            username: the entered username.
            password: the entered plaintext password.

        Returns:
            A dict {username, role, full_name} if credentials are valid,
            otherwise None.
        """
        cursor = self.connection.execute(
            "SELECT username, password_hash, role, full_name FROM users WHERE username = ?;",
            (username,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        db_username, password_hash, role, full_name = row
        if self._hash_password(password) == password_hash:
            return {"username": db_username, "role": role, "full_name": full_name}
        return None

    def get_all_waiters(self) -> List[Dict[str, Any]]:
        """
        Return every staff account with role='waiter', used to populate
        assignment dropdowns for Manager/Captain.
        """
        cursor = self.connection.execute(
            "SELECT username, full_name FROM users WHERE role = 'waiter' ORDER BY full_name;"
        )
        return [{"username": u, "full_name": n} for u, n in cursor.fetchall()]

    # ------------------------------------------------------------
    # LIVE STATUS (shared source of truth for all logged-in roles)
    # ------------------------------------------------------------
    def update_live_status_frame(
        self, table_number: int, is_occupied: bool, num_people: int, occupied_since: Optional[str]
    ):
        """
        Update the per-frame fields of a table's live status (called every
        frame/second by the Manager's monitoring loop for EVERY table).
        Deliberately does NOT touch current_log_id - that's only changed by
        set_current_log_id(), called from the occupancy transition callbacks.
        """
        now = current_timestamp()
        with self.connection:
            # Ensure a row exists for this table first (no-op if it already does).
            self.connection.execute(
                "INSERT OR IGNORE INTO live_status (table_number, is_occupied, num_people, last_updated) "
                "VALUES (?, 0, 0, ?);",
                (table_number, now),
            )
            self.connection.execute(
                """
                UPDATE live_status
                SET is_occupied = ?, num_people = ?, occupied_since = ?, last_updated = ?
                WHERE table_number = ?;
                """,
                (int(is_occupied), num_people, occupied_since, now, table_number),
            )

    def set_current_log_id(self, table_number: int, log_id: Optional[int]):
        """
        Set (or clear, with log_id=None) which table_logs row is the
        CURRENTLY OPEN visit for this table. Called from the occupancy
        transition callbacks (on_occupied / on_vacated), not every frame.
        """
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO live_status (table_number, last_updated) VALUES (?, ?);",
                (table_number, current_timestamp()),
            )
            self.connection.execute(
                "UPDATE live_status SET current_log_id = ? WHERE table_number = ?;",
                (log_id, table_number),
            )

    def get_live_status(self) -> List[Dict[str, Any]]:
        """
        Retrieve the current live status of every table, as a list of dicts.
        This is what Captain/Waiter dashboards read instead of touching any
        video - satisfying "only the Manager can see the CCTV visuals".
        """
        cursor = self.connection.execute(
            "SELECT table_number, is_occupied, num_people, occupied_since, current_log_id, last_updated "
            "FROM live_status ORDER BY table_number ASC;"
        )
        rows = cursor.fetchall()
        return [
            {
                "table_number": r[0],
                "is_occupied": bool(r[1]),
                "num_people": r[2],
                "occupied_since": r[3],
                "current_log_id": r[4],
                "last_updated": r[5],
            }
            for r in rows
        ]

    def get_live_status_for_table(self, table_number: int) -> Optional[Dict[str, Any]]:
        """Retrieve the live status row for a single table, or None if not present yet."""
        for row in self.get_live_status():
            if row["table_number"] == table_number:
                return row
        return None

    # ------------------------------------------------------------
    # TABLE ASSIGNMENTS (Manager & Captain only, enforced in the UI layer)
    # ------------------------------------------------------------
    def assign_waiter(self, table_number: int, waiter_username: str, assigned_by: str):
        """
        Assign a waiter to service a given table. Overwrites any previous
        assignment for that table (a table has exactly one assigned waiter
        at a time in this simplified model).

        Args:
            table_number: which table is being assigned.
            waiter_username: the waiter's username being assigned to it.
            assigned_by: username of the Manager/Captain making the assignment.
        """
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO table_assignments (table_number, waiter_username, assigned_by, assigned_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(table_number) DO UPDATE SET
                    waiter_username = excluded.waiter_username,
                    assigned_by = excluded.assigned_by,
                    assigned_at = excluded.assigned_at;
                """,
                (table_number, waiter_username, assigned_by, current_timestamp()),
            )

    def get_assignments(self) -> Dict[int, Dict[str, Any]]:
        """
        Retrieve all current table-to-waiter assignments.

        Returns:
            dict mapping table_number -> {waiter_username, assigned_by, assigned_at}
        """
        cursor = self.connection.execute(
            "SELECT table_number, waiter_username, assigned_by, assigned_at FROM table_assignments;"
        )
        return {
            row[0]: {"waiter_username": row[1], "assigned_by": row[2], "assigned_at": row[3]}
            for row in cursor.fetchall()
        }

    def get_tables_for_waiter(self, waiter_username: str) -> List[int]:
        """Return the list of table numbers currently assigned to a given waiter."""
        cursor = self.connection.execute(
            "SELECT table_number FROM table_assignments WHERE waiter_username = ?;",
            (waiter_username,),
        )
        return [row[0] for row in cursor.fetchall()]

    # ------------------------------------------------------------
    # FOOD SERVICE EVENTS ("food given time" tracking)
    # ------------------------------------------------------------
    def mark_food_served(self, table_number: int, log_id: Optional[int], waiter_username: str,
                          entry_time_str: Optional[str]) -> float:
        """
        Record that food was served at a table by a waiter, and compute how
        many seconds elapsed between seating and food being served.

        Args:
            table_number: which table.
            log_id: the table_logs.id of the current open visit (from live_status).
            waiter_username: who served the food.
            entry_time_str: the visit's entry_time string ("YYYY-MM-DD HH:MM:SS"),
                used to compute seconds_to_serve.

        Returns:
            seconds_to_serve (float), or -1.0 if entry_time_str was unavailable.
        """
        served_at_dt = datetime.now()
        seconds_to_serve = -1.0
        if entry_time_str:
            entry_dt = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
            seconds_to_serve = round((served_at_dt - entry_dt).total_seconds(), 2)

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO service_events (table_number, log_id, waiter_username, served_at, seconds_to_serve)
                VALUES (?, ?, ?, ?, ?);
                """,
                (table_number, log_id, waiter_username, served_at_dt.strftime("%Y-%m-%d %H:%M:%S"), seconds_to_serve),
            )
        return seconds_to_serve

    def has_food_been_served(self, log_id: Optional[int]) -> bool:
        """Check whether a food-served event already exists for a given visit (log_id)."""
        if log_id is None:
            return False
        cursor = self.connection.execute(
            "SELECT COUNT(*) FROM service_events WHERE log_id = ?;", (log_id,)
        )
        return cursor.fetchone()[0] > 0

    def get_service_events_for_date(self, date_str: str) -> List[Tuple]:
        """Retrieve all food-service events whose served_at falls on a given date."""
        cursor = self.connection.execute(
            """
            SELECT id, table_number, log_id, waiter_username, served_at, seconds_to_serve
            FROM service_events WHERE served_at LIKE ? ORDER BY served_at ASC;
            """,
            (f"{date_str}%",),
        )
        return cursor.fetchall()

    # ------------------------------------------------------------
    # NOTIFICATIONS
    # ------------------------------------------------------------
    def create_notification(self, table_number: Optional[int], message: str, target_roles: str):
        """
        Create a new notification.

        Args:
            table_number: which table this relates to (nullable for general alerts).
            message: human-readable notification text.
            target_roles: comma-separated string of roles and/or specific
                usernames that should see this notification, e.g.
                "manager,captain,waiter2" or "manager,captain,waiter" (the
                literal role name "waiter" broadcasts to every waiter account).
        """
        with self.connection:
            self.connection.execute(
                "INSERT INTO notifications (table_number, message, created_at, target_roles) VALUES (?, ?, ?, ?);",
                (table_number, message, current_timestamp(), target_roles),
            )

    def get_notifications_for_user(self, username: str, role: str, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Retrieve recent notifications visible to a given user: either their
        role or their specific username appears in the notification's
        comma-separated target_roles field.

        Args:
            username: the logged-in user's username.
            role: the logged-in user's role.
            limit: max number of notifications to return (most recent first).
        """
        cursor = self.connection.execute(
            "SELECT id, table_number, message, created_at, target_roles FROM notifications "
            "ORDER BY id DESC LIMIT ?;",
            (limit * 3,),  # over-fetch since we filter in Python below
        )
        results = []
        for row in cursor.fetchall():
            notif_id, table_number, message, created_at, target_roles = row
            targets = [t.strip() for t in target_roles.split(",")]
            if role in targets or username in targets:
                results.append({
                    "id": notif_id,
                    "table_number": table_number,
                    "message": message,
                    "created_at": created_at,
                })
            if len(results) >= limit:
                break
        return results

    def log_entry(self, table_id: int, entry_timestamp: float) -> int:
        """
        Insert a new log row when a table transitions to Occupied.
        The exit_time and duration_sec are left NULL until log_exit() is
        called later for this same visit.

        Args:
            table_id: which table became occupied.
            entry_timestamp: unix timestamp (float, from time.time()) of entry.

        Returns:
            The row id (int) of the newly inserted log entry, so it can be
            matched up later when the corresponding exit is logged.
        """
        entry_time_str = datetime.fromtimestamp(entry_timestamp).strftime("%Y-%m-%d %H:%M:%S")
        insert_sql = """
            INSERT INTO table_logs (table_number, entry_time, exit_time, duration_sec)
            VALUES (?, ?, NULL, NULL);
        """
        with self.connection:
            cursor = self.connection.execute(insert_sql, (table_id, entry_time_str))
            return cursor.lastrowid

    def log_exit(self, log_id: int, entry_timestamp: float, exit_timestamp: float):
        """
        Update an existing log row when a table transitions back to Empty,
        filling in exit_time and the computed duration.

        Args:
            log_id: the row id returned earlier by log_entry().
            entry_timestamp: unix timestamp of when the table was occupied.
            exit_timestamp: unix timestamp of when the table became empty.
        """
        exit_time_str = datetime.fromtimestamp(exit_timestamp).strftime("%Y-%m-%d %H:%M:%S")
        duration_sec = round(exit_timestamp - entry_timestamp, 2)

        update_sql = """
            UPDATE table_logs
            SET exit_time = ?, duration_sec = ?
            WHERE id = ?;
        """
        with self.connection:
            self.connection.execute(update_sql, (exit_time_str, duration_sec, log_id))

    def get_logs_for_date(self, date_str: str) -> List[Tuple]:
        """
        Retrieve all log rows whose entry_time falls on a given date.

        Args:
            date_str: date in "YYYY-MM-DD" format.

        Returns:
            List of tuples: (id, table_number, entry_time, exit_time, duration_sec)
        """
        query_sql = """
            SELECT id, table_number, entry_time, exit_time, duration_sec
            FROM table_logs
            WHERE entry_time LIKE ?
            ORDER BY entry_time ASC;
        """
        cursor = self.connection.execute(query_sql, (f"{date_str}%",))
        return cursor.fetchall()

    def get_all_logs(self) -> List[Tuple]:
        """
        Retrieve every log row in the database (used for full history export
        or debugging).

        Returns:
            List of tuples: (id, table_number, entry_time, exit_time, duration_sec)
        """
        cursor = self.connection.execute(
            "SELECT id, table_number, entry_time, exit_time, duration_sec FROM table_logs ORDER BY entry_time ASC;"
        )
        return cursor.fetchall()

    def export_daily_csv(self, date_str: Optional[str] = None) -> str:
        """
        Export all logs for a given date (defaults to today) into a CSV
        file inside the logs/ directory, named daily_report_YYYY-MM-DD.csv.

        Args:
            date_str: date to export in "YYYY-MM-DD" format. Defaults to
                today's date if not provided.

        Returns:
            The full filesystem path to the generated CSV file.
        """
        if date_str is None:
            date_str = current_date_str()

        rows = self.get_logs_for_date(date_str)

        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        csv_path = os.path.join(Config.LOGS_DIR, f"daily_report_{date_str}.csv")

        with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            # Header row
            writer.writerow(["Log ID", "Table Number", "Entry Time", "Exit Time", "Duration (formatted)", "Duration (seconds)"])

            for row in rows:
                log_id, table_number, entry_time, exit_time, duration_sec = row
                duration_formatted = format_duration(duration_sec) if duration_sec is not None else "Still occupied"
                writer.writerow([
                    log_id,
                    table_number,
                    entry_time,
                    exit_time if exit_time else "Still occupied",
                    duration_formatted,
                    duration_sec if duration_sec is not None else "",
                ])

        return csv_path

    def close(self):
        """
        Close the database connection cleanly. Should be called when the
        application shuts down (e.g., in main.py's cleanup step).
        """
        self.connection.close()
