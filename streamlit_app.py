"""
streamlit_app.py
-----------------
RestoMan-Ai — Role-based staff web dashboard for the Restaurant CCTV
Monitoring System.

Reuses every core module unchanged:
    detector.py   -> PersonDetector (YOLOv8-Pose)
    tracker.py     -> TableTracker (occupancy state machine, sitting filter)
    database.py     -> DatabaseManager (SQLite: logs, users, live status,
                        assignments, service events, notifications)
    dashboard.py      -> Dashboard (drawing overlays onto frames)
    utils.py           -> Config, TableZone, load_table_zones/save_table_zones

Run with:
    streamlit run streamlit_app.py

ROLES:
    Manager  - full access: live CCTV video, zone setup, table assignment,
               notifications, and ALL reports/analytics.
    Captain  - table assignment (which waiter services which table), a
               NON-VIDEO live table-status board, and notifications.
               No CCTV visuals, no reports.
    Waiter   - "My Tables" view (only tables assigned to them), a button to
               mark food as served (tracks time-to-serve), and their own
               notifications. No CCTV visuals, no reports, no assignment.

ARCHITECTURE NOTE ON SHARED STATE:
Only the Manager's browser session actually opens the video source and runs
YOLO detection (st.fragment loop, same technique as before). That session
writes the CURRENT status of every table into the SQLite `live_status`
table on every frame. Captain and Waiter sessions never touch video or the
model at all - they simply poll `live_status` (and notifications) from the
database every couple of seconds. This is what makes "only the Manager can
see the CCTV visuals" both an access-control rule AND the actual data flow.
"""

import os
import time
import tempfile
from datetime import datetime

import cv2
import streamlit as st

from utils import Config, load_table_zones, save_table_zones, current_date_str, format_duration, TableZone
from detector import PersonDetector
from tracker import TableTracker
from database import DatabaseManager
from dashboard import Dashboard
from theme import inject_theme, render_header, render_ticket_card_html, render_staff_badge, render_notification_html


# ==========================================================================
# PAGE CONFIG
# ==========================================================================
st.set_page_config(
    page_title="RestoMan-Ai | The Pass",
    page_icon="🛎️",
    layout="wide",
)
inject_theme()


# ==========================================================================
# CACHED RESOURCE LOADERS (shared across ALL sessions on this server)
# ==========================================================================
@st.cache_resource
def load_detector() -> PersonDetector:
    """Load the YOLOv8-Pose model exactly once, shared by every session."""
    return PersonDetector()


@st.cache_resource
def load_database() -> DatabaseManager:
    """Open the SQLite connection exactly once, shared by every session."""
    return DatabaseManager()


def build_tracker(database: DatabaseManager, table_zones) -> TableTracker:
    """
    Build a fresh TableTracker whose occupancy callbacks write straight into
    the shared SQLite tables (table_logs, live_status, notifications)
    instead of an in-memory dict - so Captain/Waiter sessions (running in a
    totally separate Python call stack) can see the exact same live state.
    """

    def on_occupied(table_id: int, entry_ts: float):
        """Fires when a table is confirmed Occupied: open a log row, update
        live_status, and notify Manager + Captain + the assigned waiter
        (or all waiters, if this table isn't assigned to anyone yet)."""
        log_id = database.log_entry(table_id, entry_ts)
        database.set_current_log_id(table_id, log_id)

        assignments = database.get_assignments()
        assigned = assignments.get(table_id, {}).get("waiter_username")
        target_roles = f"manager,captain,{assigned}" if assigned else "manager,captain,waiter"
        database.create_notification(
            table_id, f"🔴 Table {table_id} is now occupied.", target_roles
        )

    def on_vacated(table_id: int, entry_ts: float, exit_ts: float):
        """Fires when a table is confirmed Empty again: close the log row
        using whatever log_id is currently stored in live_status (the
        shared source of truth), then clear it."""
        live_row = database.get_live_status_for_table(table_id)
        log_id = live_row["current_log_id"] if live_row else None
        if log_id is not None:
            database.log_exit(log_id, entry_ts, exit_ts)
        database.set_current_log_id(table_id, None)

    return TableTracker(
        table_zones=table_zones,
        occupied_debounce_seconds=Config.OCCUPIED_DEBOUNCE_SECONDS,
        empty_grace_seconds=Config.EMPTY_GRACE_SECONDS,
        on_occupied=on_occupied,
        on_vacated=on_vacated,
    )


# ==========================================================================
# SESSION STATE INITIALIZATION
# ==========================================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None
if "full_name" not in st.session_state:
    st.session_state.full_name = None

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "capture" not in st.session_state:
    st.session_state.capture = None
if "tracker" not in st.session_state:
    st.session_state.tracker = None
if "dashboard" not in st.session_state:
    st.session_state.dashboard = None
if "current_video_source" not in st.session_state:
    st.session_state.current_video_source = None


# ==========================================================================
# LOGIN SCREEN
# ==========================================================================
def show_login_screen():
    """
    Render the login form. Stops script execution here (via st.stop()) if
    the user isn't authenticated yet, so nothing below this function ever
    runs for a logged-out visitor.
    """
    st.markdown(
        "<h1 style='text-align:center;'>🍽️ RestoMan-Ai</h1>"
        "<p style='text-align:center;color:gray;'>Restaurant Table Monitoring — Staff Login</p>",
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        st.markdown('<div class="rm-login-card">', unsafe_allow_html=True)
        st.markdown('<div class="rm-login-title">🛎️ RestoMan-Ai</div>', unsafe_allow_html=True)
        st.markdown('<div class="rm-login-sub">Staff Entrance — The Pass</div>', unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("🔐 Clock In", use_container_width=True)

        if submitted:
            database = load_database()
            user = database.authenticate(username.strip(), password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = user["username"]
                st.session_state.role = user["role"]
                st.session_state.full_name = user["full_name"]
                st.rerun()
            else:
                st.error("Invalid username or password.")

        with st.expander("ℹ️ Demo accounts (for grading/testing)"):
            st.markdown(
                "| Role | Username | Password |\n"
                "|---|---|---|\n"
                "| Manager | `manager` | `manager123` |\n"
                "| Captain | `captain` | `captain123` |\n"
                "| Waiter  | `waiter1` | `waiter123` |\n"
                "| Waiter  | `waiter2` | `waiter123` |\n"
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()


if not st.session_state.logged_in:
    show_login_screen()


# ==========================================================================
# SIDEBAR - SHARED ACROSS ALL ROLES
# ==========================================================================
database = load_database()

st.sidebar.markdown('<div class="rm-header-title" style="font-size:1.8rem;">🛎️ RestoMan-Ai</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="rm-header-sub" style="margin-bottom:12px;">The Pass</div>', unsafe_allow_html=True)
with st.sidebar:
    render_staff_badge(st.session_state.full_name, st.session_state.role)
if st.sidebar.button("🚪 Clock Out", use_container_width=True):
    # Stop any active monitoring FIRST, so the video fragment doesn't
    # keep trying to update placeholders after we navigate to the login
    # screen (this is what causes the blank-page crash).
    st.session_state.monitoring = False
    if st.session_state.capture is not None:
        st.session_state.capture.release()
        st.session_state.capture = None

    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.full_name = None
    st.rerun()

st.sidebar.divider()


# ==========================================================================
# NOTIFICATIONS WIDGET (shown to every role, filtered to what they can see)
# ==========================================================================
def render_notifications_panel():
    """
    Render a small notifications feed in the sidebar for the current user,
    pulling from the shared `notifications` table and filtering to what
    this user's role/username is allowed to see.
    """
    notifs = database.get_notifications_for_user(
        st.session_state.username, st.session_state.role, limit=15
    )
    st.sidebar.markdown(f'<div class="rm-header-sub">🔔 NOTIFICATIONS ({len(notifs)})</div>', unsafe_allow_html=True)
    if not notifs:
        st.sidebar.caption("No notifications yet.")
    else:
        with st.sidebar.container(height=240):
            for n in notifs:
                st.markdown(render_notification_html(n["created_at"], n["message"]), unsafe_allow_html=True)


render_notifications_panel()


# ==========================================================================
# MANAGER-ONLY: LIVE MONITOR (CCTV VIDEO) TAB CONTENT
# ==========================================================================
def render_live_monitor_tab():
    """
    Full CCTV video + skeletons + table zones + dashboard metrics.
    Only ever called for role == 'manager' - this is the ONLY place in the
    whole app that opens a video source or runs YOLO detection.
    """
    st.markdown("### Video Source")
    source_option = st.radio(
        "Choose a source", ["Demo Video", "Upload Video", "Webcam"], horizontal=True
    )

    video_source = None
    if source_option == "Demo Video":
        video_source = Config.VIDEO_SOURCE
    elif source_option == "Webcam":
        video_source = 0
    else:
        uploaded_file = st.file_uploader(
            "Upload a CCTV/table video", type=["mp4", "avi", "mov", "webm", "mkv"]
        )
        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(uploaded_file.read())
            temp_file.close()
            video_source = temp_file.name
        else:
            st.warning("Please upload a video file to continue.")

    col_start, col_stop, col_status = st.columns([1, 1, 3])
    start_clicked = col_start.button("▶ Start Monitoring", disabled=(video_source is None))
    stop_clicked = col_stop.button("⏹ Stop Monitoring")

    if start_clicked and video_source is not None:
        if st.session_state.capture is not None:
            st.session_state.capture.release()

        capture = cv2.VideoCapture(video_source)
        if not capture.isOpened():
            st.error("Could not open this video source. Check the file/webcam.")
        else:
            current_zones = load_table_zones(Config.ZONES_CONFIG_PATH)
            st.session_state.capture = capture
            st.session_state.current_video_source = video_source
            st.session_state.tracker = build_tracker(database, current_zones)
            st.session_state.dashboard = Dashboard(current_zones)
            st.session_state.monitoring = True

    if stop_clicked:
        st.session_state.monitoring = False
        if st.session_state.capture is not None:
            st.session_state.capture.release()
            st.session_state.capture = None

    col_status.markdown(
        "**Status:** 🟢 Monitoring Live" if st.session_state.monitoring else "**Status:** ⚪ Stopped"
    )

    metric_cols = st.columns(4)
    metric_total = metric_cols[0].empty()
    metric_occupied = metric_cols[1].empty()
    metric_empty = metric_cols[2].empty()
    metric_customers = metric_cols[3].empty()

    zone_count = len(load_table_zones(Config.ZONES_CONFIG_PATH))
    metric_total.metric("Total Tables", zone_count)
    metric_occupied.metric("Occupied", 0)
    metric_empty.metric("Empty", zone_count)
    metric_customers.metric("Total Customers", 0)

    st.markdown('<div class="rm-monitor-frame">', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    status_placeholder = st.empty()

    if not st.session_state.monitoring:
        frame_placeholder.info("Choose a video source and click ▶ Start Monitoring above.")

    @st.fragment(run_every=0.15)
    def live_frame_fragment():
        """Processes and displays ONE frame per re-execution (every 0.15s),
        and writes this frame's table statuses into the shared live_status
        table so Captain/Waiter sessions can see it too."""
        if not st.session_state.monitoring or st.session_state.capture is None:
            return

        success, frame = st.session_state.capture.read()

        if not success:
            is_file_source = isinstance(st.session_state.current_video_source, str)
            if is_file_source and Config.LOOP_VIDEO_FILE:
                st.session_state.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            st.session_state.monitoring = False
            st.warning("Video stream ended.")
            return

        frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))

        detector = load_detector()
        detections = detector.detect(frame)
        statuses = st.session_state.tracker.update(detections)
        summary = st.session_state.tracker.get_summary(statuses)
        annotated_frame = st.session_state.dashboard.render(frame, detections, statuses, summary)

        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

        metric_total.metric("Total Tables", summary["total_tables"])
        metric_occupied.metric("Occupied", summary["occupied_tables"])
        metric_empty.metric("Empty", summary["empty_tables"])
        metric_customers.metric("Total Customers", summary["total_customers"])

        # Push every table's current-frame status into the shared DB table.
        for s in statuses:
            database.update_live_status_frame(
                s.table_id, is_occupied=s.occupied, num_people=s.num_people,
                occupied_since=(datetime.fromtimestamp(s.occupied_since).strftime("%Y-%m-%d %H:%M:%S")
                                if s.occupied_since else None),
            )

        table_rows = [
            {
                "Table": s.table_id,
                "Status": "🔴 Occupied" if s.occupied else "🟢 Empty",
                "People (sitting)": s.num_people,
                "Time Seated": s.duration_str if s.duration_str else "-",
            }
            for s in statuses
        ]
        status_placeholder.dataframe(table_rows, use_container_width=True, hide_index=True)

    live_frame_fragment()


# ==========================================================================
# SHARED (Manager/Captain/Waiter): NON-VIDEO LIVE TABLE STATUS BOARD
# ==========================================================================
def render_table_status_board(filter_table_ids=None):
    """
    Non-video table status board, read entirely from the shared
    `live_status` DB table (no camera, no YOLO model touched here at all).
    Used by Captain (all tables) and Waiter (only their assigned tables).

    Args:
        filter_table_ids: optional list of table numbers to restrict the
            board to (used for the Waiter's "My Tables" view).
    """

    @st.fragment(run_every=2.0)
    def status_board_fragment():
        rows = database.get_live_status()
        if filter_table_ids is not None:
            rows = [r for r in rows if r["table_number"] in filter_table_ids]

        if not rows:
            st.info("No table data yet. Ask the Manager to start monitoring.")
            return

        # Warn if data looks stale (Manager hasn't been monitoring recently).
        try:
            most_recent = max(
                datetime.strptime(r["last_updated"], "%Y-%m-%d %H:%M:%S")
                for r in rows if r["last_updated"]
            )
            if (datetime.now() - most_recent).total_seconds() > 15:
                st.warning("⚠️ Live data looks stale - the Manager may not be actively monitoring right now.")
        except ValueError:
            pass

        assignments = database.get_assignments()
        cols = st.columns(min(len(rows), 4) or 1)
        for idx, row in enumerate(rows):
            table_id = row["table_number"]
            with cols[idx % len(cols)]:
                waiter = assignments.get(table_id, {}).get("waiter_username") or "Unassigned"
                duration_str = "-"
                if row["is_occupied"] and row["occupied_since"]:
                    entry_dt = datetime.strptime(row["occupied_since"], "%Y-%m-%d %H:%M:%S")
                    duration_str = format_duration((datetime.now() - entry_dt).total_seconds())

                already_served = (
                    database.has_food_been_served(row["current_log_id"])
                    if row["is_occupied"] and row["current_log_id"] is not None else False
                )
                extra_note = "🍽️ Food served" if already_served else None

                st.markdown(
                    render_ticket_card_html(
                        table_id=table_id,
                        occupied=row["is_occupied"],
                        num_people=row["num_people"],
                        duration_str=duration_str,
                        waiter_name=waiter,
                        extra_note=extra_note,
                    ),
                    unsafe_allow_html=True,
                )

                if row["is_occupied"] and row["current_log_id"] is not None and not already_served:
                    if st.session_state.role == "waiter":
                        if st.button("🍽️ Mark Food Served", key=f"serve_{table_id}", use_container_width=True):
                            seconds = database.mark_food_served(
                                table_id, row["current_log_id"], st.session_state.username,
                                row["occupied_since"],
                            )
                            st.success(f"Marked! ({format_duration(seconds)} after seating)")
                            st.rerun()
                st.write("")

    status_board_fragment()


# ==========================================================================
# MANAGER & CAPTAIN: TABLE ASSIGNMENT TAB
# ==========================================================================
def render_assignment_tab():
    """
    Lets Manager/Captain assign which waiter services each table.
    """
    st.markdown("### 👥 Assign Waiters to Tables")
    waiters = database.get_all_waiters()
    if not waiters:
        st.warning("No waiter accounts found.")
        return

    zones = load_table_zones(Config.ZONES_CONFIG_PATH)
    assignments = database.get_assignments()
    waiter_names = [w["username"] for w in waiters]

    for zone in zones:
        current = assignments.get(zone.table_id, {}).get("waiter_username")
        default_idx = waiter_names.index(current) + 1 if current in waiter_names else 0
        col_label, col_select, col_button = st.columns([1, 2, 1])
        col_label.markdown(f"**Table {zone.table_id}**")
        chosen = col_select.selectbox(
            f"Waiter for Table {zone.table_id}",
            options=["— Unassigned —"] + waiter_names,
            index=default_idx,
            key=f"assign_select_{zone.table_id}",
            label_visibility="collapsed",
        )
        if col_button.button("Save", key=f"assign_save_{zone.table_id}"):
            if chosen == "— Unassigned —":
                database.assign_waiter(zone.table_id, None, assigned_by=st.session_state.username)
                st.toast(f"Table {zone.table_id} unassigned.")
            else:
                database.assign_waiter(zone.table_id, chosen, assigned_by=st.session_state.username)
                database.create_notification(
                    zone.table_id, f"You've been assigned to Table {zone.table_id}.", chosen
                )
                st.toast(f"Table {zone.table_id} assigned to {chosen}.")


# ==========================================================================
# MANAGER-ONLY: ZONE SETUP TAB (click-to-draw, browser-based)
# ==========================================================================
def render_zone_setup_tab():
    """
    Manager-only table zone setup.

    Displays a preview frame and allows the manager to enter
    table rectangle coordinates manually.
    """

    st.markdown("### 🧭 Define Table Zones")
    st.caption(
        "Enter the rectangle coordinates for each table. "
        "Coordinates are based on the full CCTV frame."
    )

    # ---------------------------------------------------------
    # Load preview frame
    # ---------------------------------------------------------
    preview_source = (
        st.session_state.current_video_source
        or Config.VIDEO_SOURCE
    )

    cap_preview = cv2.VideoCapture(preview_source)

    # Select a frame from the video
    cap_preview.set(cv2.CAP_PROP_POS_FRAMES, 60)

    ok, preview_frame = cap_preview.read()
    cap_preview.release()

    if not ok:
        st.error("Could not read a preview frame from the video source.")
        return

    # Resize to project frame size
    preview_frame = cv2.resize(
        preview_frame,
        (Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
    )

    # Convert BGR -> RGB for Streamlit
    preview_rgb = cv2.cvtColor(
        preview_frame,
        cv2.COLOR_BGR2RGB
    )

    # Display preview
    st.image(
        preview_rgb,
        caption=(
            f"Preview Frame — "
            f"{Config.FRAME_WIDTH} × {Config.FRAME_HEIGHT}"
        ),
        use_container_width=True
    )

    st.divider()

    # ---------------------------------------------------------
    # Existing zones
    # ---------------------------------------------------------
    existing_zones = load_table_zones(
        Config.ZONES_CONFIG_PATH
    )

    if existing_zones:
        st.markdown("### 📍 Current Table Zones")

        existing_rows = []

        for zone in existing_zones:
            existing_rows.append({
                "Table ID": zone.table_id,
                "x1": zone.x1,
                "y1": zone.y1,
                "x2": zone.x2,
                "y2": zone.y2,
            })

        st.dataframe(
            existing_rows,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # ---------------------------------------------------------
    # Number of tables
    # ---------------------------------------------------------
    st.markdown("### 🪑 Set Table Coordinates")

    num_tables = st.number_input(
        "Number of tables",
        min_value=1,
        max_value=20,
        value=max(len(existing_zones), 2),
        step=1,
    )

    drawn_zones = []

    # ---------------------------------------------------------
    # Coordinate input for each table
    # ---------------------------------------------------------
    for table_id in range(1, int(num_tables) + 1):

        # Try to find existing zone
        existing_zone = None

        for zone in existing_zones:
            if zone.table_id == table_id:
                existing_zone = zone
                break

        st.markdown(f"#### 🪑 Table {table_id}")

        if existing_zone:
            default_x1 = existing_zone.x1
            default_y1 = existing_zone.y1
            default_x2 = existing_zone.x2
            default_y2 = existing_zone.y2
        else:
            default_x1 = 0
            default_y1 = 0
            default_x2 = 200
            default_y2 = 200

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            x1 = st.number_input(
                "Left (x1)",
                min_value=0,
                max_value=Config.FRAME_WIDTH,
                value=int(default_x1),
                key=f"table_{table_id}_x1",
            )

        with col2:
            y1 = st.number_input(
                "Top (y1)",
                min_value=0,
                max_value=Config.FRAME_HEIGHT,
                value=int(default_y1),
                key=f"table_{table_id}_y1",
            )

        with col3:
            x2 = st.number_input(
                "Right (x2)",
                min_value=0,
                max_value=Config.FRAME_WIDTH,
                value=int(default_x2),
                key=f"table_{table_id}_x2",
            )

        with col4:
            y2 = st.number_input(
                "Bottom (y2)",
                min_value=0,
                max_value=Config.FRAME_HEIGHT,
                value=int(default_y2),
                key=f"table_{table_id}_y2",
            )

        # Validate coordinates
        if x2 <= x1:
            st.warning(
                f"Table {table_id}: x2 must be greater than x1."
            )

        if y2 <= y1:
            st.warning(
                f"Table {table_id}: y2 must be greater than y1."
            )

        drawn_zones.append(
            TableZone(
                table_id=table_id,
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
            )
        )

    # ---------------------------------------------------------
    # Save zones
    # ---------------------------------------------------------
    st.divider()

    if st.button(
        "💾 Save These Zones",
        type="primary",
        use_container_width=True
    ):

        valid_zones = []

        for zone in drawn_zones:

            if zone.x2 <= zone.x1:
                st.error(
                    f"Table {zone.table_id}: "
                    f"x2 must be greater than x1."
                )
                return

            if zone.y2 <= zone.y1:
                st.error(
                    f"Table {zone.table_id}: "
                    f"y2 must be greater than y1."
                )
                return

            valid_zones.append(zone)

        save_table_zones(
            valid_zones,
            Config.ZONES_CONFIG_PATH
        )

        st.success(
            f"Saved {len(valid_zones)} table zone(s) successfully!"
        )

        st.info(
            "Click ⏹ Stop and then ▶ Start in Live Monitor "
            "to apply the updated table zones."
        )


# ==========================================================================
# MANAGER-ONLY: REPORTS & ANALYTICS TAB
# ==========================================================================
def render_reports_tab():
    """
    Full reports/analytics - table turnover history, food-service timing,
    and CSV export. Manager-only, per the access rules.
    """
    st.markdown("### 📊 Daily Occupancy Report")
    selected_date = st.date_input("Select a date", value=datetime.now())
    date_str = selected_date.strftime("%Y-%m-%d")

    logs = database.get_logs_for_date(date_str)
    if logs:
        report_rows = []
        durations = []
        for log_id, table_number, entry_time, exit_time, duration_sec in logs:
            report_rows.append({
                "Log ID": log_id,
                "Table": table_number,
                "Entry Time": entry_time,
                "Exit Time": exit_time if exit_time else "Still occupied",
                "Duration (sec)": duration_sec if duration_sec is not None else "-",
            })
            if duration_sec is not None:
                durations.append(duration_sec)

        st.dataframe(report_rows, use_container_width=True, hide_index=True)

        if durations:
            avg_turnover = sum(durations) / len(durations)
            st.metric("Average Table Turnover", format_duration(avg_turnover))

        csv_path = database.export_daily_csv(date_str)
        with open(csv_path, "rb") as csv_file:
            st.download_button(
                "⬇ Download Occupancy CSV", data=csv_file,
                file_name=os.path.basename(csv_path), mime="text/csv",
            )
    else:
        st.info(f"No occupancy records found for {date_str} yet.")

    st.divider()
    st.markdown("### 🍽️ Food Service Timing")
    events = database.get_service_events_for_date(date_str)
    if events:
        serve_rows = []
        serve_times = []
        for eid, table_number, log_id, waiter_username, served_at, seconds_to_serve in events:
            serve_rows.append({
                "Table": table_number,
                "Waiter": waiter_username,
                "Served At": served_at,
                "Time to Serve": format_duration(seconds_to_serve) if seconds_to_serve and seconds_to_serve > 0 else "-",
            })
            if seconds_to_serve and seconds_to_serve > 0:
                serve_times.append(seconds_to_serve)

        st.dataframe(serve_rows, use_container_width=True, hide_index=True)
        if serve_times:
            st.metric("Average Time to Serve Food", format_duration(sum(serve_times) / len(serve_times)))
    else:
        st.info(f"No food-service events recorded for {date_str} yet.")

    st.divider()
    st.markdown("### 👥 Current Assignments")
    assignments = database.get_assignments()
    if assignments:
        assign_rows = [
            {"Table": t, "Waiter": a["waiter_username"] or "Unassigned", "Assigned By": a["assigned_by"],
             "Assigned At": a["assigned_at"]}
            for t, a in assignments.items()
        ]
        st.dataframe(assign_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No table assignments made yet.")


# ==========================================================================
# MAIN LAYOUT - ROLE-BASED TABS
# ==========================================================================
render_header(f"{st.session_state.full_name} · {st.session_state.role.capitalize()} station")
role = st.session_state.role

if role == "manager":
    tab_live, tab_status, tab_assign, tab_zones, tab_reports = st.tabs(
        ["🎥 Live Monitor (CCTV)", "📋 Table Status", "👥 Assign Waiters", "🧭 Zone Setup", "📊 Reports"]
    )
    with tab_live:
        render_live_monitor_tab()
    with tab_status:
        render_table_status_board()
    with tab_assign:
        render_assignment_tab()
    with tab_zones:
        render_zone_setup_tab()
    with tab_reports:
        render_reports_tab()

elif role == "captain":
    tab_status, tab_assign = st.tabs(["📋 Table Status", "👥 Assign Waiters"])
    with tab_status:
        st.caption("CCTV video is restricted to Managers - showing live table status only.")
        render_table_status_board()
    with tab_assign:
        render_assignment_tab()

else:  # waiter
    my_tables = database.get_tables_for_waiter(st.session_state.username)
    st.markdown("### 🧑‍🍳 My Tables")
    if not my_tables:
        st.info("You have no tables assigned yet. Ask your Manager/Captain to assign you one.")
    else:
        render_table_status_board(filter_table_ids=my_tables)
