# Restaurant CCTV Monitoring System

A real-time table occupancy monitoring system for restaurants, built with **YOLOv8-Pose** (Ultralytics) and **OpenCV**. It detects people via CCTV/webcam, determines which predefined table zone they are seated at, tracks occupancy duration, logs visits to a SQLite database, and exports daily CSV reports.

Built as a final-year BSc Computer Science mini-project, demonstrating applied computer vision, object-oriented software design, and database integration.

---

## Features

- **Person + Pose Detection** — YOLOv8-Pose detects people and 17-point body skeletons per frame.
- **Sitting-only occupancy** — Uses knee-bend angle (hip→knee→ankle) from pose keypoints to distinguish **seated customers** from standing staff/passersby, so only sitting people count toward table occupancy.
- **Configurable table zones** — Define rectangular zones per table via simple pixel coordinates.
- **Debounced occupancy state machine** — Avoids status flicker from momentary detection noise.
- **Live dashboard overlay** — Total tables, occupied/empty counts, total customers, per-table time-since-occupied.
- **Visual overlays** — Bounding boxes, pose skeletons, red/green table zone highlighting.
- **SQLite logging** — Every visit's entry time, exit time, and duration is stored.
- **Daily CSV export** — Export or auto-export a daily report on shutdown.

---

## Project Structure

```
restaurant_cctv/
├── main.py               # Desktop entry point - OpenCV window app
├── streamlit_app.py       # RestoMan-Ai - Streamlit web dashboard (alternative UI)
├── detector.py             # YOLOv8-Pose wrapper (person + keypoint detection)
├── tracker.py               # Table occupancy state machine (zone matching, debounce, sitting filter)
├── database.py              # SQLite logging + CSV export
├── dashboard.py              # OpenCV drawing: overlays, skeletons, summary panel
├── utils.py                  # Config, TableZone, geometry/time/posture helpers
├── requirements.txt
├── README.md
├── demo_videos/
│   └── restaurant_demo.webm  # Bundled demo clip
└── logs/                      # Daily CSV reports are saved here
```

---

## Requirements

- **Python 3.11** (Windows, macOS, or Linux)
- A webcam, or a CCTV video file (e.g., `.mp4`)
- VS Code (recommended) or any Python IDE

---

## Installation (Windows + VS Code)

1. **Clone / copy the project folder** to your machine and open it in VS Code.

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

   > On first run, Ultralytics will auto-download the `yolov8n-pose.pt` weights file (~6 MB) if it isn't already present in the project folder.

4. **(Optional) GPU acceleration**: If you have an NVIDIA GPU with CUDA installed, install the CUDA-enabled build of PyTorch for faster inference, then set `DEVICE = "cuda"` in `utils.py`'s `Config` class. Otherwise, `DEVICE = "cpu"` works fine for a demo.

---

## Demo Video

A bundled demo clip is included at **`demo_videos/restaurant_demo.webm`** so you can run and present the app immediately without needing a webcam or live CCTV feed. It's set as the default `VIDEO_SOURCE` in `utils.py`, and the app automatically:
- Resizes every incoming frame to a consistent 1280x720 canvas (so `TABLE_ZONES` coordinates line up regardless of the source video's native resolution)
- Loops the clip back to the start automatically when it ends (`LOOP_VIDEO_FILE = True`), so it plays continuously during a demo/presentation instead of closing

To switch to your **live webcam** instead, open `utils.py` and change:
```python
VIDEO_SOURCE = os.path.join(BASE_DIR, "demo_videos", "restaurant_demo.webm")
```
to:
```python
VIDEO_SOURCE = 0
```

## Configuration

All settings live in **`utils.py`** inside the `Config` class and the `TABLE_ZONES` list:

```python
# Video source: 0 = webcam, or a file path string
VIDEO_SOURCE = 0
# VIDEO_SOURCE = "sample_cctv.mp4"

# Table zones - adjust these rectangle coordinates to match your camera view
TABLE_ZONES = [
    TableZone(table_id=1, x1=40,  y1=80,  x2=280,  y2=340),
    TableZone(table_id=2, x1=320, y1=80,  x2=560,  y2=340),
    TableZone(table_id=3, x1=600, y1=80,  x2=840,  y2=340),
    TableZone(table_id=4, x1=880, y1=80,  x2=1120, y2=340),
]
```

## Setting Table Zone Coordinates

Don't guess pixel coordinates manually — use the included interactive tool:

```powershell
python select_zones.py
```

This opens a window showing a frame from your configured `VIDEO_SOURCE`. For each table:
1. Click and drag from the table's **top-left** to **bottom-right** corner.
2. Press `n` to confirm that rectangle and move to the next table.
3. Press `r` to redo the current rectangle if you made a mistake.
4. Press `q` when you're done defining all tables.

The tool prints ready-to-paste `TABLE_ZONES` code (and saves it to `zones_output.txt`) — copy it into `utils.py`, replacing the existing `TABLE_ZONES` list.

Other tunable parameters:
- `CONFIDENCE_THRESHOLD` — minimum YOLO detection confidence (default `0.45`)
- `OCCUPIED_DEBOUNCE_SECONDS` — how long people must be present before a table flips to Occupied (default `2.0s`)
- `EMPTY_GRACE_SECONDS` — how long a zone must be empty before a table flips back to Empty and is logged (default `5.0s`)
- `SITTING_KNEE_ANGLE_THRESHOLD` — degrees; below this = bent knee = sitting (default `140.0`)

---

## Usage

Run the application:

```powershell
python main.py
```

**Controls (while the video window is focused):**
| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `e` | Export today's report to CSV on demand |

On startup, a window titled **"Restaurant CCTV Monitoring System"** opens, showing:
- Green rectangles for empty tables, red for occupied tables
- Per-table label: Table ID, status, people count, time since occupied
- Bounding boxes + skeleton overlays on every detected person
- A dashboard panel (top-left) with total/occupied/empty tables and total customers

On shutdown (`q` or closing the window), a final CSV report for the day is automatically saved to `logs/daily_report_YYYY-MM-DD.csv`.

---

## RestoMan-Ai — Role-Based Staff Web Dashboard

In addition to the desktop OpenCV window (`main.py`), the project includes **RestoMan-Ai**, a role-based browser dashboard (`streamlit_app.py`) for Managers, Captains, and Waiters. It reuses the exact same `detector.py`, `tracker.py`, `database.py`, and `dashboard.py` modules — no logic is duplicated.

**Run it with:**
```powershell
streamlit run streamlit_app.py
```

### Demo login accounts

| Role | Username | Password |
|---|---|---|
| Manager | `manager` | `manager123` |
| Captain | `captain` | `captain123` |
| Waiter  | `waiter1` | `waiter123` |
| Waiter  | `waiter2` | `waiter123` |

> Passwords are SHA-256 hashed in the database. This is demo-grade auth (no salting/rate-limiting) suitable for an academic project — not a production security model.

### Role permissions

| Feature | Manager | Captain | Waiter |
|---|:---:|:---:|:---:|
| **Live CCTV video** (boxes, skeletons, zones) | ✅ | ❌ | ❌ |
| Non-video live table status board | ✅ | ✅ | ✅ (their tables only) |
| Assign waiters to tables | ✅ | ✅ | ❌ |
| Zone Setup (draw table zones) | ✅ | ❌ | ❌ |
| Mark food as served | — | — | ✅ |
| Reports & analytics (turnover, food-service timing, CSV export) | ✅ | ❌ | ❌ |
| Notifications (occupancy alerts, assignment alerts) | ✅ | ✅ | ✅ (their own) |

**Why only the Manager sees CCTV video:** the Manager's browser session is the only one that opens the video source and runs YOLO detection. Every frame, it writes each table's current status into a shared `live_status` database table. Captain and Waiter sessions never touch the camera or the model — they just poll `live_status` (and their notifications) from the database every couple of seconds. This means monitoring must be actively running in the Manager's Live Monitor tab for Captain/Waiter to see fresh data; a staleness warning appears on their boards if it isn't.

### How the core workflow ties together
1. **Manager** starts monitoring on the Live Monitor tab (video + YOLO detection).
2. When someone sits down, the Manager/Captain has ideally already **assigned a waiter** to that table (Assign Waiters tab). Once the tracker confirms the table Occupied, a notification is sent to the Manager, Captain, and that **specific assigned waiter** (or broadcast to all waiters if the table is unassigned).
3. The assigned **Waiter** sees the table appear in "My Tables", and taps **🍽️ Mark Food Served** once food arrives — this records how long it took from seating to food, visible later in Reports.
4. When the table empties, the visit is closed out with entry/exit time and duration (table turnover).
5. **Manager** reviews all of this — turnover times, food-service timing, assignment history — in the Reports tab, with CSV export.

### Setting Table Zones
The Manager's **Zone Setup** tab lets you draw table rectangles directly on a video frame in the browser (via `streamlit-drawable-canvas`) — no manual coordinate typing needed. Saved zones are written to `table_zones.json` and picked up automatically by both `main.py` and `streamlit_app.py`. The standalone `select_zones.py` command-line tool (see below) does the same thing and writes to the same file, so either method works interchangeably.

---



## Database Schema

SQLite file: `restaurant_monitor.db` (auto-created on first run, including demo user accounts)

**Table: `table_logs`** — one row per table visit

| Column         | Type    | Description                              |
|----------------|---------|-------------------------------------------|
| `id`           | INTEGER | Auto-incrementing primary key              |
| `table_number` | INTEGER | Which table this log entry is for          |
| `entry_time`   | TEXT    | Timestamp when the table became occupied   |
| `exit_time`    | TEXT    | Timestamp when the table became empty (NULL if still occupied) |
| `duration_sec` | REAL    | Total occupied duration in seconds (NULL until exit) |

**Table: `users`** — staff accounts

| Column | Type | Description |
|---|---|---|
| `username` | TEXT | Primary key |
| `password_hash` | TEXT | SHA-256 hashed password |
| `role` | TEXT | `manager`, `captain`, or `waiter` |
| `full_name` | TEXT | Display name |

**Table: `live_status`** — current status of every table (shared source of truth)

| Column | Type | Description |
|---|---|---|
| `table_number` | INTEGER | Primary key |
| `is_occupied` | INTEGER | 0/1 |
| `num_people` | INTEGER | Currently seated count |
| `occupied_since` | TEXT | Entry timestamp of the current visit, if occupied |
| `current_log_id` | INTEGER | FK-style reference to the open `table_logs.id` row |
| `last_updated` | TEXT | Last time this row was refreshed (used to detect stale data) |

**Table: `table_assignments`** — which waiter services each table

| Column | Type | Description |
|---|---|---|
| `table_number` | INTEGER | Primary key |
| `waiter_username` | TEXT | Assigned waiter (NULL if unassigned) |
| `assigned_by` | TEXT | Manager/Captain username who made the assignment |
| `assigned_at` | TEXT | Timestamp of the assignment |

**Table: `service_events`** — food-served tracking

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-incrementing primary key |
| `table_number` | INTEGER | Which table |
| `log_id` | INTEGER | Which visit (`table_logs.id`) this belongs to |
| `waiter_username` | TEXT | Who served the food |
| `served_at` | TEXT | Timestamp food was marked served |
| `seconds_to_serve` | REAL | Seconds between seating and food being served |

**Table: `notifications`** — role/user-targeted alerts

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-incrementing primary key |
| `table_number` | INTEGER | Related table (nullable) |
| `message` | TEXT | Notification text |
| `created_at` | TEXT | Timestamp |
| `target_roles` | TEXT | Comma-separated roles/usernames who can see it |

---

## How It Works (Architecture)

```
Video Frame
    │
    ▼
PersonDetector (detector.py)      → YOLOv8-Pose → bounding boxes + 17 keypoints per person
    │
    ▼
TableTracker (tracker.py)          → filters to SITTING people only (knee-angle heuristic)
                                     → maps each person's hip-midpoint to a table zone
                                     → debounced Empty/Occupied state machine per table
    │
    ├──► DatabaseManager (database.py)  → logs entry/exit/duration to SQLite on state change
    │
    ▼
Dashboard (dashboard.py)            → draws table zones, skeletons, summary panel
    │
    ▼
Display (OpenCV window)
```

---

## Design Notes (for the project report)

- **Object-Oriented Design**: Each concern (detection, tracking, storage, visualization) is its own class in its own module, following the Single Responsibility Principle.
- **Dependency Injection**: `TableTracker` doesn't import `DatabaseManager` directly — `main.py` wires database logging in via callback functions (`on_occupied`, `on_vacated`). This keeps `tracker.py` reusable and independently testable.
- **Debounce state machine**: Prevents flickering occupancy status from momentary detection noise (e.g., a waiter briefly walking through a zone).
- **Sitting-posture heuristic**: Uses the knee-bend angle from pose keypoints (hip→knee→ankle) as the primary signal, with a bounding-box aspect-ratio fallback for when legs are occluded behind a table — a lightweight, explainable approach suitable for a mini-project (as opposed to training a separate posture-classification model).

---

## Possible Extensions

- Multi-camera support (one `PersonDetector` + `TableTracker` pair per camera feed)
- Web-based dashboard (Flask/Streamlit) instead of the OpenCV window
- Waiter/staff re-identification to exclude staff from customer counts more robustly
- Peak-hour occupancy analytics and heatmaps from the SQLite logs

---

## Troubleshooting

- **"Could not open video source"** — Check your webcam is not in use by another app, or that the video file path in `Config.VIDEO_SOURCE` is correct.
- **Slow performance on CPU** — This is expected with `yolov8n-pose.pt` on CPU-only machines; reduce `FRAME_WIDTH`/`FRAME_HEIGHT` in `Config`, or use a GPU if available.
- **Model download fails (no internet)** — Manually download `yolov8n-pose.pt` from the [Ultralytics releases page](https://github.com/ultralytics/assets/releases) and place it in the project root.
