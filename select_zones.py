"""
select_zones.py
----------------
Interactive helper tool to define TABLE_ZONES coordinates by clicking
directly on a frame from your video source, instead of guessing pixel
numbers from a grid.

HOW TO USE:
    1. Run: python select_zones.py
    2. A window opens showing one frame from your configured VIDEO_SOURCE
       (resized to the same 1280x720 canvas the main app uses).
    3. For EACH table, click and drag a rectangle with your mouse:
          - Left-click and hold at the TOP-LEFT corner of the table area
          - Drag to the BOTTOM-RIGHT corner
          - Release the mouse button to confirm that table's zone
    4. After releasing, the rectangle is drawn and numbered automatically.
    5. Press 'n' to move to drawing the NEXT table's zone.
    6. Press 'r' to redo/clear the CURRENT table's rectangle if you made
       a mistake before pressing 'n'.
    7. Press 'q' when you're done defining all tables you want.
    8. The tool prints ready-to-paste Python code for TABLE_ZONES in
       utils.py, and also saves it to zones_output.txt.

This does NOT modify utils.py automatically - you copy/paste the printed
code yourself, so you can review it before applying it.
"""

import cv2
from utils import Config, TableZone, save_table_zones

# ---- Global state used by the mouse callback ----
drawing = False
start_point = None
current_rect = None       # the rectangle currently being drawn (not yet confirmed)
confirmed_zones = []      # list of (x1, y1, x2, y2) tuples, one per finalized table
display_frame = None
base_frame = None


def mouse_callback(event, x, y, flags, param):
    """
    OpenCV mouse callback: handles click-drag-release to draw one
    rectangle at a time for the table currently being defined.
    """
    global drawing, start_point, current_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_rect = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_rect = (start_point[0], start_point[1], x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Normalize so x1<x2 and y1<y2 regardless of drag direction
        x1, y1 = start_point
        x2, y2 = x, y
        current_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def redraw():
    """
    Redraw the base frame plus all confirmed zones (green, numbered) and
    the in-progress rectangle currently being dragged (yellow), then show it.
    """
    global display_frame
    display_frame = base_frame.copy()

    # Draw all already-confirmed table zones in green with their table ID.
    for idx, (x1, y1, x2, y2) in enumerate(confirmed_zones, start=1):
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, f"Table {idx}", (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw the rectangle currently being dragged (not yet confirmed) in yellow.
    if current_rect is not None:
        x1, y1, x2, y2 = current_rect
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Instructions overlay at the bottom of the frame.
    instructions = "Drag to draw a table zone | 'n'=confirm & next | 'r'=redo current | 'q'=finish"
    cv2.rectangle(display_frame, (0, 690), (1280, 720), (20, 20, 20), -1)
    cv2.putText(display_frame, instructions, (10, 712),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Define Table Zones - " + Config.WINDOW_NAME, display_frame)


def main():
    """
    Load one frame from the configured video source, then run the
    interactive click-drag-confirm loop for defining table zones.
    """
    global base_frame, current_rect

    print(f"[SelectZones] Opening video source: {Config.VIDEO_SOURCE}")
    capture = cv2.VideoCapture(Config.VIDEO_SOURCE)
    if not capture.isOpened():
        print("[SelectZones] ERROR: could not open video source.")
        return

    # Skip a few frames in so we land on a frame with people visible
    # (rather than a blank opening frame), then read it.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 60)
    success, frame = capture.read()
    capture.release()

    if not success:
        print("[SelectZones] ERROR: could not read a frame from the video.")
        return

    # Resize to the exact canvas size the main app uses, so the coordinates
    # you draw here will line up perfectly with TABLE_ZONES at runtime.
    base_frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))

    cv2.namedWindow("Define Table Zones - " + Config.WINDOW_NAME)
    cv2.setMouseCallback("Define Table Zones - " + Config.WINDOW_NAME, mouse_callback)

    print("\nInstructions:")
    print("  1. Click-drag from top-left to bottom-right of a table's area.")
    print("  2. Press 'n' to confirm that rectangle and move to the next table.")
    print("  3. Press 'r' to clear the current (unconfirmed) rectangle and redo it.")
    print("  4. Press 'q' when finished defining all tables.\n")

    while True:
        redraw()
        key = cv2.waitKey(30) & 0xFF

        if key == ord('n'):
            if current_rect is not None:
                confirmed_zones.append(current_rect)
                print(f"[SelectZones] Table {len(confirmed_zones)} confirmed: {current_rect}")
                current_rect = None
            else:
                print("[SelectZones] Draw a rectangle first before pressing 'n'.")

        elif key == ord('r'):
            current_rect = None
            print("[SelectZones] Current rectangle cleared - redraw it.")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    if not confirmed_zones:
        print("[SelectZones] No table zones were confirmed. Exiting without output.")
        return

    # Build TableZone objects and save them to the shared JSON config file,
    # so main.py and streamlit_app.py pick them up automatically - no need
    # to manually paste code into utils.py anymore.
    zone_objects = [
        TableZone(table_id=idx, x1=x1, y1=y1, x2=x2, y2=y2)
        for idx, (x1, y1, x2, y2) in enumerate(confirmed_zones, start=1)
    ]
    save_table_zones(zone_objects, Config.ZONES_CONFIG_PATH)
    print(f"\n[SelectZones] Saved {len(zone_objects)} zones to {Config.ZONES_CONFIG_PATH}")
    print("[SelectZones] main.py and streamlit_app.py will use these automatically on next run.")

    # Also print ready-to-paste Python code, for reference / the project report.
    lines = ["TABLE_ZONES: List[TableZone] = ["]
    for idx, (x1, y1, x2, y2) in enumerate(confirmed_zones, start=1):
        lines.append(f"    TableZone(table_id={idx}, x1={x1}, y1={y1}, x2={x2}, y2={y2}),")
    lines.append("]")
    output_code = "\n".join(lines)

    print("\n" + "=" * 60)
    print("REFERENCE CODE (already saved to table_zones.json above):")
    print("=" * 60)
    print(output_code)
    print("=" * 60)

    with open("zones_output.txt", "w") as f:
        f.write(output_code)
    print("\n[SelectZones] Also saved as text to zones_output.txt for your report.")


if __name__ == "__main__":
    main()
