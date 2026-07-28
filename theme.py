"""
theme.py
--------
Visual design system for RestoMan-Ai: "THE PASS" - a kitchen order-ticket
aesthetic grounded in the real materials of restaurant back-of-house work
(dark slate tile floors, brass/copper service ware, dot-matrix order
tickets) rather than a generic dashboard look.

Kept in its own module so streamlit_app.py's application logic isn't
cluttered with large CSS strings, and so the visual language stays
perfectly consistent across every page (login, live monitor, table
board, reports) by importing from one place.

DESIGN TOKENS
    Color:  Charcoal Slate / Brass / Amber Heat / Sage / Ember / Chalk
    Type:   Bebas Neue (display) + IBM Plex Sans (body) + IBM Plex Mono (data)
    Signature: table status rendered as literal order tickets - a torn
               zigzag edge, brass top stripe, heat-lamp glow when occupied.
"""

import streamlit as st

# --------------------------------------------------------------------------
# DESIGN TOKENS
# --------------------------------------------------------------------------
COLORS = {
    "bg_primary": "#1E2124",       # charcoal slate - matches the tile floor in the demo footage
    "bg_panel": "#262A2D",
    "bg_panel_raised": "#2E3235",
    "brass": "#C08A4E",
    "brass_bright": "#D9A466",
    "amber_heat": "#E2A33D",       # occupied - heat-lamp glow
    "sage_empty": "#7C9885",       # empty - muted herb green
    "ember_alert": "#C1443D",      # alerts / critical
    "chalk": "#EDE8DE",            # primary text
    "chalk_dim": "#A8A29A",        # secondary/muted text
}

THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
    --bg-primary: {COLORS['bg_primary']};
    --bg-panel: {COLORS['bg_panel']};
    --bg-panel-raised: {COLORS['bg_panel_raised']};
    --brass: {COLORS['brass']};
    --brass-bright: {COLORS['brass_bright']};
    --amber-heat: {COLORS['amber_heat']};
    --sage-empty: {COLORS['sage_empty']};
    --ember-alert: {COLORS['ember_alert']};
    --chalk: {COLORS['chalk']};
    --chalk-dim: {COLORS['chalk_dim']};
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background: radial-gradient(ellipse at top, #24282b 0%, var(--bg-primary) 55%) !important;
    color: var(--chalk);
}}
[data-testid="stHeader"] {{ background: transparent; }}

* {{ font-family: 'IBM Plex Sans', sans-serif; }}

h1, h2, h3 {{
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.03em;
    color: var(--chalk) !important;
}}
p, span, label, div {{ color: var(--chalk); }}

.rm-mono, code {{ font-family: 'IBM Plex Mono', monospace !important; }}
[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace !important; color: var(--brass-bright) !important; }}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #191C1E 0%, #1E2124 100%);
    border-right: 1px solid rgba(192,138,78,0.25);
}}

/* ---- Buttons: brass service-ware styling ---- */
[data-testid="stButton"] button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {{
    background: linear-gradient(180deg, var(--brass-bright) 0%, var(--brass) 100%) !important;
    color: #1E2124 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 0 rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.25) !important;
    transition: transform 0.08s ease, filter 0.08s ease !important;
}}
[data-testid="stButton"] button:hover, .stDownloadButton button:hover {{ filter: brightness(1.08); }}
[data-testid="stButton"] button:active {{ transform: translateY(1px); box-shadow: 0 1px 0 rgba(0,0,0,0.35) !important; }}

/* ---- Metric badges ---- */
[data-testid="stMetric"] {{
    background: var(--bg-panel);
    border: 1px solid rgba(192,138,78,0.3);
    border-radius: 6px;
    padding: 12px 14px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
}}
[data-testid="stMetricLabel"] {{
    color: var(--chalk-dim) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.72rem !important;
}}

/* ---- Tabs: menu-binder dividers ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 2px solid rgba(192,138,78,0.3);
}}
[data-testid="stTabs"] button[data-baseweb="tab"] {{
    background: var(--bg-panel);
    border-radius: 6px 6px 0 0 !important;
    color: var(--chalk-dim) !important;
    font-weight: 600;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    background: var(--bg-panel-raised) !important;
    color: var(--brass-bright) !important;
    border-bottom: 2px solid var(--brass-bright) !important;
}}

/* ---- Inputs ---- */
input, textarea, select {{
    background: var(--bg-panel) !important;
    color: var(--chalk) !important;
    border: 1px solid rgba(192,138,78,0.35) !important;
    border-radius: 4px !important;
}}
[data-baseweb="select"] > div {{ background: var(--bg-panel) !important; border-color: rgba(192,138,78,0.35) !important; }}

/* ---- Dataframes ---- */
[data-testid="stDataFrame"] {{ border: 1px solid rgba(192,138,78,0.25); border-radius: 6px; overflow: hidden; }}

/* ==========================================================
   SIGNATURE ELEMENT: the order-ticket table status card
   ========================================================== */
.rm-ticket {{
    position: relative;
    background: var(--bg-panel);
    border-radius: 6px 6px 0 0;
    padding: 16px 16px 22px 16px;
    margin-bottom: -4px;
    clip-path: polygon(
        0% 0%, 100% 0%, 100% 92%,
        92% 100%, 84% 92%, 76% 100%, 68% 92%, 60% 100%,
        52% 92%, 44% 100%, 36% 92%, 28% 100%, 20% 92%,
        12% 100%, 4% 92%, 0% 100%
    );
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    border-top: 3px solid var(--status-color, var(--sage-empty));
}}
.rm-ticket--occupied {{
    --status-color: var(--amber-heat);
    animation: rm-heat-glow 2.4s ease-in-out infinite;
}}
.rm-ticket--empty {{ --status-color: var(--sage-empty); }}

@keyframes rm-heat-glow {{
    0%, 100% {{ box-shadow: 0 4px 10px rgba(0,0,0,0.3), 0 0 0px rgba(226,163,61,0); }}
    50% {{ box-shadow: 0 4px 16px rgba(0,0,0,0.35), 0 0 16px rgba(226,163,61,0.35); }}
}}

.rm-ticket-tab {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 0.05em;
    color: var(--status-color, var(--chalk));
}}
.rm-ticket-status {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--chalk-dim);
    margin-bottom: 8px;
}}
.rm-ticket-row {{
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    padding: 3px 0;
    border-bottom: 1px dashed rgba(168,162,154,0.2);
}}
.rm-ticket-row span:last-child {{ font-family: 'IBM Plex Mono', monospace; color: var(--chalk); }}

/* ---- Header wordmark ---- */
.rm-header {{
    display: flex;
    align-items: baseline;
    gap: 14px;
    padding: 4px 0 10px 0;
    border-bottom: 1px solid rgba(192,138,78,0.3);
    margin-bottom: 18px;
    flex-wrap: wrap;
}}
.rm-header-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.6rem;
    letter-spacing: 0.04em;
    color: var(--brass-bright);
    line-height: 1;
}}
.rm-header-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--chalk-dim);
}}

/* ---- Monitor bezel around the live CCTV video ---- */
.rm-monitor-frame {{
    background: #141617;
    border-radius: 10px;
    padding: 14px;
    box-shadow: inset 0 0 0 1px rgba(192,138,78,0.25), 0 6px 18px rgba(0,0,0,0.4);
    background-image:
        radial-gradient(circle at 14px 14px, #6b5636 3px, transparent 4px),
        radial-gradient(circle at calc(100% - 14px) 14px, #6b5636 3px, transparent 4px),
        radial-gradient(circle at 14px calc(100% - 14px), #6b5636 3px, transparent 4px),
        radial-gradient(circle at calc(100% - 14px) calc(100% - 14px), #6b5636 3px, transparent 4px);
}}

/* ---- Login card ---- */
.rm-login-card {{
    background: var(--bg-panel);
    border: 1px solid rgba(192,138,78,0.3);
    border-radius: 10px;
    padding: 30px 32px 12px 32px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    margin-bottom: 16px;
}}
.rm-login-title {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    text-align: center;
    color: var(--brass-bright);
    line-height: 1;
    margin-bottom: 2px;
}}
.rm-login-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    text-align: center;
    color: var(--chalk-dim);
    margin-bottom: 18px;
}}

/* ---- Notification ticket stubs (sidebar) ---- */
.rm-notif {{
    border-left: 2px solid var(--brass);
    padding: 4px 0 4px 10px;
    margin-bottom: 8px;
}}
.rm-notif-time {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--chalk-dim);
}}
.rm-notif-msg {{ font-size: 0.82rem; color: var(--chalk); }}

/* ---- Staff badge (sidebar identity) ---- */
.rm-badge {{
    background: var(--bg-panel);
    border: 1px solid rgba(192,138,78,0.3);
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 6px;
}}
.rm-badge-name {{ font-weight: 600; color: var(--chalk); font-size: 0.95rem; }}
.rm-badge-role {{
    display: inline-block;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 3px;
    background: rgba(192,138,78,0.18);
    color: var(--brass-bright);
}}
</style>
"""


def inject_theme():
    """
    Inject the global CSS theme once at the top of the app (call
    immediately after st.set_page_config).
    """
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_header(subtitle: str = "LIVE FLOOR STATUS — THE PASS"):
    """
    Render the branded RestoMan-Ai wordmark header used at the top of
    every page, with a role/context-specific subtitle underneath.
    """
    st.markdown(
        f"""
        <div class="rm-header">
            <span class="rm-header-title">🛎️ RestoMan-Ai</span>
            <span class="rm-header-sub">{subtitle}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ticket_card_html(table_id: int, occupied: bool, num_people: int,
                             duration_str: str, waiter_name: str, extra_note: str = None) -> str:
    """
    Build the HTML for one table's "order ticket" status card - the
    signature visual element of RestoMan-Ai. Returns a string meant to be
    passed to st.markdown(..., unsafe_allow_html=True).

    Interactive elements (like a "Mark Food Served" button) are placed as
    a normal Streamlit widget directly below this HTML block, not inside
    it, since Streamlit widgets can't live inside a raw HTML string.
    """
    status_class = "rm-ticket--occupied" if occupied else "rm-ticket--empty"
    status_text = "OCCUPIED" if occupied else "AVAILABLE"
    status_icon = "🔴" if occupied else "🟢"
    note_html = (
        f'<div class="rm-ticket-row"><span>{extra_note}</span><span></span></div>'
        if extra_note else ""
    )

    return f"""
    <div class="rm-ticket {status_class}">
        <div class="rm-ticket-tab">TABLE {table_id:02d}</div>
        <div class="rm-ticket-status">{status_icon} {status_text}</div>
        <div class="rm-ticket-row"><span>Seated</span><span>{num_people}</span></div>
        <div class="rm-ticket-row"><span>Time</span><span>{duration_str}</span></div>
        <div class="rm-ticket-row"><span>Waiter</span><span>{waiter_name}</span></div>
        {note_html}
    </div>
    """


def render_staff_badge(full_name: str, role: str):
    """Render the sidebar's staff identity badge."""
    st.markdown(
        f"""
        <div class="rm-badge">
            <div class="rm-badge-name">{full_name}</div>
            <span class="rm-badge-role">{role}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notification_html(created_at: str, message: str) -> str:
    """Build the HTML for one notification ticket-stub entry in the sidebar feed."""
    return f"""
    <div class="rm-notif">
        <div class="rm-notif-time">{created_at}</div>
        <div class="rm-notif-msg">{message}</div>
    </div>
    """