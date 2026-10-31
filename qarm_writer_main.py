"""
Cornell-style writing for the Quanser QArm
==========================================
Applies the method from the Cornell ECE 4760 "Writing Robot Arm" project to the QArm:

  1. Characters are stored as stroke lists (polylines) -- see GLYPHS in qarm_write_text.py.
  2. For a fixed pen height, a lookup table maps pen reach rho -> joint angles.
     The table is built once by sweeping one joint and solving the other so that
     the pen tip stays at constant height (the Cornell equations (1)-(2)).
  3. At run time, writexy(x, y) is only: rho = hypot(x, y); base = atan2(y, x);
     shoulder/elbow = table lookup at rho.  No IK solver, so no branch flips.

Keep this file in the same folder as qarm_write_text.py (it imports the glyph
database and the move helper from it).  PHI_WRITE etc. below are THIS file's
settings; they are separate from the ones in qarm_write_text.py.

QArm geometry used (read from your installed QArmUtilities, so it always matches):
    L1 = LAMBDA_1 shoulder height, L2 = LAMBDA_2 upper arm (with BETA offset),
    L3 = LAMBDA_3 forearm + gripper link (end-effector frame p4 at its end).
In the vertical plane of the arm, with phi1 = shoulder, phi2 = elbow (QArm
convention, phi = 0 is arm up / forearm horizontal), and tool pitch
tau = phi1 + phi2 measured below horizontal:

    rho = L2 sin(phi1 + BETA) + (L3 + Lp) cos(tau)                  (1)
    z   = L1 + L2 cos(phi1 + BETA) - (L3 + Lp) sin(tau)             (2)

where Lp is the pen protrusion beyond p4 (the pen tip is p4 + Lp * tool axis).
These match QArmUtilities.forward_kinematics to ~1e-16 m (checked on random
poses).  Holding the pen tip at height H gives, exactly as in the Cornell derivation,

    tau(phi1) = arcsin( (L1 + L2 cos(phi1 + BETA) - H) / (L3 + Lp) )
    phi2      = tau - phi1

and sweeping phi1 gives rho(phi1); inverting that gives the table.

What is different from the Cornell arm: they have THREE pitch servos, so they can
hold the marker vertical AND at constant height.  The QArm has two pitch joints
(shoulder, elbow); joint 4 is a roll about the tool axis and does not move the tip.
So reach and height are controlled, but the pen tilt tau is whatever the geometry
dictates and changes with reach.  The tables report it so you can see it.

Usage
-----
    python qarm_cornell_writer.py "HELLO" --dry-run         # plan + FK-verify, no motion
    python qarm_cornell_writer.py "HELLO" --center 0.5 -0.1  # place text block centre (m)
    python qarm_cornell_writer.py "HELLO" --grip VALUE       # plan, then write on the arm
                                       (find your grip value first with qarm_grip_test.py;
                                        the script refuses to run on the arm without one)

The hardware sequence mirrors your original script but has NOT been run on a real
arm.  Always do --dry-run first, and place the paper where the printed block
extents say.
"""

import argparse
import time

import numpy as np

from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities

from qarm_write_text import (GLYPHS, send_joint_target, watch_faults,
                             require_grip, set_grip)


# ============================================================
# CONFIGURATION
# ============================================================

HOME = np.deg2rad([0.0, 0.0, 0.0, 0.0])

# Measured pen-contact joint configuration (defines the paper height H and the
# constant wrist-roll angle used while writing).
PHI_WRITE = np.deg2rad([-10.0, 65.0, -40.0, -10.0])

# Distance the pen tip sticks out beyond the end-effector point p4 along the tool
# axis (metres).  0.0 means "p4 is the pen tip".  MEASURE THIS for accurate height:
# with a protruding pen, holding p4 at constant height does NOT keep the tip on
# the paper because the tool tilt changes with reach (this file compensates).

PEN_PROTRUSION = 0 # changed 0 to 0.08 after measuring the protrusion

# Pen-tip heights above the paper used while ramping the pen up/down (metres).
LIFT_LEVELS = [0.0, 0.005, 0.010, 0.015]     # last one is the travel height
APPROACH_HEIGHT = 0.025

# Text block geometry (metres)
LETTER_HEIGHT = 0.05
LETTER_WIDTH = 0.02
LETTER_GAP = 0.007
LINE_GAP = 0.03
STROKE_STEP = 0.005
TRAVEL_STEP = 0.015

# Paper axes: text reads along -y and letters point along +x when seen from
# above with the base behind you (x forward, y left).  Rotate if your paper is
# turned.
PAPER_ROTATION_DEG = 0.0

# Joint limits (same as your QArmUtilities._check_joint_limits)
LOWER = np.deg2rad([-170, -80, -95, -160])
UPPER = np.deg2rad([170, 80, 75, 160])

MAX_JOINT_JUMP = np.deg2rad(10.0)    # between consecutive waypoints
FK_TOLERANCE = 0.0005                # metres, table-vs-real-FK check
TABLE_RHO_STEP = 0.001               # reach resolution of the tables (1 mm)
SWEEP_SAMPLES = 4000

GRIPPER_OPEN = np.array([0.0])
LED_RED = np.array([1.0, 0.0, 0.0])
LED_GREEN = np.array([0.0, 1.0, 0.0])


# ============================================================
# PLANAR MODEL AND TABLES
# ============================================================

class ArmModel:
    """Planar model of the QArm built from the installed QArmUtilities."""

    def __init__(self, utils, pen_protrusion=PEN_PROTRUSION):
        self.utils = utils
        self.beta = utils.BETA
        self.l1 = utils.LAMBDA_1
        self.l2 = utils.LAMBDA_2
        self.l3 = utils.LAMBDA_3
        self.lp = float(pen_protrusion)

    def pen_tip(self, phi):
        """Pen-tip position from the REAL forward kinematics."""
        p4, R04 = self.utils.forward_kinematics(np.asarray(phi, dtype=float))
        return np.asarray(p4, dtype=float).ravel() + self.lp * np.asarray(R04)[:, 2]

    def paper_height(self, phi_write):
        """Pen-tip height at the measured contact pose = the paper plane."""
        return float(self.pen_tip(phi_write)[2])


def _runs(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return list(zip(starts, ends))


def build_table(model, tip_height, seed_phi1=PHI_WRITE[1]):
    """
    Table for one pen-tip height: arrays over a uniform reach grid.

    Returns dict(rho, phi1, phi2, tau, height).  `tau` is the tool pitch below
    horizontal.  Only the branch with the pen pointing downward (tau >= 0) and
    inside the joint limits is kept, and only the monotonic stretch of rho(phi1)
    around the seed, so rho -> angles is a function.
    """
    L = model.l3 + model.lp
    phi1 = np.linspace(LOWER[1], UPPER[1], SWEEP_SAMPLES)

    s = (model.l1 + model.l2 * np.cos(phi1 + model.beta) - tip_height) / L
    ok = np.abs(s) <= 1.0
    tau = np.arcsin(np.clip(s, -1.0, 1.0))
    phi2 = tau - phi1
    ok &= (tau >= 0.0) & (phi2 >= LOWER[2]) & (phi2 <= UPPER[2])
    rho = model.l2 * np.sin(phi1 + model.beta) + L * np.cos(tau)

    runs = _runs(ok)
    if not runs:
        raise ValueError(f"No reachable configuration holds the pen tip at {tip_height:.3f} m.")

    seed = int(np.argmin(np.abs(phi1 - seed_phi1)))
    run = next(((a, b) for a, b in runs if a <= seed <= b), None)
    if run is None:                      # seed not valid at this height: take longest run
        run = max(runs, key=lambda r: r[1] - r[0])
        seed = (run[0] + run[1]) // 2
    a, b = run

    # keep the monotonic stretch of rho around the seed
    d = np.diff(rho[a:b + 1])
    k = min(max(seed - a, 0), len(d) - 1)
    sgn = np.sign(d[k]) if d[k] != 0 else 1.0
    good = np.sign(d) == sgn
    lo = k
    while lo - 1 >= 0 and good[lo - 1]:
        lo -= 1
    hi = k
    while hi + 1 < len(d) and good[hi + 1]:
        hi += 1
    sl = slice(a + lo, a + hi + 2)

    r, p1, p2, t = rho[sl], phi1[sl], phi2[sl], tau[sl]
    if r[0] > r[-1]:
        r, p1, p2, t = r[::-1], p1[::-1], p2[::-1], t[::-1]

    grid = np.arange(np.ceil(r[0] / TABLE_RHO_STEP) * TABLE_RHO_STEP,
                     r[-1], TABLE_RHO_STEP)
    return dict(
        rho=grid,
        phi1=np.interp(grid, r, p1),
        phi2=np.interp(grid, r, p2),
        tau=np.interp(grid, r, t),
        height=tip_height,
    )


class Tables:
    """Tables for every pen height needed, plus the writexy-style lookup."""

    def __init__(self, model, paper_h, levels):
        self.model = model
        self.paper_h = paper_h
        self.by_level = {round(h, 6): build_table(model, paper_h + h) for h in levels}

    def table(self, h):
        return self.by_level[round(h, 6)]

    def rho_range(self):
        lo = max(t["rho"][0] for t in self.by_level.values())
        hi = min(t["rho"][-1] for t in self.by_level.values())
        return lo, hi

    def joints(self, x, y, h, wrist):
        """Cornell writexy(): reach -> table lookup; base = atan2(y, x)."""
        tab = self.table(h)
        rho = float(np.hypot(x, y))
        if rho < tab["rho"][0] or rho > tab["rho"][-1]:
            raise ValueError(
                f"Point ({x:.3f}, {y:.3f}) has reach {rho:.3f} m, outside the table "
                f"range {tab['rho'][0]:.3f}..{tab['rho'][-1]:.3f} m at pen height "
                f"{h * 1000:.0f} mm. Move the text block or make letters smaller."
            )
        phi0 = float(np.arctan2(y, x))
        q = np.array([phi0,
                      np.interp(rho, tab["rho"], tab["phi1"]),
                      np.interp(rho, tab["rho"], tab["phi2"]),
                      wrist])
        if np.any(q < LOWER) or np.any(q > UPPER):
            raise ValueError(f"Joint limits violated at ({x:.3f}, {y:.3f}): "
                             f"{np.round(np.rad2deg(q), 1)} deg")
        return q


# ============================================================
# TEXT -> WORLD STROKES -> WAYPOINTS
# ============================================================

def layout_block(text, center, rotation_deg=PAPER_ROTATION_DEG):
    """
    Returns (strokes, extents). strokes: list of (label, [(x, y), ...]) in the
    robot base frame, one per pen-down stroke. The text block is centred on
    `center` (x, y).
    """
    lines = text.upper().split("\n")
    ncols = max(len(l) for l in lines)
    nrows = len(lines)
    W, Hh = LETTER_WIDTH, LETTER_HEIGHT
    block_w = ncols * W + (ncols - 1) * LETTER_GAP
    block_h = nrows * Hh + (nrows - 1) * LINE_GAP

    c, s = np.cos(np.deg2rad(rotation_deg)), np.sin(np.deg2rad(rotation_deg))

    def to_world(u, v):
        # u: along the reading direction (left to right), v: up the page
        fwd = v - block_h / 2.0
        left = -(u - block_w / 2.0)
        return (center[0] + c * fwd - s * left, center[1] + s * fwd + c * left)

    strokes = []
    for row, line in enumerate(lines):
        v_base = block_h - (row + 1) * Hh - row * LINE_GAP
        for col, ch in enumerate(line):
            if ch not in GLYPHS:
                raise ValueError(f"No glyph for {ch!r}. Available: {''.join(sorted(GLYPHS))!r}")
            u_base = col * (W + LETTER_GAP)
            for k, stroke in enumerate(GLYPHS[ch]):
                pts = [to_world(u_base + u * W, v_base + v * Hh) for (u, v) in stroke]
                strokes.append((f"line {row + 1} char {col + 1} '{ch}' stroke {k + 1}", pts))

    corners = np.array([to_world(0, 0), to_world(block_w, 0),
                        to_world(block_w, block_h), to_world(0, block_h)])
    return strokes, corners


def _xy_line(a, b, step):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
    return [a + (b - a) * t for t in np.linspace(0, 1, n + 1)]


def build_path(strokes):
    """
    Waypoints [(x, y, h, label)], h = pen-tip height above paper.
    Pen is ramped down/up through LIFT_LEVELS (Cornell ramped the pen servo the
    same way) and travels between strokes at the highest level.
    """
    h_up = LIFT_LEVELS[-1]
    path = []
    cur = None
    for label, pts in strokes:
        p0 = np.asarray(pts[0], float)
        if cur is not None:
            for p in _xy_line(cur, p0, TRAVEL_STEP)[1:]:
                path.append((p[0], p[1], h_up, f"{label}: travel"))
        else:
            path.append((p0[0], p0[1], h_up, f"{label}: travel"))
        for h in reversed(LIFT_LEVELS[:-1]):
            path.append((p0[0], p0[1], h, f"{label}: lower pen"))
        for a, b in zip(pts[:-1], pts[1:]):
            for p in _xy_line(a, b, STROKE_STEP)[1:]:
                path.append((p[0], p[1], 0.0, f"{label}: draw"))
        pe = np.asarray(pts[-1], float)
        for h in LIFT_LEVELS[1:]:
            path.append((pe[0], pe[1], h, f"{label}: lift pen"))
        cur = pe
    return path


def plan(tables, path, wrist, seed_pose):
    """Joint plan for every waypoint, with a jump check. Returns list of (q, label)."""
    out, prev = [], np.asarray(seed_pose, float)
    for i, (x, y, h, label) in enumerate(path):
        q = tables.joints(x, y, h, wrist)
        jump = np.max(np.abs(q - prev))
        if jump > MAX_JOINT_JUMP:
            raise ValueError(f"Joint jump of {np.rad2deg(jump):.1f} deg at waypoint {i} ({label}).")
        out.append((q, label))
        prev = q
    return out


def verify_with_real_fk(model, path, joint_plan, paper_h):
    """Run every planned joint vector through the real FK; compare with intent."""
    exy, ez = 0.0, 0.0
    tilt = []
    for (x, y, h, _), (q, _) in zip(path, joint_plan):
        tip = model.pen_tip(q)
        exy = max(exy, float(np.hypot(tip[0] - x, tip[1] - y)))
        ez = max(ez, abs(float(tip[2]) - (paper_h + h)))
        tilt.append(q[1] + q[2])
    return exy, ez, np.rad2deg(np.array(tilt))


def default_center(tables, wrist_az):
    lo, hi = tables.rho_range()
    rho_c = 0.5 * (lo + hi)
    return (rho_c * np.cos(wrist_az), rho_c * np.sin(wrist_az))


def prepare(text, center=None, pen_protrusion=PEN_PROTRUSION, utils=None):
    """Everything up to (but not including) hardware. Used by main() and by the report script."""
    utils = utils or QArmUtilities()
    model = ArmModel(utils, pen_protrusion)
    paper_h = model.paper_height(PHI_WRITE)
    levels = sorted(set(LIFT_LEVELS) | {APPROACH_HEIGHT})
    tables = Tables(model, paper_h, levels)
    tip_c = model.pen_tip(PHI_WRITE)
    az = float(np.arctan2(tip_c[1], tip_c[0]))
    if center is None:
        center = default_center(tables, az)
    strokes, corners = layout_block(text, center)
    path = build_path(strokes)
    wrist = float(PHI_WRITE[3])
    x0, y0 = strokes[0][1][0]
    phi_approach = tables.joints(x0, y0, APPROACH_HEIGHT, wrist)
    jplan = plan(tables, path, wrist, phi_approach)
    exy, ez, tilt = verify_with_real_fk(model, path, jplan, paper_h)
    return dict(utils=utils, model=model, tables=tables, paper_h=paper_h, center=center,
                strokes=strokes, corners=corners, path=path, plan=jplan,
                phi_approach=phi_approach, fk_err_xy=exy, fk_err_z=ez, tilt_deg=tilt)


def print_summary(info, text):
    t = info["tables"]
    lo, hi = t.rho_range()
    plan_q = np.array([q for q, _ in info["plan"]])
    c = info["corners"]
    print(f"\nText: {text!r}   strokes: {len(info['strokes'])}   waypoints: {len(info['plan'])}")
    print(f"Paper plane (pen-tip height at your contact pose): z = {info['paper_h']:.4f} m")
    print(f"Reachable pen reach at that height: {lo:.3f} .. {hi:.3f} m "
          f"(table tool pitch {np.rad2deg(t.table(0.0)['tau'].min()):.0f}..{np.rad2deg(t.table(0.0)['tau'].max()):.0f} deg)")
    print(f"Text block centre: ({info['center'][0]:.3f}, {info['center'][1]:.3f}) m")
    print("Block corners (x, y) in the base frame, metres -- the paper must cover these:")
    for name, p in zip(["start-left/bottom", "right/bottom", "right/top", "left/top"], c):
        print(f"   {name:18s} ({p[0]:.3f}, {p[1]:.3f})   reach {np.hypot(*p):.3f}")
    print("Joint ranges used (deg):")
    for j, name in enumerate(["base", "shoulder", "elbow", "wrist"]):
        print(f"   {name:9s} {np.rad2deg(plan_q[:, j].min()):8.2f} .. {np.rad2deg(plan_q[:, j].max()):8.2f}")
    print(f"Pen tool pitch below horizontal while writing: "
          f"{info['tilt_deg'].min():.1f} .. {info['tilt_deg'].max():.1f} deg")
    print(f"Real-FK check of the table output: max xy error {info['fk_err_xy'] * 1000:.3f} mm, "
          f"max height error {info['fk_err_z'] * 1000:.3f} mm")


# ============================================================
# MAIN
# ============================================================

def main(text, dry_run, center, dump_table=None, pen_protrusion=PEN_PROTRUSION):
    arm = None
    try:
        info = prepare(text, center, pen_protrusion)
        print_summary(info, text)

        if dump_table:
            tab = info["tables"].table(0.0)
            data = np.column_stack([tab["rho"], np.rad2deg(tab["phi1"]),
                                    np.rad2deg(tab["phi2"]), np.rad2deg(tab["tau"])])
            np.savetxt(dump_table, data, delimiter=",", fmt="%.5f",
                       header="rho_m,shoulder_deg,elbow_deg,tool_pitch_deg", comments="")
            print(f"Wrote table to {dump_table}")

        if dry_run:
            print("\nDRY RUN: nothing was sent to the arm.")
            return

        require_grip()      # refuse to start without a calibrated gripper value
        print("\nConnecting to physical QArm. Keep its whole movement area clear.")
        input("Paper placed over the block corners above, pen ready, area clear? "
              "Press ENTER to proceed, or Ctrl+C to cancel: ")

        arm = QArm(hardware=1)
        watch_faults(arm)
        arm.write_led(LED_GREEN)

        send_joint_target(arm, HOME, "1. Move to HOME", open_gripper=True)
        send_joint_target(arm, info["phi_approach"], "2. Move above the first stroke",
                          open_gripper=True)
        input("Put the pen in the middle of the gripper, then press ENTER "
              "with your hands clear: ")

        print(f"\n3. Writing {text!r}...")
        last = None
        for q, label in info["plan"]:
            if label != last:
                print(f"  {label}")
                last = label
            send_joint_target(arm, q, label, quiet=True)

        send_joint_target(arm, HOME, "4. Return to HOME")
        print("\nFinished writing.")

    except KeyboardInterrupt:
        print("\nCancelled by user.")
    except Exception as error:
        print(f"\nERROR: {error}")
        print("Nothing was sent to the arm after this point. Check the cause before retrying.")
    finally:
        if arm is not None:
            try:
                arm.write_led(LED_RED)
            except Exception:
                pass
            try:
                arm.terminate()
            except Exception as error:
                print("Error while disconnecting:", error)
        print("Script ended.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cornell-style text writing on the QArm.")
    ap.add_argument("text", nargs="?", default="SIWES 1")
    ap.add_argument("--dry-run", action="store_true", help="plan and verify only")
    ap.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"),
                    help="text block centre in the base frame (m); default: middle of the reachable band")
    ap.add_argument("--pen-protrusion", type=float, default=PEN_PROTRUSION,
                    help="pen tip distance beyond the end-effector point (m)")
    ap.add_argument("--dump-table", metavar="CSV", help="write the pen-down table to CSV")
    ap.add_argument("--grip", type=float, metavar="V",
                    help="gripper command that holds the pen (0.1 open .. 0.9 closed); "
                         "find it with qarm_grip_test.py. Required to run on the arm.")
    a = ap.parse_args()
    if a.grip is not None:
        set_grip(a.grip)
    main(a.text.replace("\\n", "\n"), a.dry_run, tuple(a.center) if a.center else None,
         a.dump_table, a.pen_protrusion)
