"""
QArm text writer
================
Writes any text made of A-Z, 0-9, space, '-' and newlines with the QArm.

Usage
-----
    python qarm_write_text.py "HELLO"              # plan, then write
    python qarm_write_text.py "HELLO" --dry-run    # plan and report only, no motion
    python qarm_write_text.py "LINE ONE\nLINE 2"   # \n starts a new line

How it works (same idea as the Cornell Writing_Robot_Arm project)
-----------------------------------------------------------------
1. Each character is a hand-designed list of strokes. A stroke is a polyline of
   (u, v) points in a unit box: u = 0..1 across the letter width, v = 0..1 up the
   letter height. The glyph is scaled by LETTER_WIDTH / LETTER_HEIGHT, so size is a
   single setting.
2. layout_text() places glyphs on the paper, offset from the measured pen-contact
   position (your PHI_WRITE pose, run through forward kinematics).
3. build_cartesian_path() turns strokes into a dense list of Cartesian waypoints:
   lift, travel at lift height, lower, draw in STROKE_STEP increments, lift.
4. plan_joint_path() runs IK on EVERY waypoint before the arm moves. It rejects
   unreachable points, joint-limit violations and IK branch flips (a large jump in
   joint angles between neighbouring waypoints).
5. Only if the whole plan is valid does the script connect and execute it.

Axis assumptions are the same as your original script: paper lies in the XY plane,
Z points up, x is the writing direction, +y is "up the page".
"""

import argparse
import time

import numpy as np

from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities


# ============================================================
# CONFIGURATION
# ============================================================

# Change to True (or pass --dry-run) to plan without moving the arm.
DRY_RUN = False

HOME = np.deg2rad([0.0, 0.0, 0.0, 0.0])

# Your measured pen-contact joint configuration.
PHI_WRITE = np.deg2rad([-10.0, 65.0, -40.0, -10.0])

# Pen approach and drawing settings (metres).
APPROACH_HEIGHT = 0.025
PEN_LIFT = 0.015

LETTER_HEIGHT = 0.10
LETTER_WIDTH = 0.07
LETTER_GAP = 0.012      # horizontal gap between letters
LINE_GAP = 0.03         # vertical gap between lines of text
STROKE_STEP = 0.005     # spacing of waypoints while the pen is down
TRAVEL_STEP = 0.015     # spacing of waypoints while the pen is lifted

# Joint limits from your installed QArmUtilities source.
LOWER = np.deg2rad([-170, -80, -95, -160])
UPPER = np.deg2rad([170, 80, 75, 160])

# Preflight rejects a plan if consecutive waypoints differ by more than this in
# any joint. Large jumps usually mean the IK switched solution branch.
MAX_JOINT_JUMP = np.deg2rad(20.0)
IK_POSITION_TOLERANCE = 0.005   # metres

# Timing and tolerances.
MOVE_TIMEOUT = 12.0
JOINT_TOLERANCE = np.deg2rad(3.0)
WAYPOINT_PAUSE = 0.15

# QArm.write_position clips the gripper command to 0.1 (open) .. 0.9 (closed).
# Do NOT command "closed" far beyond the point where the pen stops the fingers: the
# gripper motor then stalls against the pen for the whole run and trips its overload
# protection ("persistent load that exceeded the maximum output rating").
# Find the LOWEST value that holds the pen with qarm_grip_test.py, then either set
# GRIPPER_HOLD below (e.g. np.array([0.45])) or pass --grip 0.45 on the command line.
GRIPPER_OPEN = np.array([0.1])
GRIPPER_HOLD = np.array([0.88])
LED_RED = np.array([1.0, 0.0, 0.0])
LED_GREEN = np.array([0.0, 1.0, 0.0])


# ============================================================
# GLYPH DATABASE
# ============================================================
# Each glyph is a list of strokes; each stroke is a polyline of (u, v) points.
# The pen stays down along a stroke and lifts between strokes.
# Straight segments only, so curves are approximated by corner points.

_O_SHAPE = [(0.25, 0), (0, 0.2), (0, 0.8), (0.25, 1), (0.75, 1), (1, 0.8),
            (1, 0.2), (0.75, 0), (0.25, 0)]

GLYPHS = {
    " ": [],
    "-": [[(0, 0.5), (1, 0.5)]],

    "A": [[(0, 0), (0.5, 1), (1, 0)], [(0.2, 0.4), (0.8, 0.4)]],
    "B": [[(0, 0), (0, 1), (0.7, 1), (0.95, 0.85), (0.95, 0.65), (0.7, 0.5),
           (0, 0.5), (0.7, 0.5), (1, 0.35), (1, 0.15), (0.7, 0), (0, 0)]],
    "C": [[(1, 0.85), (0.75, 1), (0.25, 1), (0, 0.8), (0, 0.2), (0.25, 0),
           (0.75, 0), (1, 0.15)]],
    "D": [[(0, 0), (0, 1), (0.6, 1), (1, 0.7), (1, 0.3), (0.6, 0), (0, 0)]],
    "E": [[(1, 1), (0, 1), (0, 0), (1, 0)], [(0, 0.5), (0.8, 0.5)]],
    "F": [[(1, 1), (0, 1), (0, 0)], [(0, 0.5), (0.8, 0.5)]],
    "G": [[(1, 0.85), (0.75, 1), (0.25, 1), (0, 0.8), (0, 0.2), (0.25, 0),
           (0.75, 0), (1, 0.2), (1, 0.45), (0.55, 0.45)]],
    "H": [[(0, 1), (0, 0), (0, 0.5), (1, 0.5), (1, 1), (1, 0)]],
    "I": [[(0.2, 1), (0.8, 1), (0.5, 1), (0.5, 0), (0.2, 0), (0.8, 0)]],
    "J": [[(1, 1), (1, 0.2), (0.75, 0), (0.25, 0), (0, 0.2)]],
    "K": [[(0, 1), (0, 0), (0, 0.45), (1, 1), (0, 0.45), (1, 0)]],
    "L": [[(0, 1), (0, 0), (1, 0)]],
    "M": [[(0, 0), (0, 1), (0.5, 0.4), (1, 1), (1, 0)]],
    "N": [[(0, 0), (0, 1), (1, 0), (1, 1)]],
    "O": [list(_O_SHAPE)],
    "P": [[(0, 0), (0, 1), (0.75, 1), (1, 0.85), (1, 0.65), (0.75, 0.5), (0, 0.5)]],
    "Q": [list(_O_SHAPE), [(0.6, 0.3), (1, 0)]],
    "R": [[(0, 0), (0, 1), (0.75, 1), (1, 0.85), (1, 0.65), (0.75, 0.5),
           (0, 0.5), (0.5, 0.5), (1, 0)]],
    "S": [[(1, 0.85), (0.75, 1), (0.25, 1), (0, 0.8), (0, 0.65), (0.25, 0.5),
           (0.75, 0.5), (1, 0.35), (1, 0.2), (0.75, 0), (0.25, 0), (0, 0.15)]],
    "T": [[(0, 1), (1, 1), (0.5, 1), (0.5, 0)]],
    "U": [[(0, 1), (0, 0.2), (0.25, 0), (0.75, 0), (1, 0.2), (1, 1)]],
    "V": [[(0, 1), (0.5, 0), (1, 1)]],
    "W": [[(0, 1), (0.25, 0), (0.5, 0.6), (0.75, 0), (1, 1)]],
    "X": [[(0, 1), (1, 0)], [(1, 1), (0, 0)]],
    "Y": [[(0, 1), (0.5, 0.5), (1, 1), (0.5, 0.5), (0.5, 0)]],
    "Z": [[(0, 1), (1, 1), (0, 0), (1, 0)]],

    "0": [list(_O_SHAPE)],
    "1": [[(0.2, 0.8), (0.5, 1), (0.5, 0)], [(0.2, 0), (0.8, 0)]],
    "2": [[(0, 0.8), (0.25, 1), (0.75, 1), (1, 0.8), (1, 0.65), (0, 0), (1, 0)]],
    "3": [[(0, 0.85), (0.25, 1), (0.75, 1), (1, 0.85), (1, 0.65), (0.75, 0.52),
           (0.35, 0.52), (0.75, 0.52), (1, 0.38), (1, 0.15), (0.75, 0),
           (0.25, 0), (0, 0.15)]],
    "4": [[(0.75, 0), (0.75, 1), (0, 0.3), (1, 0.3)]],
    "5": [[(1, 1), (0.1, 1), (0, 0.55), (0.7, 0.6), (1, 0.4), (1, 0.2),
           (0.75, 0), (0.25, 0), (0, 0.15)]],
    "6": [[(0.9, 1), (0.4, 1), (0, 0.6), (0, 0.2), (0.25, 0), (0.75, 0),
           (1, 0.2), (1, 0.4), (0.75, 0.55), (0.25, 0.55), (0, 0.4)]],
    "7": [[(0, 1), (1, 1), (0.35, 0)]],
    "8": [[(0.25, 0.52), (0, 0.7), (0, 0.85), (0.25, 1), (0.75, 1), (1, 0.85),
           (1, 0.7), (0.75, 0.52), (0.25, 0.52), (0, 0.35), (0, 0.15),
           (0.25, 0), (0.75, 0), (1, 0.15), (1, 0.35), (0.75, 0.52)]],
    "9": [[(1, 0.6), (0.75, 0.45), (0.25, 0.45), (0, 0.6), (0, 0.8), (0.25, 1),
           (0.75, 1), (1, 0.8), (1, 0.2), (0.75, 0), (0.25, 0), (0, 0.1)]],
}


# ============================================================
# HELPERS
# ============================================================

def read_angles(arm):
    arm.read_std()
    return np.array(arm.measJointPosition[:4], dtype=float).copy()


def set_grip(value):
    """Set the gripper command used while the pen is held (0.1 open .. 0.9 closed)."""
    global GRIPPER_HOLD
    value = float(value)
    if not 0.1 <= value <= 0.9:
        raise ValueError("Gripper command must be between 0.1 (open) and 0.9 (closed).")
    if value > 0.8:
        print(f"WARNING: gripper hold {value} is close to a full squeeze. If the pen "
              "stops the fingers earlier, the motor will stall against it.")
    GRIPPER_HOLD = np.array([value])


def require_grip():
    if GRIPPER_HOLD is None:
        raise ValueError(
            "No gripper hold value is set. Run qarm_grip_test.py to find the lowest "
            "value that holds your pen, then pass --grip VALUE (or set GRIPPER_HOLD)."
        )


class FaultWatch:
    """
    Transparent proxy around arm.card. QArm.read_std() and write_position() catch
    the driver's HILError and only print it, so a script would keep sending
    commands into a fault. This proxy passes every call straight through but counts
    exceptions raised by read/write so the script can stop at the first one.
    """
    _WATCHED = ("read", "write", "task_read")

    def __init__(self, card):
        object.__setattr__(self, "_card", card)
        object.__setattr__(self, "faults", 0)
        object.__setattr__(self, "last_fault", "")

    def __getattr__(self, name):
        attr = getattr(self._card, name)
        if name in self._WATCHED and callable(attr):
            def wrapped(*args, **kwargs):
                try:
                    return attr(*args, **kwargs)
                except Exception as error:
                    getter = getattr(error, "get_error_message", None)
                    object.__setattr__(self, "faults", self.faults + 1)
                    object.__setattr__(self, "last_fault", getter() if getter else str(error))
                    raise
            return wrapped
        return attr


def watch_faults(arm):
    """Call once right after QArm(...) so hardware faults stop the script."""
    arm.card = FaultWatch(arm.card)
    return arm.card


def check_faults(arm):
    card = getattr(arm, "card", None)
    if isinstance(card, FaultWatch) and card.faults:
        raise RuntimeError(
            f"QArm reported a hardware fault: {card.last_fault} "
            "Stopping. Power-cycle the arm before trying again."
        )


def print_pose(label, angles):
    print(f"{label}: {np.round(np.rad2deg(angles), 2)} degrees")


def send_joint_target(arm, target, description, open_gripper=False, quiet=False):
    """Send a joint target and wait for measured feedback."""
    target = np.asarray(target, dtype=float)

    if target.shape != (4,):
        raise ValueError("Joint target must contain four angles.")

    if np.any(target < LOWER) or np.any(target > UPPER):
        raise ValueError(
            f"Target outside configured joint limits: "
            f"{np.round(np.rad2deg(target), 2)} degrees"
        )

    if not quiet:
        print(f"\n{description}")
        print_pose("Target", target)

    if open_gripper:
        gripper = GRIPPER_OPEN
    else:
        require_grip()
        gripper = GRIPPER_HOLD
    arm.write_position(phiCMD=target.copy(), gprCMD=gripper.copy())
    check_faults(arm)

    start = time.monotonic()
    while time.monotonic() - start < MOVE_TIMEOUT:
        measured = read_angles(arm)
        check_faults(arm)
        if np.all(np.abs(measured - target) <= JOINT_TOLERANCE):
            time.sleep(WAYPOINT_PAUSE)
            return measured
        time.sleep(0.05)

    raise TimeoutError(f"Arm did not reach target in time: {description}")


def solve_ik(utils, point, gamma, previous):
    """
    Inverse kinematics for one Cartesian waypoint.
    point: end-effector position in metres; gamma: wrist angle in radians.
    The result is verified with forward kinematics.
    """
    point = np.asarray(point, dtype=float)
    _, best = utils.inverse_kinematics(
        point, float(gamma), np.asarray(previous, dtype=float)
    )
    target = np.asarray(best, dtype=float)

    if target.shape != (4,) or not np.all(np.isfinite(target)):
        raise ValueError(f"Invalid IK solution at point {point}")

    if np.any(target < LOWER) or np.any(target > UPPER):
        raise ValueError(
            f"IK solution outside configured limits at {np.round(point, 4)}: "
            f"{np.round(np.rad2deg(target), 2)} degrees"
        )

    achieved, _ = utils.forward_kinematics(target)
    error = np.linalg.norm(np.asarray(achieved, dtype=float).ravel() - point)
    if error > IK_POSITION_TOLERANCE:
        raise ValueError(
            f"IK position error is {error:.4f} m at {np.round(point, 4)}. "
            "The point may be unreachable or the IK solution invalid."
        )
    return target


# ============================================================
# TEXT -> STROKES -> WAYPOINTS -> JOINT PLAN
# ============================================================

def layout_text(origin, text):
    """
    Place glyphs on the paper. origin is the bottom-left corner of the first
    letter (the pen-contact position). Returns [(points_xyz, label), ...],
    one entry per pen-down stroke.
    """
    x0, y0, z0 = (float(v) for v in origin)
    strokes = []

    for row, line in enumerate(text.upper().split("\n")):
        y_base = y0 - row * (LETTER_HEIGHT + LINE_GAP)
        for col, ch in enumerate(line):
            if ch not in GLYPHS:
                raise ValueError(
                    f"No glyph for {ch!r}. Available: "
                    f"{''.join(sorted(GLYPHS))!r}"
                )
            x_base = x0 + col * (LETTER_WIDTH + LETTER_GAP)
            for k, stroke in enumerate(GLYPHS[ch]):
                pts = [
                    [x_base + u * LETTER_WIDTH, y_base + v * LETTER_HEIGHT, z0]
                    for (u, v) in stroke
                ]
                strokes.append((pts, f"line {row + 1} char {col + 1} '{ch}' stroke {k + 1}"))
    return strokes


def line_points(a, b, step):
    """Points from a to b inclusive, spaced at most `step` apart."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
    return [a + (b - a) * t for t in np.linspace(0.0, 1.0, n + 1)]


def build_cartesian_path(p_write, strokes):
    """
    Dense Cartesian waypoint list [(xyz, label), ...] starting from the
    pen-contact point p_write. The pen lifts straight up first, travels between
    strokes at lift height, and returns to a lifted position at the end.
    """
    lift = np.array([0.0, 0.0, PEN_LIFT])
    path = []
    cur = np.asarray(p_write, dtype=float)

    def go(target, step, label):
        nonlocal cur
        target = np.asarray(target, dtype=float)
        if np.linalg.norm(target - cur) < 1e-9:
            return
        for p in line_points(cur, target, step)[1:]:
            path.append((p, label))
        cur = target

    go(cur + lift, STROKE_STEP, "initial lift")

    for pts, label in strokes:
        pts = [np.asarray(p, dtype=float) for p in pts]
        go(pts[0] + lift, TRAVEL_STEP, f"{label}: travel")
        go(pts[0], STROKE_STEP, f"{label}: lower pen")
        for p in pts[1:]:
            go(p, STROKE_STEP, f"{label}: draw")
        go(pts[-1] + lift, STROKE_STEP, f"{label}: lift pen")

    go(np.asarray(p_write, dtype=float) + lift, TRAVEL_STEP, "finish: lift")
    return path


def plan_joint_path(utils, path, gamma, seed):
    """
    Run IK on every waypoint in order, chaining each solution as the seed for
    the next. Raises before any motion if a waypoint is unreachable or the
    joints jump suspiciously.
    """
    plan = []
    previous = np.asarray(seed, dtype=float)

    for i, (point, label) in enumerate(path):
        target = solve_ik(utils, point, gamma, previous)
        jump = np.max(np.abs(target - previous))
        if jump > MAX_JOINT_JUMP:
            raise ValueError(
                f"Joint jump of {np.rad2deg(jump):.1f} degrees at waypoint {i} "
                f"({label}). Likely an IK branch flip or a point near a "
                f"singularity. Try a smaller font or a different start position."
            )
        plan.append((target, label))
        previous = target
    return plan


def print_plan_summary(strokes, path, plan):
    targets = np.array([t for t, _ in plan])
    jumps = np.abs(np.diff(np.vstack([PHI_WRITE, targets]), axis=0))
    print(f"\nPlan OK: {len(strokes)} strokes, {len(plan)} waypoints.")
    print("Joint range used (deg):")
    for j, name in enumerate(["base", "shoulder", "elbow", "wrist"]):
        lo, hi = np.rad2deg(targets[:, j].min()), np.rad2deg(targets[:, j].max())
        print(f"  {name:9s} {lo:8.2f} .. {hi:8.2f}   "
              f"(limits {np.rad2deg(LOWER[j]):.0f} .. {np.rad2deg(UPPER[j]):.0f})")
    print(f"Largest joint step between waypoints: "
          f"{np.rad2deg(jumps.max()):.2f} degrees")


# ============================================================
# MAIN SEQUENCE
# ============================================================

def main(text, dry_run):
    arm = None

    try:
        utils = QArmUtilities()

        # --- Plan everything before touching the hardware -------------------
        p_write, _ = utils.forward_kinematics(PHI_WRITE)
        p_write = np.asarray(p_write, dtype=float).ravel()
        gamma = PHI_WRITE[3]

        print("Known writing joint target:")
        print_pose("Writing pose", PHI_WRITE)
        print("End-effector position from FK (metres):", p_write)
        print("Wrist angle gamma (degrees):", np.rad2deg(gamma))

        p_approach = p_write + np.array([0.0, 0.0, APPROACH_HEIGHT])
        phi_approach = solve_ik(utils, p_approach, gamma, HOME)
        print_pose("Approach pose", phi_approach)

        strokes = layout_text(p_write, text)
        path = build_cartesian_path(p_write, strokes)
        plan = plan_joint_path(utils, path, gamma, PHI_WRITE)
        print_plan_summary(strokes, path, plan)

        print(f"\nText to write: {text!r}")

        if dry_run:
            print(
                "\nDRY RUN: no movement commands were sent. Check the joint "
                "range and the largest step above, then run again without "
                "--dry-run."
            )
            return

        # --- Connect and execute -------------------------------------------
        require_grip()      # refuse to start without a calibrated gripper value
        print("\nConnecting to physical QArm...")
        print("Keep the arm's complete movement area clear.")
        input(
            "\nVerify the driver warning has been resolved, the pen is secure, "
            "and the work area is clear. Press ENTER to proceed, or Ctrl+C to "
            "cancel: "
        )

        arm = QArm(hardware=1)
        watch_faults(arm)
        arm.write_led(LED_GREEN)

        send_joint_target(arm, HOME, "1. Move to HOME", open_gripper=True)
        send_joint_target(arm, phi_approach, "2. Move above writing point",
                          open_gripper=True)

        print("\nThe arm is above the writing point. Insert and secure the pen now.")
        input("Press ENTER after the pen is in the middle of the gripper "
              "and your hands are clear: ")

        send_joint_target(arm, PHI_WRITE, "3. Move to writing pose")

        print(f"\n4. Writing {text!r}...")
        last_label = None
        for target, label in plan:
            if label != last_label:
                print(f"  {label}")
                last_label = label
            send_joint_target(arm, target, label, quiet=True)

        send_joint_target(arm, HOME, "5. Return to HOME")
        print("\nFinished writing.")

    except KeyboardInterrupt:
        print("\nCancelled by user.")

    except Exception as error:
        print(f"\nERROR: {error}")
        print("Do not restart until the cause has been checked.")

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
    parser = argparse.ArgumentParser(description="Write text with the QArm.")
    parser.add_argument("text", nargs="?", default="HI",
                        help="text to write (A-Z, 0-9, space, '-'; use \\n for a new line)")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan and validate only; never move the arm")
    parser.add_argument("--grip", type=float, metavar="V",
                        help="gripper command that holds the pen (0.1 open .. 0.9 closed); "
                             "find it with qarm_grip_test.py")
    args = parser.parse_args()
    if args.grip is not None:
        set_grip(args.grip)

    main(args.text.replace("\\n", "\n"), DRY_RUN or args.dry_run)
