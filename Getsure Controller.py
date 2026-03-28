import cv2, math, time, statistics
from collections import deque
import mediapipe as mp
import vgamepad as vg

# ---------- SETTINGS ----------
CAM_ID = 0
FRAME_WIDTH = 480
FRAME_HEIGHT = 320

STEERING_DEADZONE_DEG = 1.2
SMOOTHING_WINDOW = 4

NEUTRAL_SAMPLE_SEC = 0.8
SWEEP_SAMPLE_SEC = 3.0

GESTURE_HOLD_TIME = 0.07
NITRO_HOLD_TIME = 0.08
NITRO_DEBOUNCE = 0.6

MIRROR_FRAME = True
SWAP_HANDS = False
INVERT_STEERING = False

FALLBACK_SENSITIVITY = 0.06  # used if sweep not performed

# finger indices for curl-based detection (same as previous)
FINGER_INDICES = {
    'thumb': (1,2,3,4),
    'index': (5,6,7,8),
    'middle': (9,10,11,12),
    'ring': (13,14,15,16),
    'pinky': (17,18,19,20)
}

# ---------- helpers ----------
def clamp(x, a, b): return max(a, min(b, x))
def angle_between(p1, p2):
    vx = p2[0]-p1[0]; vy = p2[1]-p1[1]
    return math.degrees(math.atan2(vy, vx))
def vec_from(a,b): return (b[0]-a[0], b[1]-a[1], 0.0)
def dot(u,v): return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
def norm(u): return (u[0]*u[0] + u[1]*u[1] + u[2]*u[2])**0.5 + 1e-8
def angle_between_vecs(u,v):
    ca = dot(u,v)/(norm(u)*norm(v)); ca = max(-1.0, min(1.0, ca))
    return math.degrees(math.acos(ca))

def signed_angle_diff(a, b):
    # shortest signed difference a - b in degrees, result in [-180,180]
    d = (a - b + 180.0) % 360.0 - 180.0
    return d

def landmarks_to_pixels(hand_landmarks, w, h):
    pts = []
    for lm in hand_landmarks.landmark:
        pts.append((int(lm.x * w), int(lm.y * h), lm.z))
    return pts

def finger_curl(pts, finger):
    ids = FINGER_INDICES[finger]
    mcp = pts[ids[0]]; pip = pts[ids[1]]; tip = pts[ids[3]]
    v1 = vec_from(mcp, pip); v2 = vec_from(pip, tip)
    return angle_between_vecs(v1, v2)

def is_finger_extended(pts, finger):
    return finger_curl(pts, finger) < 60.0

def is_fist_by_curl(pts):
    curled = 0
    for name in ('index','middle','ring','pinky'):
        if not is_finger_extended(pts, name):
            curled += 1
    return curled >= 3

def is_index_only_by_curl(pts):
    idx = is_finger_extended(pts, 'index')
    mid = is_finger_extended(pts, 'middle')
    ring = is_finger_extended(pts, 'ring')
    pink = is_finger_extended(pts, 'pinky')
    return idx and (not mid) and (not ring) and (not pink)

def is_grip(pts):
    return is_fist_by_curl(pts) or is_index_only_by_curl(pts)

def stable_wrist_point(pts):
    return ((pts[0][0] + pts[5][0]) // 2, (pts[0][1] + pts[5][1]) // 2)

# ---------- main ----------
def run():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.55, min_tracking_confidence=0.55)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    try:
        gp = vg.VX360Gamepad()
        print("[INFO] virtual gamepad created")
    except Exception as e:
        print("[ERROR] creating virtual gamepad:", e)
        return

    angle_hist = deque(maxlen=SMOOTHING_WINDOW)
    neutral_angle = None        # will store angle in degrees (raw line angle)
    left_extreme = None         # left/right extremes stored as signed diff from 90 deg
    right_extreme = None

    last_nitro_time = 0.0
    gesture_start = {}

    print("[INFO] Press C to AUTO calibrate (neutral sample then sweep). ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if MIRROR_FRAME:
            frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        detected = {'Left': None, 'Right': None}
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, handness in zip(res.multi_hand_landmarks, res.multi_handedness):
                lab = handness.classification[0].label
                if SWAP_HANDS: lab = 'Left' if lab == 'Right' else 'Right'
                pts = landmarks_to_pixels(lm, w, h)
                detected[lab] = (lm, pts)
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        both_present = detected['Left'] is not None and detected['Right'] is not None

        steering_active = False
        steer_value = 0.0

        if both_present:
            _, lpts = detected['Left']; _, rpts = detected['Right']
            left_grip = is_grip(lpts); right_grip = is_grip(rpts)
            if left_grip and right_grip:
                steering_active = True
                lw = stable_wrist_point(lpts)
                rw = stable_wrist_point(rpts)
                raw_ang = angle_between(rw, lw)            #  -180..180
                if raw_ang < 0:
                    raw_ang_norm = raw_ang + 360.0         # 0..360
                else:
                    raw_ang_norm = raw_ang
                if INVERT_STEERING:
                    raw_ang_norm = (raw_ang_norm + 180.0) % 360.0

                # smooth angle (in 0..360 space — smoothing on circular values acceptable here)
                angle_hist.append(raw_ang_norm)
                smooth_ang = sum(angle_hist) / len(angle_hist)

                # compute signed diff from 90 degrees: negative = left, positive = right
                diff = signed_angle_diff(smooth_ang, 90.0)   # -180..180
                # clamp diff to -180..180; typical useful range will be within ~[-90,90]
                # map diff->-1..1 using calibrated extremes if available
                if neutral_angle is not None and left_extreme is not None and right_extreme is not None:
                    # left_extreme and right_extreme stored as diffs
                    left_delta = neutral_angle - left_extreme
                    right_delta = right_extreme - neutral_angle
                    scale = max(abs(left_delta), abs(right_delta), 1e-6)
                    steer_value = clamp((diff - neutral_angle) / scale, -1.0, 1.0)
                elif neutral_angle is not None:
                    # fallback linear mapping from diff (use FALLBACK_SENSITIVITY)
                    steer_value = clamp((diff - neutral_angle) * FALLBACK_SENSITIVITY, -1.0, 1.0)
                else:
                    steer_value = 0.0

                # draw and debug
                cv2.line(frame, rw, lw, (0,220,0), 2)
                cv2.putText(frame, f"Ang: {smooth_ang:.1f}", (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,0), 1)
                cv2.putText(frame, f"Diff(ang-90): {diff:.1f}", (8,36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1)
                cv2.putText(frame, f"SteerVal: {steer_value:.3f}", (8,54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 1)
            else:
                angle_hist.clear()
                cv2.putText(frame, "Steering: both hands must be grip (fist or index-only)", (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,180,220),1)
        else:
            angle_hist.clear()
            cv2.putText(frame, "Steering: both hands not detected", (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,180,220),1)

        # gestures (index-only)
        right_index = False; left_index = False
        now = time.time()
        if detected['Right'] is not None:
            _, rpts = detected['Right']
            if is_index_only_by_curl(rpts):
                if 'r_index' not in gesture_start: gesture_start['r_index'] = now
                elif now - gesture_start['r_index'] > GESTURE_HOLD_TIME: right_index = True
                cv2.putText(frame, "R:index", (w-160,18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,200,0), 1)
            else:
                gesture_start.pop('r_index', None)
        if detected['Left'] is not None:
            _, lpts = detected['Left']
            if is_index_only_by_curl(lpts):
                if 'l_index' not in gesture_start: gesture_start['l_index'] = now
                elif now - gesture_start['l_index'] > GESTURE_HOLD_TIME: left_index = True
                cv2.putText(frame, "L:index", (w-160,36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,200,0), 1)
            else:
                gesture_start.pop('l_index', None)

        # nitro
        if right_index and left_index:
            if 'both_index' not in gesture_start: gesture_start['both_index'] = now
            elif now - gesture_start['both_index'] > GESTURE_HOLD_TIME:
                if now - last_nitro_time > NITRO_DEBOUNCE:
                    gp.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update()
                    time.sleep(NITRO_HOLD_TIME)
                    gp.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A); gp.update()
                    last_nitro_time = time.time()
                    gesture_start['both_index'] = now + NITRO_DEBOUNCE
                cv2.putText(frame, "NITRO", (w//2 - 30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,180,255), 2)
        else:
            if 'both_index' in gesture_start and isinstance(gesture_start['both_index'], float) and gesture_start['both_index'] > now:
                pass
            else:
                gesture_start.pop('both_index', None)

        # triggers
        right_trig = 1.0 if right_index and not (right_index and left_index) else 0.0
        left_trig = 1.0 if left_index and not (right_index and left_index) else 0.0

        if not steering_active:
            steer_value = 0.0

        # write to gamepad
        try:
            gp.left_joystick_float(x_value_float=steer_value, y_value_float=0.0)
            gp.left_trigger_float(value_float=left_trig)
            gp.right_trigger_float(value_float=right_trig)
            gp.update()
        except Exception as e:
            cv2.putText(frame, f"Gamepad error: {e}", (8, FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # show cal readouts
        cal_str = f"CAL: neutral_diff={neutral_angle if neutral_angle is not None else 'None'} left={left_extreme if left_extreme is not None else 'None'} right={right_extreme if right_extreme is not None else 'None'}"
        cv2.putText(frame, cal_str, (8, FRAME_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200), 1)

        cv2.imshow("Fist-Steer AutoCal Fixed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('c'):
            # AUTO CAL: neutral then sweep
            if both_present and detected['Left'] and detected['Right']:
                _, lpts = detected['Left']; _, rpts = detected['Right']
                if is_grip(lpts) and is_grip(rpts):
                    print("[AUTO CAL] sampling neutral...")
                    samples = []
                    t_end = time.time() + NEUTRAL_SAMPLE_SEC
                    while time.time() < t_end:
                        ret2, fr = cap.read()
                        if not ret2: break
                        if MIRROR_FRAME: fr = cv2.flip(fr,1)
                        hh, ww, _ = fr.shape
                        img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                        res2 = hands.process(img)
                        if res2.multi_hand_landmarks and res2.multi_handedness:
                            pts_map = {}
                            for lm2, hnd2 in zip(res2.multi_hand_landmarks, res2.multi_handedness):
                                lb = hnd2.classification[0].label
                                if SWAP_HANDS: lb = 'Left' if lb == 'Right' else 'Right'
                                pts_map[lb] = landmarks_to_pixels(lm2, ww, hh)
                            if 'Left' in pts_map and 'Right' in pts_map:
                                lw2 = stable_wrist_point(pts_map['Left'])
                                rw2 = stable_wrist_point(pts_map['Right'])
                                raw_ang2 = angle_between(rw2, lw2)
                                raw_ang2 = raw_ang2 + 360.0 if raw_ang2 < 0 else raw_ang2
                                if INVERT_STEERING:
                                    raw_ang2 = (raw_ang2 + 180.0) % 360.0
                                diff2 = signed_angle_diff(raw_ang2, 90.0)
                                samples.append(diff2)
                        time.sleep(0.01)
                    if len(samples) > 0:
                        neutral_angle = statistics.mean(samples)
                        print(f"[AUTO CAL] neutral_diff = {neutral_angle:.2f} deg")
                    else:
                        print("[AUTO CAL] neutral sampling failed; try again")
                        continue

                    # sweep
                    print(f"[AUTO CAL] now sweep left/right for {SWEEP_SAMPLE_SEC:.1f}s")
                    sweep = []
                    t_end = time.time() + SWEEP_SAMPLE_SEC
                    while time.time() < t_end:
                        ret2, fr = cap.read()
                        if not ret2: break
                        if MIRROR_FRAME: fr = cv2.flip(fr,1)
                        hh, ww, _ = fr.shape
                        img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                        res2 = hands.process(img)
                        if res2.multi_hand_landmarks and res2.multi_handedness:
                            pts_map = {}
                            for lm2, hnd2 in zip(res2.multi_hand_landmarks, res2.multi_handedness):
                                lb = hnd2.classification[0].label
                                if SWAP_HANDS: lb = 'Left' if lb == 'Right' else 'Right'
                                pts_map[lb] = landmarks_to_pixels(lm2, ww, hh)
                            if 'Left' in pts_map and 'Right' in pts_map:
                                lw2 = stable_wrist_point(pts_map['Left'])
                                rw2 = stable_wrist_point(pts_map['Right'])
                                raw_ang2 = angle_between(rw2, lw2)
                                raw_ang2 = raw_ang2 + 360.0 if raw_ang2 < 0 else raw_ang2
                                if INVERT_STEERING:
                                    raw_ang2 = (raw_ang2 + 180.0) % 360.0
                                diff2 = signed_angle_diff(raw_ang2, 90.0)
                                sweep.append(diff2)
                        time.sleep(0.01)
                    if len(sweep) > 0:
                        left_extreme = min(sweep)
                        right_extreme = max(sweep)
                        print(f"[AUTO CAL] left_extreme = {left_extreme:.2f}, right_extreme = {right_extreme:.2f}")
                        print("[AUTO CAL] calibration complete")
                    else:
                        print("[AUTO CAL] sweep failed; try again")
                else:
                    print("[AUTO CAL] both hands present but not in grip pose; make fists/index-only and try again")
            else:
                print("[AUTO CAL] both hands not detected; position both hands into the frame and press C")

        # manual legacy keys
        elif key == ord('v'):
            if both_present and detected['Left'] and detected['Right']:
                _, lpts = detected['Left']; _, rpts = detected['Right']
                lw = stable_wrist_point(lpts); rw = stable_wrist_point(rpts)
                raw_ang = angle_between(rw, lw); raw_ang = raw_ang + 360.0 if raw_ang < 0 else raw_ang
                if INVERT_STEERING: raw_ang = (raw_ang + 180) % 360.0
                diff = signed_angle_diff(raw_ang, 90.0)
                left_extreme = diff
                print(f"[MANUAL] left_extreme = {left_extreme:.2f}")
            else:
                print("[MANUAL] cannot capture left extreme; both hands not detected")
        elif key == ord('b'):
            if both_present and detected['Left'] and detected['Right']:
                _, lpts = detected['Left']; _, rpts = detected['Right']
                lw = stable_wrist_point(lpts); rw = stable_wrist_point(rpts)
                raw_ang = angle_between(rw, lw); raw_ang = raw_ang + 360.0 if raw_ang < 0 else raw_ang
                if INVERT_STEERING: raw_ang = (raw_ang + 180) % 360.0
                diff = signed_angle_diff(raw_ang, 90.0)
                right_extreme = diff
                print(f"[MANUAL] right_extreme = {right_extreme:.2f}")
            else:
                print("[MANUAL] cannot capture right extreme; both hands not detected")

    # cleanup
    try:
        gp.left_joystick_float(0.0, 0.0); gp.left_trigger_float(0.0); gp.right_trigger_float(0.0); gp.update()
    except:
        pass
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
