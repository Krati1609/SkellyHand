"""
handSkeleton
------------
Tracks both hands with MediaPipe's HandLandmarker, then draws a glowing
web between fingertips. Line thickness + brightness react to how close
your hand is to the camera.

WHY things are built the way they are (read this before changing stuff):

1. LIVE_STREAM mode, not VIDEO mode.
   MediaPipe can run in three modes: IMAGE (one-off photos), VIDEO
   (synchronous, blocks until each frame is processed), and LIVE_STREAM
   (asynchronous, runs inference in the background via a callback).
   For a live webcam feed, VIDEO mode would make you wait for inference
   to finish before grabbing the next frame -> visible lag as soon as
   your machine is even slightly loaded. LIVE_STREAM lets the webcam
   keep capturing while detection runs in parallel; we just draw with
   whichever result arrived most recently. One frame of "staleness" is
   invisible to the eye; blocking is not.

2. Depth from wrist->knuckle pixel distance, not MediaPipe's raw z.
   The z landmark MediaPipe gives you is relative-depth and noisy frame
   to frame -> makes the glow flicker instead of scaling smoothly.
   Measuring the on-screen distance between the wrist and the middle
   finger's knuckle is way more stable: that distance visibly grows as
   your hand gets closer to the camera (it's just perspective), and
   shrinks as it moves away.

3. Blurring at half resolution, not full resolution.
   Gaussian blur cost scales with (pixel count x kernel size). Blurring
   a full 640x480 frame with a big kernel every frame is real work.
   Shrinking the glow layer to half size first, blurring THAT, then
   scaling back up gives a near-identical look for a fraction of the
   compute -- this is the single biggest performance win here.

4. Capturing at 640x480, not your webcam's max resolution.
   MediaPipe doesn't need 1080p to find 21 landmarks accurately, and
   every extra pixel is extra work for both detection and blur.

Run:
    pip install opencv-python mediapipe numpy
    (download hand_landmarker.task from MediaPipe's model index, put it
     next to this script)
    python handLandmarker.py

Press 'q' to quit.
"""

import time
import itertools

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# CONFIG - tweak these to change look/feel/performance
# ---------------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
CAM_INDEX = 0
CAPTURE_WIDTH = 640      # internal processing resolution -- keep this LOW for speed
CAPTURE_HEIGHT = 480
DISPLAY_SCALE = 1.8      # window is shown at (capture size * this factor)

# Landmark indices (see MediaPipe's hand landmark diagram - 21 points/hand)
WRIST = 0
MIDDLE_MCP = 9  # used only for the depth proxy, never drawn
FINGERTIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

# Per-finger colors (BGR, since that's OpenCV's convention). A line between
# two DIFFERENT fingers blends their two colors; a line between the SAME
# finger on both hands (see the cross-hand logic below) shows this color
# cleanly, since both endpoints share it.
FINGER_COLORS = {
    "thumb":  (180, 105, 255),  # pink
    "index":  (255, 120, 40),   # blue
    "middle": (0, 220, 255),    # yellow
    "ring":   (100, 255, 100),  # green
    "pinky":  (255, 60, 200),   # magenta/purple
}

# Depth calibration: pixel distance between wrist and middle knuckle at
# arm's length vs. right up close. If the glow barely reacts, or maxes
# out too easily, adjust these two after watching your own webcam feed.
DEPTH_FAR_PX = 35.0
DEPTH_NEAR_PX = 140.0

# Visual tuning
MIN_THICKNESS = 2
MAX_THICKNESS = 7
MIN_DOT_RADIUS = 4        # fingertip circle size, scales with closeness
MAX_DOT_RADIUS = 10
GLOW_DOWNSCALE = 0.5      # render glow at this fraction of frame size
MIN_GLOW_BLUR = 9         # kernel size, must end up odd
MAX_GLOW_BLUR = 25


def _precompute_blended_colors():
    """
    blended_color() used to do numpy math on every single line draw call --
    ~25+ times per frame, 30-60 times a second. But the inputs never
    change (finger colors are constants), so the answer is always the
    same for a given pair of names. Precompute every combination ONCE
    here and just look it up at draw time instead.
    """
    table = {}
    names = list(FINGER_COLORS.keys())
    for n1 in names:
        for n2 in names:
            c1 = np.array(FINGER_COLORS[n1], dtype=np.float32)
            c2 = np.array(FINGER_COLORS[n2], dtype=np.float32)
            table[(n1, n2)] = tuple(((c1 + c2) / 2).astype(int).tolist())
    return table


BLENDED_COLORS = _precompute_blended_colors()


def depth_score(landmarks, w, h):
    """0 (far from camera) .. 1 (close to camera) for one detected hand."""
    wrist = landmarks[WRIST]
    mcp = landmarks[MIDDLE_MCP]
    dx = (wrist.x - mcp.x) * w
    dy = (wrist.y - mcp.y) * h
    dist = (dx ** 2 + dy ** 2) ** 0.5
    score = (dist - DEPTH_FAR_PX) / (DEPTH_NEAR_PX - DEPTH_FAR_PX)
    return float(np.clip(score, 0.0, 1.0))


def landmark_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def draw_glow_line(glow_layer, core_layer, p1, p2, name1, name2, closeness, scale=1.0):
    """
    p1/p2 are full-frame coordinates; `scale` maps them down onto the
    (smaller) glow_layer. core_layer is always drawn at full resolution
    for a crisp bright center line. name1/name2 identify which fingers
    p1/p2 belong to, so we know which color(s) to use.
    """
    thickness = int(MIN_THICKNESS + closeness * (MAX_THICKNESS - MIN_THICKNESS))
    brightness = 0.5 + 0.5 * closeness
    base = BLENDED_COLORS[(name1, name2)]
    color = tuple(int(c * brightness) for c in base)

    gp1 = (int(p1[0] * scale), int(p1[1] * scale))
    gp2 = (int(p2[0] * scale), int(p2[1] * scale))
    cv2.line(glow_layer, gp1, gp2, color, thickness=max(1, thickness * 2), lineType=cv2.LINE_AA)
    # Core line also carries a hint of the finger color rather than plain
    # white, so the color identity reads clearly even up close.
    # Core line stays close to the actual finger color (small boost only)
    # instead of being pushed toward white -- that's what was causing the
    # harsh white glow before.
    core_color = tuple(int(min(255, c * 1.15)) for c in base)
    cv2.line(core_layer, p1, p2, core_color, thickness=1, lineType=cv2.LINE_AA)


def draw_glow_dot(glow_layer, core_layer, p, name, closeness, scale=1.0):
    """
    Draws a small glowing circle at a single fingertip, colored by that
    finger's identity. Same two-layer trick as the lines: a soft blob on
    the glow layer, a crisp bright dot on the core layer.
    """
    radius = int(MIN_DOT_RADIUS + closeness * (MAX_DOT_RADIUS - MIN_DOT_RADIUS))
    brightness = 0.6 + 0.4 * closeness
    color = FINGER_COLORS[name]
    glow_color = tuple(int(c * brightness) for c in color)

    gp = (int(p[0] * scale), int(p[1] * scale))
    cv2.circle(glow_layer, gp, max(1, int(radius * scale * 1.8)), glow_color, -1, lineType=cv2.LINE_AA)

    core_color = tuple(int(min(255, c * 1.15)) for c in color)
    cv2.circle(core_layer, p, radius, core_color, -1, lineType=cv2.LINE_AA)


def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

    # A mutable holder so the async callback can hand results back to the
    # main loop. The callback runs on a MediaPipe-managed thread, so we
    # only ever read/write this single reference from here.
    latest = {"result": None}

    def on_result(result, output_image, timestamp_ms):
        latest["result"] = result

    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    if not cap.isOpened():
        print("Couldn't open the webcam. Check CAM_INDEX.")
        return

    start_time = time.time()
    prev_tick = time.time()
    fps = 0.0

    # Preallocate these ONCE. Creating a fresh np.zeros array every frame
    # means a new memory allocation 30-60 times a second -- cheap-looking
    # in Python but it adds up. We allocate once here and just wipe them
    # back to black (.fill(0)) each frame instead, which is much faster.
    gw, gh = int(CAPTURE_WIDTH * GLOW_DOWNSCALE), int(CAPTURE_HEIGHT * GLOW_DOWNSCALE)
    glow_layer_small = np.zeros((gh, gw, 3), dtype=np.uint8)
    core_layer = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mirror the feed so "your" left hand reads as Left on screen.
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Non-blocking: schedules detection, returns immediately.
        landmarker.detect_async(mp_image, timestamp_ms)

        result = latest["result"]  # most recent finished result, if any

        # Wipe the reused buffers back to black instead of reallocating.
        glow_layer_small.fill(0)
        core_layer.fill(0)

        hands_found = []
        if result and result.hand_landmarks:
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                label = "Unknown"
                if result.handedness and len(result.handedness) > idx:
                    label = result.handedness[idx][0].category_name

                closeness = depth_score(hand_landmarks, w, h)
                tips_px = {
                    name: landmark_to_px(hand_landmarks[i], w, h)
                    for name, i in FINGERTIPS.items()
                }
                hands_found.append({"label": label, "tips": tips_px, "closeness": closeness})

        # web within each hand (all 5 fingertips connect to each other,
        # same as before -- only the CROSS-hand rule changes below)
        for hand in hands_found:
            for name1, name2 in itertools.combinations(FINGERTIPS.keys(), 2):
                p1, p2 = hand["tips"][name1], hand["tips"][name2]
                draw_glow_line(glow_layer_small, core_layer, p1, p2,
                                name1, name2, hand["closeness"], scale=GLOW_DOWNSCALE)

        # web across both hands -- ONLY matching fingers connect now
        # (index-to-index, middle-to-middle, etc.), not every combination.
        if len(hands_found) == 2:
            avg_closeness = (hands_found[0]["closeness"] + hands_found[1]["closeness"]) / 2
            for name in FINGERTIPS.keys():
                p1 = hands_found[0]["tips"][name]
                p2 = hands_found[1]["tips"][name]
                draw_glow_line(glow_layer_small, core_layer, p1, p2,
                                name, name, avg_closeness, scale=GLOW_DOWNSCALE)

        # fingertip dots -- drawn AFTER lines so they sit visibly on top
        for hand in hands_found:
            for name, p in hand["tips"].items():
                draw_glow_dot(glow_layer_small, core_layer, p, name,
                              hand["closeness"], scale=GLOW_DOWNSCALE)

        avg_close_all = np.mean([hh["closeness"] for hh in hands_found]) if hands_found else 0.0
        blur_k = int(MIN_GLOW_BLUR + avg_close_all * (MAX_GLOW_BLUR - MIN_GLOW_BLUR))
        blur_k = blur_k + 1 if blur_k % 2 == 0 else blur_k

        glow_blurred_small = cv2.GaussianBlur(glow_layer_small, (blur_k, blur_k), 0)
        glow_full = cv2.resize(glow_blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)

        out = cv2.addWeighted(frame, 1.0, glow_full, 0.9, 0)
        out = cv2.addWeighted(out, 1.0, core_layer, 0.85, 0)

        # --- FPS + HUD ---
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_tick, 1e-6))
        prev_tick = now

        if len(hands_found) < 2:
            missing = 2 - len(hands_found)
            cv2.putText(out, f"Need {missing} more hand(s) in frame...",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2, cv2.LINE_AA)
        for i, hand in enumerate(hands_found):
            cv2.putText(out, f"{hand['label']} hand | closeness: {hand['closeness']:.2f}",
                        (20, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"FPS: {fps:.1f}", (w - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Upscale ONLY for display. All the tracking/drawing math above
        # runs at the small, fast CAPTURE_WIDTH/HEIGHT resolution -- we
        # just stretch the final image bigger right before showing it,
        # so the window is comfortable to look at without costing any
        # extra detection/blur performance.
        if DISPLAY_SCALE != 1.0:
            display_frame = cv2.resize(
                out, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            display_frame = out

        cv2.imshow("SkellyHand", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()