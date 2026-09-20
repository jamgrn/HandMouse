import ctypes
import json
import math
import time
import urllib.request
from pathlib import Path

try:
    import cv2
    import mediapipe as mp
    import pyautogui
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        HandLandmarksConnections,
        RunningMode,
        drawing_utils,
    )
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark
except ModuleNotFoundError:
    print("依存パッケージが見つかりません。venvのPythonで実行してください。")
    print(r"  .\venv\Scripts\python.exe main.py")
    raise SystemExit(1)

WINDOW_NAME = "Hand Mouse"
VK_ESCAPE = 0x1B
VK_F8 = 0x77
VK_F10 = 0x79
_user32 = ctypes.windll.user32
HOTKEY_GRACE_SEC = 1.0

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
CALIB_PATH = ROOT / "calib.json"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# 画面解像度
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# 平滑化(指数移動平均)。大きいほど反応が早いがガクつく
ALPHA = 0.25

# 手の大きさ(手首〜中指MCP)に対するピンチ判定比率
PINCH_RATIO = 0.38

# スクロール感度。大きいほど同じ動きで多く動く
SCROLL_GAIN = 1200

# キャリブレーション時間(秒)
CALIB_SECONDS = 4.0

# デフォルトの操作領域(正規化座標)。手が動く範囲を画面全体へ引き伸ばす
DEFAULT_REGION = {
    "x_min": 0.20,
    "x_max": 0.80,
    "y_min": 0.15,
    "y_max": 0.80,
}


def ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def load_region() -> dict[str, float]:
    if not CALIB_PATH.exists():
        return dict(DEFAULT_REGION)
    try:
        data = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
        return {key: float(data[key]) for key in DEFAULT_REGION}
    except (OSError, KeyError, TypeError, ValueError):
        return dict(DEFAULT_REGION)


def save_region(region: dict[str, float]) -> None:
    CALIB_PATH.write_text(
        json.dumps(region, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def distance(p1, p2) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def map_to_screen(nx: float, ny: float, region: dict[str, float]) -> tuple[float, float]:
    x_span = max(region["x_max"] - region["x_min"], 0.05)
    y_span = max(region["y_max"] - region["y_min"], 0.05)
    mx = clamp((nx - region["x_min"]) / x_span)
    my = clamp((ny - region["y_min"]) / y_span)
    return mx * screen_w, my * screen_h


def detect_gesture(lm) -> tuple[str, dict[str, float], float]:
    thumb = lm[HandLandmark.THUMB_TIP]
    scale = distance(lm[HandLandmark.WRIST], lm[HandLandmark.MIDDLE_FINGER_MCP]) or 0.2
    pinches = {
        "left": distance(thumb, lm[HandLandmark.INDEX_FINGER_TIP]) / scale,
        "scroll": distance(thumb, lm[HandLandmark.MIDDLE_FINGER_TIP]) / scale,
        "right": distance(thumb, lm[HandLandmark.RING_FINGER_TIP]) / scale,
    }
    closest = min(pinches, key=pinches.get)
    gesture = closest if pinches[closest] < PINCH_RATIO else "move"
    return gesture, pinches, scale


def draw_hud(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 24
    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 22


class Hotkeys:
    """押した瞬間だけ True。ウィンドウフォーカスは不要。"""

    def __init__(self) -> None:
        self._prev = {VK_ESCAPE: False, VK_F8: False, VK_F10: False}
        self.edge(VK_ESCAPE)
        self.edge(VK_F8)
        self.edge(VK_F10)

    def down(self, vk: int) -> bool:
        return bool(_user32.GetAsyncKeyState(vk) & 0x8000)

    def edge(self, vk: int) -> bool:
        now = self.down(vk)
        was = self._prev.get(vk, False)
        self._prev[vk] = now
        return now and not was


def create_landmarker() -> HandLandmarker:
    options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return HandLandmarker.create_from_options(options)


def main() -> None:
    ensure_model()
    region = load_region()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Webカメラを開けませんでした。接続を確認してください。")

    smooth_x, smooth_y = screen_w / 2, screen_h / 2
    is_left_clicking = False
    is_right_clicking = False
    is_scrolling = False
    last_scroll_y = 0.0

    calibrating = False
    calib_until = 0.0
    calib_samples: list[tuple[float, float]] = []

    start_ms = time.perf_counter()
    paused = False
    hotkeys = Hotkeys()
    ready_at = time.perf_counter() + HOTKEY_GRACE_SEC
    frame_i = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        with create_landmarker() as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
                timestamp_ms = int((time.perf_counter() - start_ms) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                gesture = "paused" if paused else "none"
                pinches = {"left": 9.0, "scroll": 9.0, "right": 9.0}

                if result.hand_landmarks:
                    lm = result.hand_landmarks[0]
                    index_tip = lm[HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = lm[HandLandmark.MIDDLE_FINGER_TIP]
                    gesture, pinches, _scale = detect_gesture(lm)

                    if calibrating:
                        calib_samples.append((index_tip.x, index_tip.y))
                        gesture = "calibrate"
                        if time.perf_counter() >= calib_until:
                            xs = [x for x, _y in calib_samples]
                            ys = [y for _x, y in calib_samples]
                            if xs:
                                pad = 0.02
                                region = {
                                    "x_min": max(0.0, min(xs) - pad),
                                    "x_max": min(1.0, max(xs) + pad),
                                    "y_min": max(0.0, min(ys) - pad),
                                    "y_max": min(1.0, max(ys) + pad),
                                }
                                save_region(region)
                            calibrating = False
                            calib_samples = []
                    elif paused:
                        gesture = "paused"
                        is_left_clicking = False
                        is_right_clicking = False
                        is_scrolling = False
                    elif gesture == "scroll":
                        if not is_scrolling:
                            is_scrolling = True
                            last_scroll_y = middle_tip.y
                        else:
                            dy = middle_tip.y - last_scroll_y
                            last_scroll_y = middle_tip.y
                            ticks = int(-dy * SCROLL_GAIN)
                            if ticks:
                                pyautogui.scroll(ticks)
                        is_left_clicking = False
                        is_right_clicking = False
                    else:
                        is_scrolling = False
                        target_x, target_y = map_to_screen(
                            index_tip.x, index_tip.y, region
                        )
                        smooth_x += (target_x - smooth_x) * ALPHA
                        smooth_y += (target_y - smooth_y) * ALPHA
                        pyautogui.moveTo(smooth_x, smooth_y)

                        if gesture == "left" and not is_left_clicking:
                            pyautogui.click()
                            is_left_clicking = True
                        elif gesture != "left":
                            is_left_clicking = False

                        if gesture == "right" and not is_right_clicking:
                            pyautogui.rightClick()
                            is_right_clicking = True
                        elif gesture != "right":
                            is_right_clicking = False

                    drawing_utils.draw_landmarks(
                        frame,
                        lm,
                        HandLandmarksConnections.HAND_CONNECTIONS,
                    )
                else:
                    is_left_clicking = False
                    is_right_clicking = False
                    is_scrolling = False

                status_color = {
                    "left": (0, 220, 0),
                    "right": (0, 165, 255),
                    "scroll": (255, 180, 0),
                    "calibrate": (255, 0, 255),
                    "paused": (0, 0, 255),
                    "move": (200, 200, 200),
                    "none": (120, 120, 120),
                }.get(gesture, (200, 200, 200))

                remaining = (
                    max(0.0, calib_until - time.perf_counter()) if calibrating else 0.0
                )
                draw_hud(
                    frame,
                    [
                        (f"gesture: {gesture}", status_color),
                        (
                            f"pinch L={pinches['left']:.2f}  S={pinches['scroll']:.2f}  "
                            f"R={pinches['right']:.2f}  th={PINCH_RATIO:.2f}",
                            (220, 220, 220),
                        ),
                        (
                            f"region x={region['x_min']:.2f}-{region['x_max']:.2f}  "
                            f"y={region['y_min']:.2f}-{region['y_max']:.2f}",
                            (180, 220, 255),
                        ),
                        (
                            "ESC/F10:quit  F8:pause  c:calib  r:reset"
                            + (f"  calibrating {remaining:.1f}s" if calibrating else ""),
                            (180, 180, 180),
                        ),
                    ],
                )

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                frame_i += 1
                visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                window_closed = frame_i > 10 and visible < 1
                hotkey_ready = time.perf_counter() >= ready_at
                quit_pressed = (
                    key in (ord("q"), 27)
                    or window_closed
                    or (hotkey_ready and (hotkeys.edge(VK_ESCAPE) or hotkeys.edge(VK_F10)))
                )
                if quit_pressed:
                    break
                if key == ord("f") or (hotkey_ready and hotkeys.edge(VK_F8)):
                    paused = not paused
                    is_left_clicking = False
                    is_right_clicking = False
                    is_scrolling = False
                if key == ord("c") and not calibrating:
                    calibrating = True
                    calib_until = time.perf_counter() + CALIB_SECONDS
                    calib_samples = []
                if key == ord("r"):
                    region = dict(DEFAULT_REGION)
                    if CALIB_PATH.exists():
                        CALIB_PATH.unlink()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
