# Getsure Controller

A gesture-based virtual gamepad controller that uses your webcam and hand gestures to control games. Hold your hands like a steering wheel and drive — no physical controller needed.

## How It Works

Getsure Controller uses [MediaPipe](https://google.github.io/mediapipe/) to track both hands in real time. The angle formed between your two gripped fists maps to the left joystick (steering), while individual index-finger gestures trigger the accelerator and brake. Pointing both index fingers at the same time fires a configurable "nitro" button (A button). Virtual inputs are sent to games via a [vgamepad](https://github.com/yannbouteiller/vgamepad) Xbox 360 controller emulator.

## Features

- **Two-hand steering** — tilt your fists left/right to steer
- **Accelerator & brake** — point your right or left index finger independently
- **Nitro / boost** — point both index fingers simultaneously to press the A button
- **Auto calibration** — press `C` to sample your neutral position and full sweep range in one step
- **Manual calibration** — press `V` / `B` to lock left/right extremes individually
- **Configurable settings** — mirror, swap hands, invert steering, deadzone, smoothing, and more

## Requirements

- Python 3.8+
- Windows (required by vgamepad / ViGEmBus driver)
- A webcam

### Python Dependencies

```
opencv-python
mediapipe
vgamepad
```

Install them with:

```bash
pip install opencv-python mediapipe vgamepad
```

> **Note:** `vgamepad` requires the [ViGEmBus driver](https://github.com/ViGEm/ViGEmBus/releases) to be installed on your system before running the script.

## Usage

```bash
python "Getsure Controller.py"
```

A camera preview window will open. Position both hands in front of your webcam and follow the calibration steps below.

## Controls

| Gesture | Action |
|---|---|
| Both hands in **fist** (or index-only) → tilt left/right | Steer (left joystick X-axis) |
| **Right** index finger only | Accelerator (right trigger) |
| **Left** index finger only | Brake (left trigger) |
| **Both** index fingers simultaneously | Nitro / boost (A button) |

## Calibration

Calibration tells the controller where your neutral position is and how far you turn left/right.

### Auto Calibration (recommended)

1. Hold both hands in a grip pose (fist or index-only) in front of the camera.
2. Press **`C`**.
3. Hold your hands level (neutral steering) for ~0.8 seconds while the neutral position is sampled.
4. Sweep your hands from full-left to full-right over ~3 seconds.
5. The script prints the captured values and calibration is complete.

### Manual Calibration

| Key | Action |
|---|---|
| `V` | Capture current angle as **left extreme** |
| `B` | Capture current angle as **right extreme** |

Set your neutral position by running auto-cal (`C`) without sweeping, or edit `neutral_angle` directly in the code.

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `C` | Auto calibrate (neutral + sweep) |
| `V` | Set left extreme manually |
| `B` | Set right extreme manually |
| `ESC` | Quit |

## Settings

All settings are at the top of `Getsure Controller.py`:

| Setting | Default | Description |
|---|---|---|
| `CAM_ID` | `0` | Camera device index |
| `FRAME_WIDTH` | `480` | Capture width in pixels |
| `FRAME_HEIGHT` | `320` | Capture height in pixels |
| `STEERING_DEADZONE_DEG` | `1.2` | Angle deadzone in degrees |
| `SMOOTHING_WINDOW` | `4` | Number of frames to average steering angle over |
| `NEUTRAL_SAMPLE_SEC` | `0.8` | Duration (s) to sample neutral position during auto-cal |
| `SWEEP_SAMPLE_SEC` | `3.0` | Duration (s) to sample the steering sweep during auto-cal |
| `GESTURE_HOLD_TIME` | `0.07` | Seconds a gesture must be held before it registers |
| `NITRO_HOLD_TIME` | `0.08` | Duration (s) the A button is held per nitro press |
| `NITRO_DEBOUNCE` | `0.6` | Minimum seconds between nitro activations |
| `MIRROR_FRAME` | `True` | Mirror the camera preview horizontally |
| `SWAP_HANDS` | `False` | Swap which hand MediaPipe labels as Left/Right |
| `INVERT_STEERING` | `False` | Invert the steering direction |
| `FALLBACK_SENSITIVITY` | `0.06` | Steering sensitivity if no sweep calibration was done |

## Troubleshooting

**Virtual gamepad not created**
Make sure the [ViGEmBus driver](https://github.com/ViGEm/ViGEmBus/releases) is installed and you are running on Windows.

**Wrong hand labeled as Left/Right**
Set `SWAP_HANDS = True` in the settings, or reposition your camera.

**Steering feels inverted**
Set `INVERT_STEERING = True` in the settings.

**Camera not opening**
Try changing `CAM_ID` to `1` or another index if you have multiple cameras.

**Steering is too sensitive / not sensitive enough**
Run auto calibration (`C`) to set proper extremes. If you skip the sweep, adjust `FALLBACK_SENSITIVITY`.

## License

This project is provided as-is for personal use. See the repository for any additional license information.
