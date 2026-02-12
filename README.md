# G1 Print Monitor

> AI-powered real-time FDM print monitoring. Detects failures via webcam, takes corrective action through OctoPrint, and learns from every print.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

## What It Does

Watches your 3D printer via webcam and automatically:

1. **Detects** 10 types of print failures using image analysis
2. **Decides** the right corrective action based on severity
3. **Intervenes** through the OctoPrint API (pause, adjust temps, cancel)
4. **Alerts** you via WhatsApp/notification with a photo of the issue
5. **Learns** from every print to improve detection over time

## Failure Detection

| Failure | Detection Method | Auto Response |
|---|---|---|
| **Spaghetti** | Edge density in upper frame regions | Pause/Cancel |
| **Layer shift** | Frame-to-frame horizontal displacement | Pause + Alert |
| **Warping** | Corner brightness change vs reference | Increase bed temp, reduce fan |
| **Blob** | Localized bright spot + high edge density | Alert / Pause |
| **Stringing** | Texture analysis between moves | Reduce nozzle temp |
| **Under-extrusion** | High local variance (gap pattern) | Increase flow + temp |
| **Over-extrusion** | Surface quality analysis | Reduce flow |
| **Bed detachment** | Bottom region comparison vs reference | Pause + Alert |
| **Thermal anomaly** | OctoPrint temp deviation monitoring | Emergency stop if critical |
| **First layer issues** | Enhanced monitoring first 10 minutes | Pause + Alert |

## Architecture

```
Webcam --> [Frame Analyzer] --> [Decision Engine] --> [Action Executor] --> OctoPrint API
              |                       |                     |
              v                       v                     v
         Detections:             Actions:              Commands:
         - spaghetti             - alert               - pause
         - layer_shift           - pause               - set temp
         - warping               - adjust_temp         - set flow
         - blob                  - adjust_flow         - set speed
         - thermal               - cancel              - M112 stop
              |
              v
         [Print History] --> Learning database (improves over time)
```

## Quick Start

```python
from g1_monitor.print_monitor import PrintMonitor, MonitorConfig

# Configure
config = MonitorConfig(
    octoprint_url="http://your-printer:5000",
    octoprint_api_key="YOUR_API_KEY",
    snapshot_url="http://your-printer:8080/?action=snapshot",
    snapshot_interval_sec=30,
    auto_pause_enabled=True,
    auto_adjust_enabled=True,
)

# Initialize
monitor = PrintMonitor(config)

# Set reference frame (empty bed before print starts)
ref_frame = monitor.capture_snapshot()
monitor.set_reference_frame(ref_frame)

# Monitor loop
while True:
    frame = monitor.capture_snapshot()
    if frame:
        result = monitor.analyze_frame(frame)

        # Check for alerts
        for alert in monitor.get_pending_alerts():
            print(f"ALERT: {alert['reason']}")
            # Send to WhatsApp, email, etc.

    time.sleep(config.snapshot_interval_sec)

# After print finishes
monitor.record_outcome("success")  # or "failed", "cancelled"
```

## Corrective Actions

The system is **conservative by default**: it alerts before intervening.

### Safety Limits
- Temperature adjustments clamped: nozzle 170-260C, bed 0-110C
- Flow rate clamped: 80-120%
- Speed clamped: 50-150%
- Max 3 auto-adjustments per failure type before escalating to pause
- Alert cooldown: 5 minutes between same failure type

### Configuration

```python
config = MonitorConfig(
    auto_pause_enabled=True,    # Auto-pause on HIGH severity
    auto_cancel_enabled=False,  # Manual cancel only (conservative)
    auto_adjust_enabled=True,   # Auto-adjust temp/flow/speed
    confidence_threshold=0.6,   # Minimum detection confidence
    first_layer_watch_sec=600,  # Extra vigilant first 10 min
    alert_cooldown_sec=300,     # 5 min between repeat alerts
)
```

## Learning System

Every print session is recorded:
- What failures were detected
- What actions were taken
- Final outcome (success/fail/cancel)
- Print settings used

Over time, the system:
- Tracks failure frequency by type
- Calculates overall success rate
- Suggests settings based on past successes
- Identifies patterns (e.g., "warping always happens with this filament")

```python
# Get learning stats
stats = monitor.get_stats()
print(f"Success rate: {stats['history']['success_rate']}")
print(f"Common failures: {stats['history']['common_failures']}")

# Get suggested settings from history
settings = monitor.history.suggest_settings("PLA")
```

## Standalone Analysis

Analyze a single image (no live printer needed):

```python
from g1_monitor.print_monitor import analyze_image

result = analyze_image("failed_print.jpg", reference_path="empty_bed.jpg")
for d in result["detections"]:
    print(f"{d['type']}: {d['description']} ({d['confidence']})")
```

## Components

| Module | Lines | Description |
|---|---|---|
| `print_monitor.py` | 750+ | Complete monitoring system |

### Classes
- **`OctoPrintClient`** - REST API communication with OctoPrint
- **`FrameAnalyzer`** - Image analysis for failure detection (PIL-based, no GPU)
- **`DecisionEngine`** - Maps detections to corrective actions with severity escalation
- **`ActionExecutor`** - Executes actions via OctoPrint (with safety limits)
- **`PrintHistory`** - JSON-based learning database
- **`PrintMonitor`** - Main orchestrator tying everything together

## Hardware Requirements

- Any webcam (USB or IP camera)
- OctoPrint instance (local or remote)
- Python 3.11+ with Pillow
- No GPU required (pure CPU image analysis)

## Roadmap

- [ ] Multi-camera support (top-down + side angle)
- [ ] Deep learning model for higher accuracy detection
- [ ] Time-lapse generation from monitoring snapshots
- [ ] Expected-vs-actual layer comparison using G-code geometry
- [ ] Integration with G1 Print Pipeline for closed-loop optimization
- [ ] Web dashboard for real-time monitoring view

## Part of the G1 Platform

This is one module of the G1 Manufacturing Platform:

| Module | Repo | Status |
|---|---|---|
| Design + Optimize + Slice | [print-pipeline](https://github.com/2233morpheus/print-pipeline) | Done |
| Remote Print Management | [octoprint-remote](https://github.com/2233morpheus/octoprint-remote) | Done |
| **Print Monitoring + AI** | **print-monitor** | **Done** |
| Unified Platform | g1-platform | Planned |

## License

MIT License. See [LICENSE](LICENSE).

## Author

**Khaled Elmajed** ([@2233morpheus](https://github.com/2233morpheus))
