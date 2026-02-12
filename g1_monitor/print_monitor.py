"""
G1 Print Monitor - Real-time FDM print failure detection and intervention.

Monitors a 3D print via webcam snapshots, detects failures, and takes
corrective action through the OctoPrint API.

Part of the G1 Manufacturing Platform.
Author: Khaled Elmajed (@2233morpheus)
"""

import json
import time
import math
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path

try:
    from PIL import Image, ImageFilter, ImageStat, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class FailureType(Enum):
    SPAGHETTI = "spaghetti"
    LAYER_SHIFT = "layer_shift"
    WARPING = "warping"
    BLOB = "blob"
    STRINGING = "stringing"
    UNDER_EXTRUSION = "under_extrusion"
    OVER_EXTRUSION = "over_extrusion"
    BED_DETACH = "bed_detachment"
    THERMAL = "thermal_anomaly"
    FIRST_LAYER = "first_layer_issue"
    UNKNOWN = "unknown"


class Severity(Enum):
    LOW = "low"           # Log + continue
    MEDIUM = "medium"     # Alert user
    HIGH = "high"         # Pause print + alert
    CRITICAL = "critical" # Cancel print + alert


class ActionType(Enum):
    NONE = "none"
    ALERT = "alert"
    PAUSE = "pause"
    CANCEL = "cancel"
    ADJUST_TEMP = "adjust_temp"
    ADJUST_SPEED = "adjust_speed"
    ADJUST_FAN = "adjust_fan"
    ADJUST_FLOW = "adjust_flow"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class Detection:
    """A detected print issue."""
    failure_type: FailureType
    severity: Severity
    confidence: float  # 0.0 - 1.0
    description: str
    region: Optional[tuple] = None  # (x, y, w, h) in image coords
    frame_number: int = 0
    timestamp: float = 0.0


@dataclass
class Action:
    """A corrective action to take."""
    action_type: ActionType
    parameters: dict = field(default_factory=dict)
    reason: str = ""
    detection: Optional[Detection] = None


@dataclass
class PrintState:
    """Current state of the print job."""
    is_printing: bool = False
    current_layer: int = 0
    total_layers: int = 0
    progress_pct: float = 0.0
    nozzle_temp: float = 0.0
    nozzle_target: float = 0.0
    bed_temp: float = 0.0
    bed_target: float = 0.0
    elapsed_sec: float = 0.0
    remaining_sec: float = 0.0
    filename: str = ""


@dataclass
class MonitorConfig:
    """Configuration for the print monitor."""
    octoprint_url: str = "http://localhost:5000"
    octoprint_api_key: str = ""
    snapshot_url: str = "http://localhost:8080/?action=snapshot"
    snapshot_interval_sec: float = 30.0
    alert_cooldown_sec: float = 300.0  # Don't spam alerts
    auto_pause_enabled: bool = True
    auto_cancel_enabled: bool = False  # Conservative default
    auto_adjust_enabled: bool = True
    confidence_threshold: float = 0.6
    first_layer_watch_sec: float = 600.0  # Extra vigilant first 10 min
    snapshot_dir: str = "/tmp/g1-monitor"
    log_path: str = "/tmp/g1-monitor/monitor.log"
    history_path: str = "/tmp/g1-monitor/history.json"


# ---------------------------------------------------------------------------
# OctoPrint API client
# ---------------------------------------------------------------------------

class OctoPrintClient:
    """Communicate with OctoPrint REST API."""

    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _request(self, endpoint: str, method: str = "GET",
                 data: Optional[dict] = None) -> Optional[dict]:
        url = f"{self.base_url}/api/{endpoint}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key

        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 204:
                    return {}
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return {"error": str(e), "code": e.code}
        except Exception as e:
            return {"error": str(e)}

    # -- Status --

    def get_printer_state(self) -> PrintState:
        """Get current printer and job state."""
        state = PrintState()

        printer = self._request("printer")
        if printer and "error" not in printer:
            temps = printer.get("temperature", {})
            tool0 = temps.get("tool0", {})
            bed = temps.get("bed", {})
            state.nozzle_temp = tool0.get("actual", 0)
            state.nozzle_target = tool0.get("target", 0)
            state.bed_temp = bed.get("actual", 0)
            state.bed_target = bed.get("target", 0)
            ps = printer.get("state", {})
            state.is_printing = ps.get("flags", {}).get("printing", False)

        job = self._request("job")
        if job and "error" not in job:
            prog = job.get("progress", {})
            state.progress_pct = prog.get("completion", 0) or 0
            state.elapsed_sec = prog.get("printTime", 0) or 0
            state.remaining_sec = prog.get("printTimeLeft", 0) or 0
            jf = job.get("job", {}).get("file", {})
            state.filename = jf.get("name", "")

        return state

    def get_temperatures(self) -> dict:
        """Get current temperatures."""
        resp = self._request("printer")
        if resp and "error" not in resp:
            return resp.get("temperature", {})
        return {}

    # -- Control --

    def pause(self) -> dict:
        return self._request("job", "POST", {"command": "pause", "action": "pause"})

    def resume(self) -> dict:
        return self._request("job", "POST", {"command": "pause", "action": "resume"})

    def cancel(self) -> dict:
        return self._request("job", "POST", {"command": "cancel"})

    def set_nozzle_temp(self, temp: float) -> dict:
        return self._request("printer/tool", "POST",
                             {"command": "target", "targets": {"tool0": temp}})

    def set_bed_temp(self, temp: float) -> dict:
        return self._request("printer/bed", "POST",
                             {"command": "target", "target": temp})

    def set_feedrate(self, pct: int) -> dict:
        """Set feedrate override (100 = normal speed)."""
        return self._request("printer/printhead", "POST",
                             {"command": "feedrate", "factor": pct})

    def set_flowrate(self, pct: int) -> dict:
        """Set flow rate override (100 = normal)."""
        return self._request("printer/tool", "POST",
                             {"command": "flowrate", "factor": pct})

    def set_fan_speed(self, speed: int) -> dict:
        """Send M106 to set fan speed (0-255)."""
        return self.send_gcode(f"M106 S{max(0, min(255, speed))}")

    def send_gcode(self, command: str) -> dict:
        return self._request("printer/command", "POST",
                             {"command": command})

    def emergency_stop(self) -> dict:
        """M112 emergency stop."""
        return self.send_gcode("M112")

    # -- Webcam --

    def get_snapshot_url(self) -> str:
        settings = self._request("settings")
        if settings and "error" not in settings:
            wc = settings.get("webcam", {})
            return wc.get("snapshotUrl", "")
        return ""


# ---------------------------------------------------------------------------
# Image analyzer - failure detection from webcam frames
# ---------------------------------------------------------------------------

class FrameAnalyzer:
    """
    Analyze webcam frames to detect print failures.

    Uses image processing (no deep learning / no GPU needed):
    - Edge density analysis (spaghetti detection)
    - Color distribution (thermal issues, blob detection)
    - Frame-to-frame difference (layer shift, detachment)
    - Region analysis (first layer, bed adhesion)
    - Texture analysis (stringing, surface quality)
    """

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.reference_frame: Optional[Image.Image] = None
        self.previous_frame: Optional[Image.Image] = None
        self.baseline_stats: Optional[dict] = None
        self.frame_count: int = 0
        self.first_layer_frames: list = []

    def set_reference(self, frame: Image.Image):
        """Set the reference frame (empty bed or first good layer)."""
        self.reference_frame = frame.copy()
        self.baseline_stats = self._compute_stats(frame)

    def analyze(self, frame: Image.Image, print_state: PrintState) -> list[Detection]:
        """
        Analyze a frame and return list of detected issues.
        """
        if not HAS_PIL:
            return []

        detections = []
        self.frame_count += 1
        timestamp = time.time()

        # Convert to common format
        frame = frame.convert("RGB")
        gray = frame.convert("L")
        stats = self._compute_stats(frame)

        # 1. Spaghetti detection (high edge density in unexpected areas)
        spaghetti = self._detect_spaghetti(gray, stats, print_state)
        if spaghetti:
            spaghetti.frame_number = self.frame_count
            spaghetti.timestamp = timestamp
            detections.append(spaghetti)

        # 2. Layer shift (sudden horizontal displacement between frames)
        if self.previous_frame:
            shift = self._detect_layer_shift(gray, self.previous_frame.convert("L"))
            if shift:
                shift.frame_number = self.frame_count
                shift.timestamp = timestamp
                detections.append(shift)

        # 3. Warping / bed detachment
        if self.reference_frame:
            warp = self._detect_warping(frame, print_state)
            if warp:
                warp.frame_number = self.frame_count
                warp.timestamp = timestamp
                detections.append(warp)

        # 4. Blob / over-extrusion detection
        blob = self._detect_blob(gray, stats)
        if blob:
            blob.frame_number = self.frame_count
            blob.timestamp = timestamp
            detections.append(blob)

        # 5. Thermal anomaly (from print state, not vision)
        thermal = self._detect_thermal(print_state)
        if thermal:
            thermal.frame_number = self.frame_count
            thermal.timestamp = timestamp
            detections.append(thermal)

        # 6. Under-extrusion (gaps in layers)
        under_ext = self._detect_under_extrusion(gray, stats)
        if under_ext:
            under_ext.frame_number = self.frame_count
            under_ext.timestamp = timestamp
            detections.append(under_ext)

        # 7. First layer special analysis
        if print_state.elapsed_sec < self.config.first_layer_watch_sec:
            first = self._detect_first_layer_issues(frame, gray, print_state)
            if first:
                first.frame_number = self.frame_count
                first.timestamp = timestamp
                detections.append(first)

        # Update state
        self.previous_frame = frame.copy()

        return detections

    def _compute_stats(self, frame: Image.Image) -> dict:
        """Compute image statistics for analysis."""
        gray = frame.convert("L")
        stat = ImageStat.Stat(gray)

        # Edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)

        # Divide into regions (3x3 grid)
        w, h = frame.size
        regions = {}
        for ry in range(3):
            for rx in range(3):
                box = (rx * w // 3, ry * h // 3,
                       (rx + 1) * w // 3, (ry + 1) * h // 3)
                region = gray.crop(box)
                r_stat = ImageStat.Stat(region)
                r_edges = region.filter(ImageFilter.FIND_EDGES)
                r_edge_stat = ImageStat.Stat(r_edges)
                regions[f"{rx},{ry}"] = {
                    "mean": r_stat.mean[0],
                    "stddev": r_stat.stddev[0],
                    "edge_mean": r_edge_stat.mean[0],
                    "edge_stddev": r_edge_stat.stddev[0],
                }

        return {
            "mean": stat.mean[0],
            "stddev": stat.stddev[0],
            "edge_mean": edge_stat.mean[0],
            "edge_stddev": edge_stat.stddev[0],
            "regions": regions,
            "size": frame.size,
        }

    def _detect_spaghetti(self, gray: Image.Image, stats: dict,
                          print_state: PrintState) -> Optional[Detection]:
        """
        Spaghetti = filament extruding into air, creating tangled mess.
        Detection: High edge density in upper regions of the frame where
        there shouldn't be print geometry. Spaghetti creates chaotic
        edge patterns above the print surface.
        """
        # Check upper third of image for unexpected edge activity
        upper_regions = [stats["regions"].get(f"{x},0", {}) for x in range(3)]
        upper_edge_avg = sum(r.get("edge_mean", 0) for r in upper_regions) / 3

        # Compare against baseline
        if self.baseline_stats:
            baseline_upper = [self.baseline_stats["regions"].get(f"{x},0", {})
                              for x in range(3)]
            baseline_edge_avg = sum(r.get("edge_mean", 0) for r in baseline_upper) / 3

            edge_increase = upper_edge_avg - baseline_edge_avg

            # Significant increase in upper frame edges = possible spaghetti
            if edge_increase > 15:
                confidence = min(1.0, edge_increase / 40)
                severity = Severity.HIGH if confidence > 0.7 else Severity.MEDIUM
                return Detection(
                    failure_type=FailureType.SPAGHETTI,
                    severity=severity,
                    confidence=confidence,
                    description=f"Possible spaghetti detected. Edge density in upper frame "
                                f"increased by {edge_increase:.1f} from baseline.",
                )
        return None

    def _detect_layer_shift(self, current_gray: Image.Image,
                            previous_gray: Image.Image) -> Optional[Detection]:
        """
        Layer shift = sudden horizontal displacement.
        Detection: Large frame-to-frame difference concentrated in
        horizontal bands (shift moves entire layer sideways).
        """
        from PIL import ImageChops

        # Compute absolute difference
        diff = ImageChops.difference(current_gray, previous_gray)
        diff_stat = ImageStat.Stat(diff)

        # High mean difference suggests something changed drastically
        if diff_stat.mean[0] > 25:
            # Check if the change is in a horizontal band pattern
            w, h = current_gray.size
            # Sample horizontal strips
            strip_diffs = []
            for y_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
                y = int(h * y_frac)
                strip_h = max(1, h // 20)
                strip_curr = current_gray.crop((0, y, w, y + strip_h))
                strip_prev = previous_gray.crop((0, y, w, y + strip_h))
                s_diff = ImageChops.difference(strip_curr, strip_prev)
                strip_diffs.append(ImageStat.Stat(s_diff).mean[0])

            # Layer shift: some strips change a lot, others don't
            max_strip = max(strip_diffs)
            min_strip = min(strip_diffs)

            if max_strip > 30 and (max_strip - min_strip) > 15:
                confidence = min(1.0, max_strip / 60)
                return Detection(
                    failure_type=FailureType.LAYER_SHIFT,
                    severity=Severity.HIGH,
                    confidence=confidence,
                    description=f"Possible layer shift. Frame difference: {diff_stat.mean[0]:.1f}, "
                                f"strip variance: {max_strip - min_strip:.1f}",
                )
        return None

    def _detect_warping(self, frame: Image.Image,
                        print_state: PrintState) -> Optional[Detection]:
        """
        Warping = corners lifting from bed.
        Detection: Changes in the bottom region of the print area,
        especially corners showing increased brightness (lifted edge
        catches light differently).
        """
        if not self.reference_frame:
            return None

        from PIL import ImageChops

        # Focus on bottom third (bed level)
        w, h = frame.size
        bottom_curr = frame.crop((0, 2 * h // 3, w, h)).convert("L")
        bottom_ref = self.reference_frame.crop((0, 2 * h // 3, w, h)).convert("L")

        diff = ImageChops.difference(bottom_curr, bottom_ref)
        diff_stat = ImageStat.Stat(diff)

        # Check corners specifically
        corner_size = w // 4
        corners = [
            (0, 0, corner_size, bottom_curr.size[1] // 2),  # bottom-left
            (w - corner_size, 0, w, bottom_curr.size[1] // 2),  # bottom-right
        ]

        max_corner_diff = 0
        for box in corners:
            try:
                corner_diff = ImageChops.difference(
                    bottom_curr.crop(box), bottom_ref.crop(box))
                corner_stat = ImageStat.Stat(corner_diff)
                max_corner_diff = max(max_corner_diff, corner_stat.mean[0])
            except Exception:
                pass

        if max_corner_diff > 20:
            confidence = min(1.0, max_corner_diff / 45)
            return Detection(
                failure_type=FailureType.WARPING,
                severity=Severity.MEDIUM,
                confidence=confidence,
                description=f"Possible warping/lifting. Corner difference: {max_corner_diff:.1f}",
            )
        return None

    def _detect_blob(self, gray: Image.Image, stats: dict) -> Optional[Detection]:
        """
        Blob = excess material buildup at one point.
        Detection: Localized bright spot with high edge density
        (blob creates a raised shiny area).
        """
        # Look for regions with unusually high brightness + edge density
        for key, region in stats["regions"].items():
            if region["mean"] > stats["mean"] + 30 and region["edge_mean"] > stats["edge_mean"] + 10:
                confidence = min(1.0, (region["mean"] - stats["mean"]) / 60)
                if confidence > 0.4:
                    return Detection(
                        failure_type=FailureType.BLOB,
                        severity=Severity.MEDIUM,
                        confidence=confidence,
                        description=f"Possible blob in region {key}. "
                                    f"Brightness: {region['mean']:.0f} vs avg {stats['mean']:.0f}",
                    )
        return None

    def _detect_thermal(self, print_state: PrintState) -> Optional[Detection]:
        """Detect thermal anomalies from OctoPrint temperature data."""
        if print_state.nozzle_target == 0:
            return None

        # Nozzle temp deviation
        nozzle_diff = abs(print_state.nozzle_temp - print_state.nozzle_target)
        if nozzle_diff > 15:
            severity = Severity.CRITICAL if nozzle_diff > 30 else Severity.HIGH
            return Detection(
                failure_type=FailureType.THERMAL,
                severity=severity,
                confidence=0.95,
                description=f"Nozzle temp deviation: {print_state.nozzle_temp:.0f}C "
                            f"vs target {print_state.nozzle_target:.0f}C (diff: {nozzle_diff:.0f}C)",
            )

        # Bed temp deviation
        if print_state.bed_target > 0:
            bed_diff = abs(print_state.bed_temp - print_state.bed_target)
            if bed_diff > 10:
                return Detection(
                    failure_type=FailureType.THERMAL,
                    severity=Severity.HIGH,
                    confidence=0.9,
                    description=f"Bed temp deviation: {print_state.bed_temp:.0f}C "
                                f"vs target {print_state.bed_target:.0f}C (diff: {bed_diff:.0f}C)",
                )
        return None

    def _detect_under_extrusion(self, gray: Image.Image,
                                 stats: dict) -> Optional[Detection]:
        """
        Under-extrusion = gaps between lines, incomplete layers.
        Detection: High local variance in the print area (gaps create
        alternating light/dark pattern).
        """
        # Check middle regions for high texture variance
        mid_regions = [stats["regions"].get(f"{x},1", {}) for x in range(3)]
        avg_stddev = sum(r.get("stddev", 0) for r in mid_regions) / 3

        if self.baseline_stats:
            baseline_mid = [self.baseline_stats["regions"].get(f"{x},1", {})
                            for x in range(3)]
            baseline_stddev = sum(r.get("stddev", 0) for r in baseline_mid) / 3

            stddev_increase = avg_stddev - baseline_stddev
            if stddev_increase > 20:
                confidence = min(1.0, stddev_increase / 50)
                return Detection(
                    failure_type=FailureType.UNDER_EXTRUSION,
                    severity=Severity.MEDIUM,
                    confidence=confidence,
                    description=f"Possible under-extrusion. Texture variance increase: "
                                f"{stddev_increase:.1f}",
                )
        return None

    def _detect_first_layer_issues(self, frame: Image.Image,
                                    gray: Image.Image,
                                    print_state: PrintState) -> Optional[Detection]:
        """
        First layer monitoring - extra sensitive during initial layers.
        Checks for: poor adhesion, inconsistent extrusion, gaps.
        """
        if print_state.elapsed_sec > self.config.first_layer_watch_sec:
            return None

        # Store first layer frames for comparison
        if len(self.first_layer_frames) < 10:
            self.first_layer_frames.append(gray.copy())
            return None

        # After 10 frames, check consistency
        from PIL import ImageChops

        # Compare current to average of first frames
        ref = self.first_layer_frames[0]
        diff = ImageChops.difference(gray, ref)
        diff_stat = ImageStat.Stat(diff)

        # Very low difference = nothing printing (bed adhesion failure)
        if print_state.elapsed_sec > 120 and diff_stat.mean[0] < 3:
            return Detection(
                failure_type=FailureType.FIRST_LAYER,
                severity=Severity.HIGH,
                confidence=0.7,
                description="First layer may not be adhering. Very little change "
                            f"from start after {print_state.elapsed_sec:.0f}s",
            )
        return None


# ---------------------------------------------------------------------------
# Decision engine - maps detections to corrective actions
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Takes detections and decides what corrective actions to take.
    Conservative by default: alert first, intervene only when confident.
    """

    # Mapping: failure type -> (action, parameters)
    RESPONSE_MAP = {
        FailureType.SPAGHETTI: {
            Severity.MEDIUM: ActionType.ALERT,
            Severity.HIGH: ActionType.PAUSE,
            Severity.CRITICAL: ActionType.CANCEL,
        },
        FailureType.LAYER_SHIFT: {
            Severity.MEDIUM: ActionType.ALERT,
            Severity.HIGH: ActionType.PAUSE,
            Severity.CRITICAL: ActionType.PAUSE,
        },
        FailureType.WARPING: {
            Severity.LOW: ActionType.NONE,
            Severity.MEDIUM: ActionType.ADJUST_TEMP,
            Severity.HIGH: ActionType.PAUSE,
        },
        FailureType.BLOB: {
            Severity.LOW: ActionType.NONE,
            Severity.MEDIUM: ActionType.ALERT,
            Severity.HIGH: ActionType.PAUSE,
        },
        FailureType.STRINGING: {
            Severity.LOW: ActionType.NONE,
            Severity.MEDIUM: ActionType.ADJUST_TEMP,
        },
        FailureType.UNDER_EXTRUSION: {
            Severity.LOW: ActionType.NONE,
            Severity.MEDIUM: ActionType.ADJUST_FLOW,
            Severity.HIGH: ActionType.PAUSE,
        },
        FailureType.OVER_EXTRUSION: {
            Severity.LOW: ActionType.NONE,
            Severity.MEDIUM: ActionType.ADJUST_FLOW,
        },
        FailureType.BED_DETACH: {
            Severity.MEDIUM: ActionType.ALERT,
            Severity.HIGH: ActionType.PAUSE,
            Severity.CRITICAL: ActionType.CANCEL,
        },
        FailureType.THERMAL: {
            Severity.MEDIUM: ActionType.ALERT,
            Severity.HIGH: ActionType.PAUSE,
            Severity.CRITICAL: ActionType.EMERGENCY_STOP,
        },
        FailureType.FIRST_LAYER: {
            Severity.MEDIUM: ActionType.ALERT,
            Severity.HIGH: ActionType.PAUSE,
        },
    }

    # Parameters for corrective adjustments
    ADJUSTMENT_MAP = {
        (FailureType.WARPING, ActionType.ADJUST_TEMP): {
            "bed_temp_delta": +5,  # Increase bed temp by 5C
            "fan_speed_pct": -20,  # Reduce fan by 20%
            "speed_pct": -10,      # Slow down 10%
        },
        (FailureType.STRINGING, ActionType.ADJUST_TEMP): {
            "nozzle_temp_delta": -5,  # Decrease nozzle temp by 5C
        },
        (FailureType.UNDER_EXTRUSION, ActionType.ADJUST_FLOW): {
            "flow_delta": +5,         # Increase flow 5%
            "nozzle_temp_delta": +5,  # Bump nozzle temp 5C
        },
        (FailureType.OVER_EXTRUSION, ActionType.ADJUST_FLOW): {
            "flow_delta": -5,  # Decrease flow 5%
        },
    }

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.last_alert_time: dict = {}  # failure_type -> timestamp
        self.adjustment_count: dict = {}  # Track how many times we've adjusted

    def decide(self, detections: list[Detection],
               print_state: PrintState) -> list[Action]:
        """Given detections, return list of actions to take."""
        actions = []

        for det in detections:
            # Skip low confidence detections
            if det.confidence < self.config.confidence_threshold:
                continue

            # Check alert cooldown
            ft = det.failure_type
            now = time.time()
            if ft in self.last_alert_time:
                if now - self.last_alert_time[ft] < self.config.alert_cooldown_sec:
                    continue

            # Look up response
            response_map = self.RESPONSE_MAP.get(ft, {})
            action_type = response_map.get(det.severity, ActionType.ALERT)

            # Check if action is enabled
            if action_type == ActionType.PAUSE and not self.config.auto_pause_enabled:
                action_type = ActionType.ALERT
            if action_type == ActionType.CANCEL and not self.config.auto_cancel_enabled:
                action_type = ActionType.PAUSE
            if action_type in (ActionType.ADJUST_TEMP, ActionType.ADJUST_FAN,
                               ActionType.ADJUST_FLOW, ActionType.ADJUST_SPEED):
                if not self.config.auto_adjust_enabled:
                    action_type = ActionType.ALERT

            # Limit adjustments (don't keep adjusting forever)
            adj_key = (ft, action_type)
            if adj_key in self.adjustment_count and self.adjustment_count[adj_key] >= 3:
                action_type = ActionType.PAUSE  # Adjustments not working, pause
                det.description += " (max adjustments reached, escalating to pause)"

            # Get parameters for adjustments
            params = self.ADJUSTMENT_MAP.get((ft, action_type), {})

            action = Action(
                action_type=action_type,
                parameters=params,
                reason=f"{ft.value}: {det.description} (confidence: {det.confidence:.0%})",
                detection=det,
            )
            actions.append(action)

            # Update cooldown
            self.last_alert_time[ft] = now
            if adj_key not in self.adjustment_count:
                self.adjustment_count[adj_key] = 0
            self.adjustment_count[adj_key] += 1

        return actions


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------

class ActionExecutor:
    """Executes corrective actions via OctoPrint API."""

    def __init__(self, client: OctoPrintClient, config: MonitorConfig):
        self.client = client
        self.config = config
        self.actions_taken: list = []

    def execute(self, action: Action, print_state: PrintState) -> dict:
        """Execute a corrective action. Returns result dict."""
        result = {
            "action": action.action_type.value,
            "reason": action.reason,
            "timestamp": time.time(),
            "success": False,
        }

        try:
            if action.action_type == ActionType.NONE:
                result["success"] = True

            elif action.action_type == ActionType.ALERT:
                # Alert is handled by the monitor (sends WhatsApp)
                result["success"] = True
                result["message"] = action.reason

            elif action.action_type == ActionType.PAUSE:
                resp = self.client.pause()
                result["success"] = "error" not in (resp or {})
                result["response"] = resp

            elif action.action_type == ActionType.CANCEL:
                resp = self.client.cancel()
                result["success"] = "error" not in (resp or {})
                result["response"] = resp

            elif action.action_type == ActionType.EMERGENCY_STOP:
                resp = self.client.emergency_stop()
                result["success"] = True  # Best effort
                result["response"] = resp

            elif action.action_type == ActionType.ADJUST_TEMP:
                params = action.parameters
                if "nozzle_temp_delta" in params:
                    new_temp = print_state.nozzle_target + params["nozzle_temp_delta"]
                    new_temp = max(170, min(260, new_temp))  # Safety clamp
                    self.client.set_nozzle_temp(new_temp)
                    result["nozzle_temp"] = new_temp
                if "bed_temp_delta" in params:
                    new_temp = print_state.bed_target + params["bed_temp_delta"]
                    new_temp = max(0, min(110, new_temp))  # Safety clamp
                    self.client.set_bed_temp(new_temp)
                    result["bed_temp"] = new_temp
                if "fan_speed_pct" in params:
                    # Approximate: reduce fan proportionally
                    fan_cmd = max(0, min(255, int(255 * (100 + params["fan_speed_pct"]) / 100)))
                    self.client.set_fan_speed(fan_cmd)
                    result["fan"] = fan_cmd
                if "speed_pct" in params:
                    new_speed = max(50, min(150, 100 + params["speed_pct"]))
                    self.client.set_feedrate(new_speed)
                    result["feedrate"] = new_speed
                result["success"] = True

            elif action.action_type == ActionType.ADJUST_FLOW:
                params = action.parameters
                if "flow_delta" in params:
                    new_flow = max(80, min(120, 100 + params["flow_delta"]))
                    self.client.set_flowrate(new_flow)
                    result["flow"] = new_flow
                if "nozzle_temp_delta" in params:
                    new_temp = print_state.nozzle_target + params["nozzle_temp_delta"]
                    new_temp = max(170, min(260, new_temp))
                    self.client.set_nozzle_temp(new_temp)
                    result["nozzle_temp"] = new_temp
                result["success"] = True

            elif action.action_type == ActionType.ADJUST_SPEED:
                params = action.parameters
                delta = params.get("speed_pct", -10)
                new_speed = max(50, min(150, 100 + delta))
                self.client.set_feedrate(new_speed)
                result["feedrate"] = new_speed
                result["success"] = True

            elif action.action_type == ActionType.ADJUST_FAN:
                params = action.parameters
                fan = max(0, min(255, params.get("fan_value", 128)))
                self.client.set_fan_speed(fan)
                result["fan"] = fan
                result["success"] = True

        except Exception as e:
            result["error"] = str(e)

        self.actions_taken.append(result)
        return result


# ---------------------------------------------------------------------------
# Print history / learning database
# ---------------------------------------------------------------------------

class PrintHistory:
    """
    Records print outcomes for learning.
    Stores: settings used, failures detected, actions taken, final outcome.
    Over time, builds a database of what works and what doesn't.
    """

    def __init__(self, path: str):
        self.path = path
        self.records: list = []
        self._load()

    def _load(self):
        try:
            with open(self.path, "r") as f:
                self.records = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.records = []

    def save(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2)

    def record_session(self, filename: str, detections: list,
                       actions: list, outcome: str, settings: dict = None):
        """Record a complete print session."""
        record = {
            "timestamp": time.time(),
            "filename": filename,
            "detections": [
                {
                    "type": d.failure_type.value,
                    "severity": d.severity.value,
                    "confidence": d.confidence,
                    "description": d.description,
                    "frame": d.frame_number,
                }
                for d in detections
            ],
            "actions": [
                {
                    "type": a.action_type.value,
                    "reason": a.reason,
                    "params": a.parameters,
                }
                for a in actions
            ],
            "outcome": outcome,  # "success", "failed", "cancelled"
            "settings": settings or {},
        }
        self.records.append(record)
        self.save()

    def get_failure_stats(self) -> dict:
        """Get failure frequency by type."""
        stats = {}
        for rec in self.records:
            for det in rec.get("detections", []):
                ft = det["type"]
                stats[ft] = stats.get(ft, 0) + 1
        return stats

    def get_success_rate(self) -> float:
        """Overall print success rate."""
        if not self.records:
            return 0.0
        successes = sum(1 for r in self.records if r.get("outcome") == "success")
        return successes / len(self.records)

    def suggest_settings(self, material: str = "PLA") -> dict:
        """
        Based on history, suggest settings that had the best outcomes.
        Simple heuristic: find successful prints with fewest detections.
        """
        successful = [r for r in self.records
                      if r.get("outcome") == "success" and r.get("settings")]
        if not successful:
            return {}

        # Sort by fewest issues
        successful.sort(key=lambda r: len(r.get("detections", [])))
        return successful[0].get("settings", {})


# ---------------------------------------------------------------------------
# Main monitor class - ties everything together
# ---------------------------------------------------------------------------

class PrintMonitor:
    """
    Main print monitor. Orchestrates:
    1. Periodic webcam snapshots
    2. Frame analysis for failure detection
    3. Decision making on corrective actions
    4. Action execution via OctoPrint API
    5. Alert delivery (returns alerts for external handling)
    6. History recording for learning
    """

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.client = OctoPrintClient(config.octoprint_url, config.octoprint_api_key)
        self.analyzer = FrameAnalyzer(config)
        self.decision = DecisionEngine(config)
        self.executor = ActionExecutor(self.client, config)
        self.history = PrintHistory(config.history_path)

        self.is_running = False
        self.all_detections: list = []
        self.all_actions: list = []
        self.alerts_queue: list = []  # For external consumption
        self.snapshot_count = 0

        # Ensure dirs exist
        Path(config.snapshot_dir).mkdir(parents=True, exist_ok=True)

    def capture_snapshot(self) -> Optional[Image.Image]:
        """Capture a frame from the webcam."""
        if not HAS_PIL:
            return None

        try:
            req = urllib.request.urlopen(self.config.snapshot_url, timeout=10)
            from io import BytesIO
            img = Image.open(BytesIO(req.read()))
            self.snapshot_count += 1

            # Save snapshot
            path = Path(self.config.snapshot_dir) / f"snap_{self.snapshot_count:05d}.jpg"
            img.save(str(path), "JPEG", quality=85)

            return img
        except Exception as e:
            self._log(f"Snapshot failed: {e}")
            return None

    def analyze_frame(self, frame: Image.Image) -> dict:
        """
        Analyze a single frame. Returns dict with detections and actions.
        This is the main entry point for each monitoring cycle.
        """
        # Get print state
        print_state = self.client.get_printer_state()

        # Analyze frame
        detections = self.analyzer.analyze(frame, print_state)
        self.all_detections.extend(detections)

        # Decide actions
        actions = self.decision.decide(detections, print_state)

        # Execute actions
        results = []
        for action in actions:
            result = self.executor.execute(action, print_state)
            results.append(result)

            # Queue alerts
            if action.action_type in (ActionType.ALERT, ActionType.PAUSE,
                                       ActionType.CANCEL, ActionType.EMERGENCY_STOP):
                self.alerts_queue.append({
                    "action": action.action_type.value,
                    "reason": action.reason,
                    "severity": action.detection.severity.value if action.detection else "unknown",
                    "timestamp": time.time(),
                    "print_state": {
                        "progress": print_state.progress_pct,
                        "layer": print_state.current_layer,
                        "nozzle_temp": print_state.nozzle_temp,
                        "bed_temp": print_state.bed_temp,
                        "elapsed": print_state.elapsed_sec,
                    },
                    "snapshot_num": self.snapshot_count,
                })

        self.all_actions.extend(actions)

        return {
            "frame": self.snapshot_count,
            "print_state": {
                "printing": print_state.is_printing,
                "progress": f"{print_state.progress_pct:.1f}%",
                "nozzle": f"{print_state.nozzle_temp:.0f}/{print_state.nozzle_target:.0f}C",
                "bed": f"{print_state.bed_temp:.0f}/{print_state.bed_target:.0f}C",
            },
            "detections": [
                {
                    "type": d.failure_type.value,
                    "severity": d.severity.value,
                    "confidence": f"{d.confidence:.0%}",
                    "description": d.description,
                }
                for d in detections
            ],
            "actions": [
                {
                    "type": a.action_type.value,
                    "reason": a.reason,
                }
                for a in actions
            ],
            "results": results,
        }

    def get_pending_alerts(self) -> list:
        """Get and clear pending alerts."""
        alerts = self.alerts_queue.copy()
        self.alerts_queue.clear()
        return alerts

    def set_reference_frame(self, frame: Image.Image):
        """Set reference frame (call before print starts)."""
        self.analyzer.set_reference(frame)
        frame.save(str(Path(self.config.snapshot_dir) / "reference.jpg"), "JPEG")
        self._log("Reference frame set")

    def record_outcome(self, outcome: str, settings: dict = None):
        """Record the print session outcome for learning."""
        state = self.client.get_printer_state()
        self.history.record_session(
            filename=state.filename,
            detections=self.all_detections,
            actions=self.all_actions,
            outcome=outcome,
            settings=settings,
        )
        self._log(f"Session recorded: {outcome}, {len(self.all_detections)} detections, "
                  f"{len(self.all_actions)} actions")

    def get_stats(self) -> dict:
        """Get monitor statistics."""
        return {
            "snapshots": self.snapshot_count,
            "total_detections": len(self.all_detections),
            "total_actions": len(self.all_actions),
            "pending_alerts": len(self.alerts_queue),
            "detection_breakdown": {
                ft.value: sum(1 for d in self.all_detections if d.failure_type == ft)
                for ft in FailureType
                if any(d.failure_type == ft for d in self.all_detections)
            },
            "history": {
                "total_prints": len(self.history.records),
                "success_rate": f"{self.history.get_success_rate():.0%}",
                "common_failures": self.history.get_failure_stats(),
            },
        }

    def _log(self, msg: str):
        """Write to monitor log."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        try:
            Path(self.config.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.log_path, "a") as f:
                f.write(line)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Convenience: single-shot analysis (for testing without live printer)
# ---------------------------------------------------------------------------

def analyze_image(image_path: str, reference_path: str = None) -> dict:
    """
    Analyze a single image for print failures.
    Useful for testing or post-mortem analysis.
    """
    config = MonitorConfig()
    monitor = PrintMonitor(config)

    frame = Image.open(image_path).convert("RGB")

    if reference_path:
        ref = Image.open(reference_path).convert("RGB")
        monitor.set_reference_frame(ref)

    # Mock print state for standalone analysis
    mock_state = PrintState(is_printing=True, elapsed_sec=600)

    detections = monitor.analyzer.analyze(frame, mock_state)

    return {
        "image": image_path,
        "detections": [
            {
                "type": d.failure_type.value,
                "severity": d.severity.value,
                "confidence": f"{d.confidence:.0%}",
                "description": d.description,
            }
            for d in detections
        ],
        "total_issues": len(detections),
    }


if __name__ == "__main__":
    print("G1 Print Monitor")
    print("=" * 50)
    print(f"PIL available: {HAS_PIL}")
    print()
    print("Components:")
    print("  - OctoPrintClient: OctoPrint REST API communication")
    print("  - FrameAnalyzer: Webcam frame failure detection")
    print("  - DecisionEngine: Detection -> action mapping")
    print("  - ActionExecutor: Corrective action execution")
    print("  - PrintHistory: Learning database")
    print("  - PrintMonitor: Main orchestrator")
    print()
    print("Detectable failures:")
    for ft in FailureType:
        print(f"  - {ft.value}")
    print()
    print("Available actions:")
    for at in ActionType:
        print(f"  - {at.value}")
    print()
    print("Usage:")
    print("  monitor = PrintMonitor(MonitorConfig(")
    print("      octoprint_url='http://localhost:5000',")
    print("      snapshot_url='http://localhost:8080/?action=snapshot',")
    print("  ))")
    print("  frame = monitor.capture_snapshot()")
    print("  result = monitor.analyze_frame(frame)")
    print("  alerts = monitor.get_pending_alerts()")
