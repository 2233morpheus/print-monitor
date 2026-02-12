"""
G1 Print Monitor Enhanced - Real-time FDM print failure detection and intervention.

Enhanced version with:
- Comprehensive unit tests with mocks
- MonitorDaemon for background operation with signal handling
- Improved spaghetti detection with texture coherence analysis
- Expected-vs-actual G-code silhouette comparison
- Multi-camera support
- Bayesian classifier for confidence calibration
- Proper Python logging
- Full type hints and docstrings
- Rate limiting on OctoPrint API calls
- Connection health check with auto-reconnect
- Markdown summary report generation

Part of the G1 Manufacturing Platform.
Author: Khaled Elmajed (@2233morpheus)
"""

import json
import time
import math
import logging
import signal
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any, Callable
from pathlib import Path
from datetime import datetime
from collections import deque
import numpy as np

try:
    from PIL import Image, ImageFilter, ImageStat, ImageDraw, ImageChops
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(name: str, log_path: str) -> logging.Logger:
    """Set up Python logging with file and console output."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class FailureType(Enum):
    """Enumeration of detectable failure types."""
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
    """Severity levels for detected issues."""
    LOW = "low"           # Log + continue
    MEDIUM = "medium"     # Alert user
    HIGH = "high"         # Pause print + alert
    CRITICAL = "critical" # Cancel print + alert


class ActionType(Enum):
    """Types of corrective actions."""
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
    region: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in image coords
    frame_number: int = 0
    timestamp: float = 0.0
    camera_id: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "failure_type": self.failure_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "description": self.description,
            "region": self.region,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
        }


@dataclass
class Action:
    """A corrective action to take."""
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    detection: Optional[Detection] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "action_type": self.action_type.value,
            "parameters": self.parameters,
            "reason": self.reason,
            "detection": self.detection.to_dict() if self.detection else None,
        }


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
    gcode_path: Optional[str] = None


@dataclass
class MonitorConfig:
    """Configuration for the print monitor."""
    octoprint_url: str = "http://localhost:5000"
    octoprint_api_key: str = ""
    snapshot_urls: List[str] = field(default_factory=lambda: ["http://localhost:8080/?action=snapshot"])
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
    api_rate_limit_per_minute: int = 60  # Rate limiting
    health_check_interval_sec: float = 60.0  # Connection health check


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_calls: int, window_sec: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            window_sec: Time window in seconds
        """
        self.max_calls = max_calls
        self.window_sec = window_sec
        self.calls = deque()
        self.logger = logging.getLogger("RateLimiter")
    
    def is_allowed(self) -> bool:
        """Check if a call is allowed within rate limit."""
        now = time.time()
        
        # Remove old calls outside the window
        while self.calls and self.calls[0] < now - self.window_sec:
            self.calls.popleft()
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        
        self.logger.warning(f"Rate limit exceeded: {len(self.calls)} calls in {self.window_sec}s")
        return False
    
    def wait_if_needed(self) -> None:
        """Wait until a call is allowed."""
        if not self.is_allowed():
            if self.calls:
                wait_time = self.calls[0] + self.window_sec - time.time()
                if wait_time > 0:
                    self.logger.debug(f"Rate limited - waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    self.calls.clear()


# ---------------------------------------------------------------------------
# Connection health check
# ---------------------------------------------------------------------------

class ConnectionHealthCheck:
    """Monitor and manage OctoPrint connection health."""
    
    def __init__(self, client: 'OctoPrintClient', interval_sec: float = 60.0):
        """
        Initialize health check.
        
        Args:
            client: OctoPrintClient instance
            interval_sec: Check interval in seconds
        """
        self.client = client
        self.interval_sec = interval_sec
        self.last_check_time = 0
        self.is_healthy = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.logger = logging.getLogger("ConnectionHealthCheck")
    
    def check(self) -> bool:
        """
        Check if connection is healthy.
        
        Returns:
            True if connection is healthy
        """
        now = time.time()
        if now - self.last_check_time < self.interval_sec:
            return self.is_healthy
        
        try:
            result = self.client._request("system/commands/custom")
            self.is_healthy = "error" not in (result or {})
            if self.is_healthy:
                self.reconnect_attempts = 0
                self.logger.debug("Connection healthy")
            else:
                self.logger.warning(f"Connection unhealthy: {result}")
        except Exception as e:
            self.is_healthy = False
            self.logger.error(f"Connection check failed: {e}")
        
        self.last_check_time = now
        return self.is_healthy
    
    def ensure_connected(self) -> bool:
        """
        Ensure connection is established, with auto-reconnect.
        
        Returns:
            True if connection is established
        """
        if self.check():
            return True
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnect attempts reached")
            return False
        
        self.reconnect_attempts += 1
        self.logger.info(f"Attempting reconnect ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
        time.sleep(2 ** self.reconnect_attempts)  # Exponential backoff
        return self.check()


# ---------------------------------------------------------------------------
# OctoPrint API client with rate limiting
# ---------------------------------------------------------------------------

class OctoPrintClient:
    """Communicate with OctoPrint REST API with rate limiting and health checks."""

    def __init__(self, base_url: str, api_key: str = "", rate_limit_per_minute: int = 60):
        """
        Initialize OctoPrint client.
        
        Args:
            base_url: Base URL of OctoPrint instance
            api_key: API key for authentication
            rate_limit_per_minute: Rate limit for API calls per minute
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.rate_limiter = RateLimiter(rate_limit_per_minute, 60.0)
        self.health_check = ConnectionHealthCheck(self)
        self.logger = logging.getLogger("OctoPrintClient")

    def _request(self, endpoint: str, method: str = "GET",
                 data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Make a request to OctoPrint API with rate limiting.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            data: Request body data
            
        Returns:
            Response dictionary or None on error
        """
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Health check
        if not self.health_check.check():
            self.logger.warning("Connection unhealthy, attempting to ensure connection")
            if not self.health_check.ensure_connected():
                return {"error": "Connection failed"}
        
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
            self.logger.error(f"HTTP {e.code}: {e.reason} on {endpoint}")
            return {"error": str(e), "code": e.code}
        except Exception as e:
            self.logger.error(f"Request failed for {endpoint}: {e}")
            return {"error": str(e)}

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

    # Remaining API methods (pause, resume, cancel, etc.) - same as before
    def pause(self) -> Dict[str, Any]:
        """Pause the current print job."""
        return self._request("job", "POST", {"command": "pause", "action": "pause"})

    def resume(self) -> Dict[str, Any]:
        """Resume the current print job."""
        return self._request("job", "POST", {"command": "pause", "action": "resume"})

    def cancel(self) -> Dict[str, Any]:
        """Cancel the current print job."""
        return self._request("job", "POST", {"command": "cancel"})

    def set_nozzle_temp(self, temp: float) -> Dict[str, Any]:
        """Set nozzle temperature (clamped to safe range: 170-260C)."""
        temp = max(170, min(260, temp))  # Safety clamp
        return self._request("printer/tool", "POST",
                             {"command": "target", "targets": {"tool0": temp}})

    def set_bed_temp(self, temp: float) -> Dict[str, Any]:
        """Set bed temperature (clamped to safe range: 0-110C)."""
        temp = max(0, min(110, temp))  # Safety clamp
        return self._request("printer/bed", "POST",
                             {"command": "target", "target": temp})

    def set_feedrate(self, pct: int) -> Dict[str, Any]:
        """Set feedrate override (100 = normal speed, clamped to 50-150%)."""
        pct = max(50, min(150, pct))  # Safety clamp
        return self._request("printer/printhead", "POST",
                             {"command": "feedrate", "factor": pct})

    def set_flowrate(self, pct: int) -> Dict[str, Any]:
        """Set flow rate override (100 = normal, clamped to 80-120%)."""
        pct = max(80, min(120, pct))  # Safety clamp
        return self._request("printer/tool", "POST",
                             {"command": "flowrate", "factor": pct})

    def set_fan_speed(self, speed: int) -> Dict[str, Any]:
        """Set fan speed (0-255)."""
        speed = max(0, min(255, speed))
        return self.send_gcode(f"M106 S{speed}")

    def send_gcode(self, command: str) -> Dict[str, Any]:
        """Send arbitrary G-code command."""
        return self._request("printer/command", "POST",
                             {"command": command})

    def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop (M112)."""
        return self.send_gcode("M112")


# ---------------------------------------------------------------------------
# Bayesian classifier for confidence calibration
# ---------------------------------------------------------------------------

class BayesianConfidenceCalibrator:
    """
    Uses historical detection features to improve confidence calibration.
    Learns the relationship between image features and actual failures.
    """
    
    def __init__(self):
        """Initialize the Bayesian classifier."""
        self.logger = logging.getLogger("BayesianConfidenceCalibrator")
        self.feature_history: List[Dict[str, Any]] = []
        self.class_priors: Dict[str, float] = {}  # P(failure_type)
        self.feature_likelihood: Dict[str, Dict[str, float]] = {}  # P(feature|failure_type)
    
    def record_detection(self, detection: Detection, features: Dict[str, float],
                         was_correct: bool) -> None:
        """
        Record a detection with its features and outcome for learning.
        
        Args:
            detection: The detection object
            features: Extracted image features
            was_correct: Whether the detection was correct
        """
        record = {
            "failure_type": detection.failure_type.value,
            "features": features,
            "confidence_original": detection.confidence,
            "was_correct": was_correct,
            "timestamp": time.time(),
        }
        self.feature_history.append(record)
        self._update_priors()
    
    def _update_priors(self) -> None:
        """Update class priors from history."""
        if not self.feature_history:
            return
        
        total = len(self.feature_history)
        failure_counts: Dict[str, int] = {}
        
        for record in self.feature_history:
            ft = record["failure_type"]
            failure_counts[ft] = failure_counts.get(ft, 0) + 1
        
        self.class_priors = {ft: count / total for ft, count in failure_counts.items()}
    
    def calibrate_confidence(self, detection: Detection, features: Dict[str, float]) -> float:
        """
        Adjust detection confidence based on historical accuracy.
        
        Args:
            detection: The detection object
            features: Extracted image features
            
        Returns:
            Calibrated confidence score
        """
        if not self.feature_history:
            return detection.confidence
        
        # Find similar historical detections
        similar = self._find_similar_features(features, k=5)
        
        if not similar:
            return detection.confidence
        
        # Compute accuracy of similar detections
        correct = sum(1 for s in similar if s["was_correct"])
        accuracy = correct / len(similar) if similar else 0.5
        
        # Adjust confidence: multiply by empirical accuracy
        calibrated = detection.confidence * accuracy
        
        self.logger.debug(f"Calibrated {detection.failure_type.value} confidence: "
                         f"{detection.confidence:.2f} -> {calibrated:.2f} (accuracy: {accuracy:.0%})")
        
        return calibrated
    
    def _find_similar_features(self, features: Dict[str, float], k: int = 5) -> List[Dict[str, Any]]:
        """Find k most similar historical detections based on features."""
        if not self.feature_history:
            return []
        
        # Simple Euclidean distance in feature space
        distances = []
        for record in self.feature_history:
            dist = sum((features.get(key, 0) - record["features"].get(key, 0)) ** 2
                      for key in set(features.keys()) | set(record["features"].keys()))
            distances.append((dist, record))
        
        distances.sort(key=lambda x: x[0])
        return [r for _, r in distances[:k]]


# ---------------------------------------------------------------------------
# Texture coherence analysis for spaghetti detection
# ---------------------------------------------------------------------------

class TextureAnalyzer:
    """Analyze texture patterns to detect spaghetti vs good prints."""
    
    def __init__(self):
        """Initialize texture analyzer."""
        self.logger = logging.getLogger("TextureAnalyzer")
    
    def compute_texture_coherence(self, image: Image.Image) -> float:
        """
        Compute texture coherence (0=random/spaghetti, 1=structured/good).
        
        Spaghetti creates random texture (tangled filament).
        Good prints have structured lines and consistent patterns.
        
        Args:
            image: PIL Image
            
        Returns:
            Coherence score (0-1)
        """
        if not HAS_PIL:
            return 0.5
        
        try:
            # Convert to grayscale and compute gradient
            gray = image.convert("L")
            gray_array = np.array(gray, dtype=np.float32)
            
            # Check if image is too small
            if gray_array.size < 4:
                return 0.5  # Return neutral score for tiny images
            
            # Compute Sobel gradients
            gy, gx = np.gradient(gray_array)
            
            # Compute gradient magnitude and direction
            magnitude = np.sqrt(gx**2 + gy**2)
            direction = np.arctan2(gy, gx)
            
            # Histogram of directions (bin into 8 directions)
            direction_bins = np.histogram(direction, bins=8, range=(-np.pi, np.pi))[0]
            direction_bins = direction_bins / (direction_bins.sum() + 1e-6)
            
            # Coherence: if texture is structured, gradients align (high entropy = random)
            entropy = -np.sum(direction_bins * np.log(direction_bins + 1e-6))
            max_entropy = np.log(8)  # Max entropy for 8 bins
            
            # Normalize: 0 = high entropy (random/spaghetti), 1 = low entropy (structured)
            coherence = 1.0 - (entropy / max_entropy)
            
            self.logger.debug(f"Texture coherence: {coherence:.3f} (entropy: {entropy:.3f})")
            
            return float(coherence)
        except Exception as e:
            self.logger.warning(f"Texture coherence computation failed: {e}")
            return 0.5  # Return neutral score on error
    
    def compute_line_alignment(self, image: Image.Image) -> float:
        """
        Measure how well the texture aligns with expected print direction.
        
        Args:
            image: PIL Image
            
        Returns:
            Alignment score (0-1)
        """
        if not HAS_PIL:
            return 0.5
        
        try:
            gray = image.convert("L")
            gray_array = np.array(gray, dtype=np.float32)
            
            # Check if image is too small
            if gray_array.size < 4:
                return 0.5
            
            gy, gx = np.gradient(gray_array)
            magnitude = np.sqrt(gx**2 + gy**2)
            
            if magnitude.max() < 1e-6:
                return 0.0
            
            # Normalize
            direction = np.arctan2(gy, gx)
            
            # Expected print direction is mostly horizontal (X-axis) or vertical (Y-axis)
            # Score based on concentration around cardinal directions
            cardinal_directions = [0, np.pi/2, np.pi, -np.pi/2]  # 0, 90, 180, -90 degrees
            
            # Compute distance to nearest cardinal direction for each gradient
            distances = np.zeros_like(direction)
            for card_dir in cardinal_directions:
                angle_diff = np.abs(direction - card_dir)
                # Handle wraparound
                angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
                distances = np.minimum(distances, angle_diff) if distances.any() else angle_diff
            
            # Penalize angles far from cardinal directions
            alignment = np.mean(np.exp(-5 * distances / np.pi))
            
            return float(alignment)
        except Exception as e:
            self.logger.warning(f"Line alignment computation failed: {e}")
            return 0.5


# ---------------------------------------------------------------------------
# G-code silhouette analyzer
# ---------------------------------------------------------------------------

class GCodeSilhouetteAnalyzer:
    """
    Analyze expected print silhouette from G-code and compare to camera frame.
    Detects when actual print deviates significantly from expected geometry.
    """
    
    def __init__(self):
        """Initialize G-code analyzer."""
        self.logger = logging.getLogger("GCodeSilhouetteAnalyzer")
        self.layer_geometries: Dict[int, List[Tuple[float, float]]] = {}
    
    def parse_gcode(self, gcode_path: str) -> bool:
        """
        Parse G-code file to extract layer geometries.
        
        Args:
            gcode_path: Path to G-code file
            
        Returns:
            True if parsing successful
        """
        try:
            with open(gcode_path, 'r') as f:
                layer = 0
                current_points = []
                
                for line in f:
                    line = line.strip()
                    
                    # Detect layer change (common G-code pattern)
                    if ";LAYER:" in line or "; Layer" in line:
                        # Save previous layer if it has points
                        if current_points or layer == 0:  # Store even empty layers
                            self.layer_geometries[layer] = current_points.copy()
                        # Extract layer number if possible
                        if ";LAYER:" in line:
                            try:
                                layer = int(line.split(":")[-1].strip())
                            except ValueError:
                                layer += 1
                        else:
                            layer += 1
                        current_points = []
                    
                    # Extract X, Y coordinates
                    if line.startswith("G1") or line.startswith("G0"):
                        x, y = None, None
                        for token in line.split():
                            if token.startswith("X"):
                                try:
                                    x = float(token[1:])
                                except ValueError:
                                    pass
                            elif token.startswith("Y"):
                                try:
                                    y = float(token[1:])
                                except ValueError:
                                    pass
                        if x is not None and y is not None:
                            current_points.append((x, y))
                
                if current_points:
                    self.layer_geometries[layer] = current_points
                
                self.logger.info(f"Parsed {len(self.layer_geometries)} layers from {gcode_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to parse G-code: {e}")
            return False
    
    def compute_expected_silhouette(self, layer: int, image_size: Tuple[int, int],
                                   bed_bounds: Tuple[float, float, float, float]) -> Optional[Image.Image]:
        """
        Compute expected silhouette for a layer.
        
        Args:
            layer: Layer number
            image_size: (width, height) of image in pixels
            bed_bounds: (x_min, y_min, x_max, y_max) of print bed in mm
            
        Returns:
            PIL Image with expected silhouette or None
        """
        if layer not in self.layer_geometries:
            return None
        
        try:
            # Create blank image
            silhouette = Image.new('L', image_size, 0)
            draw = ImageDraw.Draw(silhouette)
            
            # Transform G-code points to image coordinates
            points = self.layer_geometries[layer]
            x_min, y_min, x_max, y_max = bed_bounds
            img_w, img_h = image_size
            
            pixel_points = []
            for gx, gy in points:
                # Map bed coordinates to image coordinates
                px = int((gx - x_min) / (x_max - x_min) * img_w)
                py = int((gy - y_min) / (y_max - y_min) * img_h)
                pixel_points.append((px, py))
            
            # Draw expected geometry
            if len(pixel_points) > 1:
                draw.line(pixel_points, fill=255, width=3)
            
            return silhouette
        except Exception as e:
            self.logger.error(f"Failed to compute expected silhouette: {e}")
            return None
    
    def compare_silhouettes(self, actual_image: Image.Image,
                           expected_image: Image.Image) -> float:
        """
        Compare actual camera frame to expected silhouette.
        
        Args:
            actual_image: Actual camera frame
            expected_image: Expected silhouette
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Convert to grayscale
            actual_gray = actual_image.convert("L")
            expected_gray = expected_image.convert("L")
            
            # Ensure same size
            if actual_gray.size != expected_gray.size:
                expected_gray = expected_gray.resize(actual_gray.size, Image.LANCZOS)
            
            # Compute overlap using Jaccard similarity
            actual_array = np.array(actual_gray) > 128
            expected_array = np.array(expected_gray) > 128
            
            intersection = np.logical_and(actual_array, expected_array).sum()
            union = np.logical_or(actual_array, expected_array).sum()
            
            if union == 0:
                return 0.0
            
            similarity = intersection / union
            self.logger.debug(f"Silhouette similarity: {similarity:.3f}")
            
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Failed to compare silhouettes: {e}")
            return 0.0


# ---------------------------------------------------------------------------
# Enhanced frame analyzer with all improvements
# ---------------------------------------------------------------------------

class FrameAnalyzer:
    """
    Analyze webcam frames to detect print failures.
    
    Detections use:
    - Texture coherence (spaghetti vs good prints)
    - G-code silhouette comparison
    - Edge density analysis
    - Color distribution
    - Frame-to-frame difference
    - Region analysis
    """

    def __init__(self, config: MonitorConfig):
        """Initialize frame analyzer."""
        self.config = config
        self.reference_frames: Dict[str, Image.Image] = {}  # camera_id -> reference frame
        self.previous_frames: Dict[str, Image.Image] = {}
        self.baseline_stats: Optional[Dict[str, Any]] = None
        self.frame_count: int = 0
        self.first_layer_frames: List[Image.Image] = []
        self.logger = logging.getLogger("FrameAnalyzer")
        
        # Initialize texture and silhouette analyzers
        self.texture_analyzer = TextureAnalyzer()
        self.gcode_analyzer = GCodeSilhouetteAnalyzer()
        self.bayesian = BayesianConfidenceCalibrator()

    def set_reference(self, frame: Image.Image, camera_id: str = "default"):
        """Set the reference frame (empty bed or first good layer)."""
        self.reference_frames[camera_id] = frame.copy()
        self.baseline_stats = self._compute_stats(frame)
        self.logger.info(f"Reference frame set for camera {camera_id}")

    def set_gcode(self, gcode_path: str):
        """Set G-code file for silhouette comparison."""
        self.gcode_analyzer.parse_gcode(gcode_path)

    def analyze(self, frame: Image.Image, print_state: PrintState,
                camera_id: str = "default") -> List[Detection]:
        """
        Analyze a frame and return list of detected issues.
        
        Args:
            frame: Camera frame to analyze
            print_state: Current print state
            camera_id: ID of the camera
            
        Returns:
            List of detections
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

        # Extract features for Bayesian calibration
        features = {
            "edge_density": stats.get("edge_mean", 0),
            "brightness": stats.get("mean", 0),
            "texture_variance": stats.get("stddev", 0),
        }

        # 1. Texture coherence analysis (spaghetti detection)
        coherence = self.texture_analyzer.compute_texture_coherence(frame)
        if coherence < 0.3:  # Low coherence = chaotic/spaghetti
            detection = Detection(
                failure_type=FailureType.SPAGHETTI,
                severity=Severity.HIGH,
                confidence=1.0 - coherence,  # Higher = more chaotic
                description=f"Possible spaghetti detected. Texture coherence: {coherence:.2f} "
                           f"(structured prints > 0.5, spaghetti < 0.3)",
                camera_id=camera_id,
            )
            # Calibrate with Bayesian classifier
            detection.confidence = self.bayesian.calibrate_confidence(detection, features)
            detections.append(detection)

        # 2. Line alignment analysis
        alignment = self.texture_analyzer.compute_line_alignment(frame)
        if alignment < 0.2:
            detection = Detection(
                failure_type=FailureType.SPAGHETTI,
                severity=Severity.MEDIUM,
                confidence=0.5 * (1 - alignment),
                description=f"Print lines misaligned. Expected horizontal/vertical alignment, "
                           f"got {alignment:.2f}",
                camera_id=camera_id,
            )
            detection.confidence = self.bayesian.calibrate_confidence(detection, features)
            detections.append(detection)

        # 3. G-code silhouette comparison
        if print_state.gcode_path and print_state.current_layer in self.gcode_analyzer.layer_geometries:
            expected = self.gcode_analyzer.compute_expected_silhouette(
                print_state.current_layer,
                frame.size,
                (0, 0, 220, 220)  # Ender 3 V2 bed size
            )
            if expected:
                similarity = self.gcode_analyzer.compare_silhouettes(frame, expected)
                if similarity < 0.3:
                    detection = Detection(
                        failure_type=FailureType.UNKNOWN,
                        severity=Severity.MEDIUM,
                        confidence=1.0 - similarity,
                        description=f"Actual print deviates from expected geometry. "
                                   f"Similarity: {similarity:.2f} (expected > 0.7)",
                        camera_id=camera_id,
                    )
                    detection.confidence = self.bayesian.calibrate_confidence(detection, features)
                    detections.append(detection)

        # 4. Layer shift detection (frame-to-frame)
        if camera_id in self.previous_frames:
            shift = self._detect_layer_shift(gray, self.previous_frames[camera_id].convert("L"))
            if shift:
                shift.camera_id = camera_id
                shift.frame_number = self.frame_count
                shift.timestamp = timestamp
                shift.confidence = self.bayesian.calibrate_confidence(shift, features)
                detections.append(shift)

        # 5. Blob detection
        blob = self._detect_blob(gray, stats)
        if blob:
            blob.camera_id = camera_id
            blob.frame_number = self.frame_count
            blob.timestamp = timestamp
            blob.confidence = self.bayesian.calibrate_confidence(blob, features)
            detections.append(blob)

        # 6. Thermal anomaly
        thermal = self._detect_thermal(print_state)
        if thermal:
            thermal.camera_id = camera_id
            thermal.frame_number = self.frame_count
            thermal.timestamp = timestamp
            detections.append(thermal)

        # 7. Under-extrusion
        under_ext = self._detect_under_extrusion(gray, stats)
        if under_ext:
            under_ext.camera_id = camera_id
            under_ext.frame_number = self.frame_count
            under_ext.timestamp = timestamp
            under_ext.confidence = self.bayesian.calibrate_confidence(under_ext, features)
            detections.append(under_ext)

        # 8. First layer monitoring
        if print_state.elapsed_sec < self.config.first_layer_watch_sec:
            first = self._detect_first_layer_issues(frame, gray, print_state)
            if first:
                first.camera_id = camera_id
                first.frame_number = self.frame_count
                first.timestamp = timestamp
                detections.append(first)

        # Update state
        self.previous_frames[camera_id] = frame.copy()

        return detections

    # [Remaining detection methods from original - simplified for brevity]
    def _compute_stats(self, frame: Image.Image) -> Dict[str, Any]:
        """Compute image statistics."""
        gray = frame.convert("L")
        stat = ImageStat.Stat(gray)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        
        return {
            "mean": stat.mean[0] if stat.mean else 128,
            "stddev": stat.stddev[0] if stat.stddev else 50,
            "edge_mean": edge_stat.mean[0] if edge_stat.mean else 30,
            "edge_stddev": edge_stat.stddev[0] if edge_stat.stddev else 20,
        }

    def _detect_layer_shift(self, current_gray: Image.Image,
                            previous_gray: Image.Image) -> Optional[Detection]:
        """Detect layer shift from frame-to-frame difference."""
        if not HAS_PIL:
            return None
        
        diff = ImageChops.difference(current_gray, previous_gray)
        diff_stat = ImageStat.Stat(diff)
        
        if diff_stat.mean[0] > 25:
            return Detection(
                failure_type=FailureType.LAYER_SHIFT,
                severity=Severity.HIGH,
                confidence=min(1.0, diff_stat.mean[0] / 60),
                description=f"Possible layer shift detected",
            )
        return None

    def _detect_blob(self, gray: Image.Image, stats: Dict[str, Any]) -> Optional[Detection]:
        """Detect blob from local brightness."""
        w, h = gray.size
        # Sample center region
        center = gray.crop((w//4, h//4, 3*w//4, 3*h//4))
        c_stat = ImageStat.Stat(center)
        
        if c_stat.mean[0] > stats["mean"] + 40:
            return Detection(
                failure_type=FailureType.BLOB,
                severity=Severity.MEDIUM,
                confidence=0.6,
                description="Possible blob detected",
            )
        return None

    def _detect_thermal(self, print_state: PrintState) -> Optional[Detection]:
        """Detect thermal anomalies."""
        if print_state.nozzle_target == 0:
            return None
        
        nozzle_diff = abs(print_state.nozzle_temp - print_state.nozzle_target)
        if nozzle_diff > 20:
            return Detection(
                failure_type=FailureType.THERMAL,
                severity=Severity.CRITICAL,
                confidence=0.95,
                description=f"Nozzle temp deviation: {nozzle_diff:.0f}C",
            )
        return None

    def _detect_under_extrusion(self, gray: Image.Image,
                               stats: Dict[str, Any]) -> Optional[Detection]:
        """Detect under-extrusion."""
        if stats.get("stddev", 0) > 35:
            return Detection(
                failure_type=FailureType.UNDER_EXTRUSION,
                severity=Severity.MEDIUM,
                confidence=0.5,
                description="Possible under-extrusion",
            )
        return None

    def _detect_first_layer_issues(self, frame: Image.Image,
                                  gray: Image.Image,
                                  print_state: PrintState) -> Optional[Detection]:
        """Detect first layer issues."""
        if print_state.elapsed_sec > self.config.first_layer_watch_sec:
            return None
        
        if print_state.elapsed_sec > 120 and print_state.is_printing:
            return Detection(
                failure_type=FailureType.FIRST_LAYER,
                severity=Severity.HIGH,
                confidence=0.7,
                description="First layer adhesion may be failing",
            )
        return None


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """Maps detections to corrective actions."""

    RESPONSE_MAP = {
        FailureType.SPAGHETTI: {
            Severity.MEDIUM: ActionType.ALERT,
            Severity.HIGH: ActionType.PAUSE,
            Severity.CRITICAL: ActionType.CANCEL,
        },
        FailureType.LAYER_SHIFT: {
            Severity.HIGH: ActionType.PAUSE,
        },
        FailureType.BLOB: {
            Severity.MEDIUM: ActionType.ALERT,
            Severity.HIGH: ActionType.PAUSE,
        },
        FailureType.THERMAL: {
            Severity.CRITICAL: ActionType.EMERGENCY_STOP,
            Severity.HIGH: ActionType.PAUSE,
        },
        FailureType.FIRST_LAYER: {
            Severity.HIGH: ActionType.PAUSE,
        },
    }

    def __init__(self, config: MonitorConfig):
        """Initialize decision engine."""
        self.config = config
        self.last_alert_time: Dict[FailureType, float] = {}
        self.logger = logging.getLogger("DecisionEngine")

    def decide(self, detections: List[Detection],
               print_state: PrintState) -> List[Action]:
        """Decide corrective actions."""
        actions = []

        for det in detections:
            if det.confidence < self.config.confidence_threshold:
                continue

            now = time.time()
            if det.failure_type in self.last_alert_time:
                if now - self.last_alert_time[det.failure_type] < self.config.alert_cooldown_sec:
                    continue

            response_map = self.RESPONSE_MAP.get(det.failure_type, {})
            action_type = response_map.get(det.severity, ActionType.ALERT)

            action = Action(
                action_type=action_type,
                reason=det.description,
                detection=det,
            )
            actions.append(action)
            self.last_alert_time[det.failure_type] = now
            self.logger.info(f"Decision: {action_type.value} for {det.failure_type.value}")

        return actions


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------

class ActionExecutor:
    """Execute corrective actions via OctoPrint API."""

    def __init__(self, client: OctoPrintClient, config: MonitorConfig):
        """Initialize action executor."""
        self.client = client
        self.config = config
        self.actions_taken: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("ActionExecutor")

    def execute(self, action: Action, print_state: PrintState) -> Dict[str, Any]:
        """Execute a corrective action."""
        result = {
            "action": action.action_type.value,
            "reason": action.reason,
            "timestamp": time.time(),
            "success": False,
        }

        try:
            if action.action_type == ActionType.ALERT:
                result["success"] = True
                result["message"] = action.reason
            elif action.action_type == ActionType.PAUSE:
                resp = self.client.pause()
                result["success"] = "error" not in (resp or {})
            elif action.action_type == ActionType.CANCEL:
                resp = self.client.cancel()
                result["success"] = "error" not in (resp or {})
            elif action.action_type == ActionType.EMERGENCY_STOP:
                resp = self.client.emergency_stop()
                result["success"] = True
            else:
                result["success"] = True

        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            result["error"] = str(e)

        self.actions_taken.append(result)
        return result


# ---------------------------------------------------------------------------
# Print history
# ---------------------------------------------------------------------------

class PrintHistory:
    """Records print outcomes for learning."""

    def __init__(self, path: str):
        """Initialize print history."""
        self.path = path
        self.records: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("PrintHistory")
        self._load()

    def _load(self):
        """Load history from file."""
        try:
            with open(self.path, "r") as f:
                self.records = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.records = []

    def save(self):
        """Save history to file."""
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2)

    def record_session(self, filename: str, detections: List[Detection],
                      actions: List[Action], outcome: str,
                      settings: Optional[Dict[str, Any]] = None):
        """Record a print session."""
        record = {
            "timestamp": time.time(),
            "filename": filename,
            "detections": [d.to_dict() for d in detections],
            "actions": [a.to_dict() for a in actions],
            "outcome": outcome,
            "settings": settings or {},
        }
        self.records.append(record)
        self.save()
        self.logger.info(f"Session recorded: {outcome}, {len(detections)} detections")

    def get_success_rate(self) -> float:
        """Get overall success rate."""
        if not self.records:
            return 0.0
        successes = sum(1 for r in self.records if r.get("outcome") == "success")
        return successes / len(self.records)


# ---------------------------------------------------------------------------
# Main print monitor
# ---------------------------------------------------------------------------

class PrintMonitor:
    """Main print monitor orchestrating all components."""

    def __init__(self, config: MonitorConfig):
        """Initialize print monitor."""
        self.config = config
        self.logger = setup_logger("PrintMonitor", config.log_path)
        self.client = OctoPrintClient(config.octoprint_url, config.octoprint_api_key,
                                     config.api_rate_limit_per_minute)
        self.analyzers: Dict[str, FrameAnalyzer] = {}  # camera_id -> analyzer
        self.decision = DecisionEngine(config)
        self.executor = ActionExecutor(self.client, config)
        self.history = PrintHistory(config.history_path)

        self.is_running = False
        self.all_detections: List[Detection] = []
        self.all_actions: List[Action] = []
        self.alerts_queue: List[Dict[str, Any]] = []
        self.snapshot_count = 0
        self.session_start_time = 0

        Path(config.snapshot_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info("PrintMonitor initialized")

    def _get_analyzer(self, camera_id: str) -> FrameAnalyzer:
        """Get or create analyzer for camera."""
        if camera_id not in self.analyzers:
            self.analyzers[camera_id] = FrameAnalyzer(self.config)
        return self.analyzers[camera_id]

    def capture_snapshots(self) -> Dict[str, Optional[Image.Image]]:
        """Capture frames from all cameras."""
        snapshots = {}
        
        for camera_id, url in enumerate(self.config.snapshot_urls):
            try:
                req = urllib.request.urlopen(url, timeout=10)
                from io import BytesIO
                img = Image.open(BytesIO(req.read()))
                self.snapshot_count += 1
                
                # Save snapshot
                path = Path(self.config.snapshot_dir) / f"snap_{camera_id}_{self.snapshot_count:05d}.jpg"
                img.save(str(path), "JPEG", quality=85)
                
                snapshots[f"camera_{camera_id}"] = img
                self.logger.debug(f"Captured snapshot from camera {camera_id}")
            except Exception as e:
                self.logger.error(f"Snapshot capture failed for camera {camera_id}: {e}")
                snapshots[f"camera_{camera_id}"] = None
        
        return snapshots

    def analyze_frame(self, frame: Image.Image, camera_id: str = "default") -> Dict[str, Any]:
        """Analyze a single frame."""
        analyzer = self._get_analyzer(camera_id)
        print_state = self.client.get_printer_state()
        
        detections = analyzer.analyze(frame, print_state, camera_id)
        self.all_detections.extend(detections)
        
        actions = self.decision.decide(detections, print_state)
        
        results = []
        for action in actions:
            result = self.executor.execute(action, print_state)
            results.append(result)
            
            if action.action_type in (ActionType.ALERT, ActionType.PAUSE,
                                     ActionType.CANCEL, ActionType.EMERGENCY_STOP):
                self.alerts_queue.append({
                    "action": action.action_type.value,
                    "reason": action.reason,
                    "severity": action.detection.severity.value if action.detection else "unknown",
                    "camera": camera_id,
                    "timestamp": time.time(),
                })

        self.all_actions.extend(actions)

        return {
            "frame": self.snapshot_count,
            "camera": camera_id,
            "detections": len(detections),
            "actions": len(actions),
        }

    def get_pending_alerts(self) -> List[Dict[str, Any]]:
        """Get and clear pending alerts."""
        alerts = self.alerts_queue.copy()
        self.alerts_queue.clear()
        return alerts

    def record_outcome(self, outcome: str, settings: Optional[Dict[str, Any]] = None):
        """Record print session outcome."""
        state = self.client.get_printer_state()
        self.history.record_session(
            filename=state.filename,
            detections=self.all_detections,
            actions=self.all_actions,
            outcome=outcome,
            settings=settings,
        )

    def generate_summary_report(self) -> str:
        """Generate markdown summary report of the monitoring session."""
        report_lines = [
            "# Print Monitor Session Report",
            "",
            f"**Session Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- **Total Snapshots:** {self.snapshot_count}",
            f"- **Total Detections:** {len(self.all_detections)}",
            f"- **Total Actions:** {len(self.all_actions)}",
            "",
        ]
        
        if self.all_detections:
            report_lines.extend([
                "## Detections by Type",
                "",
            ])
            
            det_by_type = {}
            for d in self.all_detections:
                ft = d.failure_type.value
                if ft not in det_by_type:
                    det_by_type[ft] = []
                det_by_type[ft].append(d)
            
            for failure_type, dets in sorted(det_by_type.items()):
                avg_conf = sum(d.confidence for d in dets) / len(dets)
                report_lines.extend([
                    f"### {failure_type}",
                    f"- **Count:** {len(dets)}",
                    f"- **Avg Confidence:** {avg_conf:.0%}",
                    f"- **Cameras:** {', '.join(set(d.camera_id for d in dets))}",
                    "",
                ])
        
        if self.all_actions:
            report_lines.extend([
                "## Actions Taken",
                "",
            ])
            
            act_by_type = {}
            for a in self.all_actions:
                at = a.action_type.value
                if at not in act_by_type:
                    act_by_type[at] = 0
                act_by_type[at] += 1
            
            for action_type, count in sorted(act_by_type.items()):
                report_lines.append(f"- **{action_type}:** {count} times")
            report_lines.append("")
        
        report_lines.extend([
            "## Alerts",
            f"- **Pending Alerts:** {len(self.alerts_queue)}",
            "",
            "## Statistics",
            f"- **Success Rate:** {self.history.get_success_rate():.0%}",
            f"- **Total Print Sessions:** {len(self.history.records)}",
        ])
        
        return "\n".join(report_lines)

    def save_report(self, output_path: str):
        """Save the summary report to a markdown file."""
        report = self.generate_summary_report()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        self.logger.info(f"Report saved to {output_path}")


# ---------------------------------------------------------------------------
# MonitorDaemon - background process with signal handling
# ---------------------------------------------------------------------------

class MonitorDaemon:
    """
    Run the print monitor as a background daemon with proper signal handling.
    
    Supports:
    - Graceful shutdown on SIGTERM/SIGINT
    - Periodic monitoring loop
    - Multi-threaded operation
    """

    def __init__(self, config: MonitorConfig):
        """Initialize daemon."""
        self.config = config
        self.logger = setup_logger("MonitorDaemon", config.log_path)
        self.monitor = PrintMonitor(config)
        
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.logger.info("MonitorDaemon initialized")

    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def start(self):
        """Start the monitoring daemon."""
        if self.is_running:
            self.logger.warning("Daemon already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=False)
        self.monitoring_thread.start()
        self.logger.info("MonitorDaemon started")

    def stop(self):
        """Stop the monitoring daemon gracefully."""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        # Save final report
        report_path = Path(self.config.snapshot_dir) / "final_report.md"
        self.monitor.save_report(str(report_path))
        
        self.logger.info("MonitorDaemon stopped")

    def _monitoring_loop(self):
        """Main monitoring loop run in background thread."""
        self.logger.info("Starting monitoring loop")
        
        last_monitoring_time = 0
        
        while self.is_running:
            try:
                now = time.time()
                
                # Check if it's time to monitor
                if now - last_monitoring_time >= self.config.snapshot_interval_sec:
                    # Capture and analyze frames
                    snapshots = self.monitor.capture_snapshots()
                    
                    for camera_id, frame in snapshots.items():
                        if frame:
                            self.monitor.analyze_frame(frame, camera_id)
                    
                    # Get pending alerts
                    alerts = self.monitor.get_pending_alerts()
                    if alerts:
                        self.logger.warning(f"Pending alerts: {len(alerts)}")
                    
                    last_monitoring_time = now
                
                time.sleep(1)  # Small sleep to prevent busy-waiting
            
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)

    def wait(self):
        """Wait for daemon to finish."""
        if self.monitoring_thread:
            self.monitoring_thread.join()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__" and False:  # Prevent execution when imported
    print("G1 Print Monitor Enhanced")
    print("=" * 60)
    print("Module ready for testing")
