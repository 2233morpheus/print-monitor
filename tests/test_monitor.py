"""
Unit tests for G1 Print Monitor Enhanced

All tests use mocks to avoid requiring a running OctoPrint instance or real cameras.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open
import json
import time
from pathlib import Path
import tempfile
import shutil
from io import BytesIO

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock PIL if not available
try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = MagicMock()

import numpy as np
from lib.print_monitor_enhanced import (
    FailureType, Severity, ActionType, Detection, Action, PrintState,
    MonitorConfig, RateLimiter, ConnectionHealthCheck, OctoPrintClient,
    BayesianConfidenceCalibrator, TextureAnalyzer, GCodeSilhouetteAnalyzer,
    FrameAnalyzer, DecisionEngine, ActionExecutor, PrintHistory, PrintMonitor,
    MonitorDaemon
)


class TestRateLimiter(unittest.TestCase):
    """Test rate limiter functionality."""

    def test_rate_limiter_allows_within_limit(self):
        """Test that calls within limit are allowed."""
        limiter = RateLimiter(max_calls=5, window_sec=1.0)
        
        for i in range(5):
            self.assertTrue(limiter.is_allowed())
        
        # 6th call should be blocked
        self.assertFalse(limiter.is_allowed())

    def test_rate_limiter_resets_after_window(self):
        """Test that limiter resets after window expires."""
        limiter = RateLimiter(max_calls=2, window_sec=0.1)
        
        self.assertTrue(limiter.is_allowed())
        self.assertTrue(limiter.is_allowed())
        self.assertFalse(limiter.is_allowed())
        
        time.sleep(0.15)
        self.assertTrue(limiter.is_allowed())


class TestConnectionHealthCheck(unittest.TestCase):
    """Test connection health checking."""

    def test_health_check_success(self):
        """Test successful health check."""
        mock_client = MagicMock()
        mock_client._request.return_value = {"status": "ok"}
        
        health_check = ConnectionHealthCheck(mock_client, interval_sec=0.01)
        
        self.assertTrue(health_check.check())
        self.assertTrue(health_check.is_healthy)

    def test_health_check_failure(self):
        """Test failed health check."""
        mock_client = MagicMock()
        mock_client._request.return_value = {"error": "connection failed"}
        
        health_check = ConnectionHealthCheck(mock_client, interval_sec=0.01)
        
        self.assertFalse(health_check.check())
        self.assertFalse(health_check.is_healthy)

    def test_health_check_respects_interval(self):
        """Test that health check respects interval."""
        mock_client = MagicMock()
        mock_client._request.return_value = {"status": "ok"}
        
        health_check = ConnectionHealthCheck(mock_client, interval_sec=10.0)
        
        self.assertTrue(health_check.check())
        mock_client._request.reset_mock()
        
        # Second check within interval shouldn't call API
        health_check.check()
        mock_client._request.assert_not_called()


class TestOctoPrintClient(unittest.TestCase):
    """Test OctoPrint API client."""

    @patch('urllib.request.urlopen')
    def test_get_printer_state(self, mock_urlopen):
        """Test getting printer state."""
        mock_response = {
            "temperature": {
                "tool0": {"actual": 200, "target": 205},
                "bed": {"actual": 60, "target": 60}
            },
            "state": {"flags": {"printing": True}}
        }
        
        mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(mock_response).encode()
        mock_urlopen.return_value.__enter__.return_value.status = 200
        
        client = OctoPrintClient("http://localhost:5000", "test_key")
        
        # Mock the health check to avoid unnecessary requests
        client.health_check.check = Mock(return_value=True)
        client.health_check.ensure_connected = Mock(return_value=True)
        
        state = client.get_printer_state()
        
        self.assertEqual(state.nozzle_temp, 200)
        self.assertEqual(state.nozzle_target, 205)
        self.assertEqual(state.bed_temp, 60)
        self.assertTrue(state.is_printing)

    @patch('urllib.request.urlopen')
    def test_set_nozzle_temp_clamping(self, mock_urlopen):
        """Test that nozzle temperature is clamped to safe range."""
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"{}"
        mock_urlopen.return_value.__enter__.return_value.status = 200
        
        client = OctoPrintClient("http://localhost:5000", "test_key")
        client.health_check.check = Mock(return_value=True)
        
        # Try to set unsafe temperature
        client.set_nozzle_temp(300)  # Too high
        
        # Should have clamped to 260C
        call_args = mock_urlopen.call_args
        request_data = call_args[0][0].data
        request_dict = json.loads(request_data.decode())
        
        self.assertEqual(request_dict["targets"]["tool0"], 260)

    @patch('urllib.request.urlopen')
    def test_set_bed_temp_clamping(self, mock_urlopen):
        """Test that bed temperature is clamped to safe range."""
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"{}"
        mock_urlopen.return_value.__enter__.return_value.status = 200
        
        client = OctoPrintClient("http://localhost:5000", "test_key")
        client.health_check.check = Mock(return_value=True)
        
        # Try to set unsafe temperature
        client.set_bed_temp(150)  # Too high
        
        # Should have clamped to 110C
        call_args = mock_urlopen.call_args
        request_data = call_args[0][0].data
        request_dict = json.loads(request_data.decode())
        
        self.assertEqual(request_dict["target"], 110)


class TestDetection(unittest.TestCase):
    """Test Detection data class."""

    def test_detection_to_dict(self):
        """Test detection serialization."""
        det = Detection(
            failure_type=FailureType.SPAGHETTI,
            severity=Severity.HIGH,
            confidence=0.85,
            description="Test spaghetti",
            camera_id="camera_0"
        )
        
        d = det.to_dict()
        
        self.assertEqual(d["failure_type"], "spaghetti")
        self.assertEqual(d["severity"], "high")
        self.assertEqual(d["confidence"], 0.85)
        self.assertEqual(d["camera_id"], "camera_0")


class TestBayesianConfidenceCalibrator(unittest.TestCase):
    """Test Bayesian confidence calibration."""

    def test_calibrator_initial_confidence(self):
        """Test that initial calibration returns original confidence."""
        calibrator = BayesianConfidenceCalibrator()
        
        det = Detection(
            failure_type=FailureType.SPAGHETTI,
            severity=Severity.HIGH,
            confidence=0.8,
            description="Test"
        )
        
        features = {"edge_density": 50, "brightness": 128}
        calibrated = calibrator.calibrate_confidence(det, features)
        
        # Without history, should return original
        self.assertEqual(calibrated, 0.8)

    def test_calibrator_learning(self):
        """Test that calibrator learns from history."""
        calibrator = BayesianConfidenceCalibrator()
        
        det = Detection(
            failure_type=FailureType.SPAGHETTI,
            severity=Severity.HIGH,
            confidence=0.8,
            description="Test"
        )
        
        features = {"edge_density": 50, "brightness": 128}
        
        # Record detection as correct
        calibrator.record_detection(det, features, was_correct=True)
        
        # Record another similar detection as incorrect
        calibrator.record_detection(det, features, was_correct=False)
        
        # Now calibrate should adjust confidence
        calibrated = calibrator.calibrate_confidence(det, features)
        
        # Should be reduced due to 50% accuracy
        self.assertLess(calibrated, 0.8)


class TestTextureAnalyzer(unittest.TestCase):
    """Test texture analysis for spaghetti detection."""

    @patch('lib.print_monitor_enhanced.HAS_PIL', True)
    def test_texture_coherence_mock(self):
        """Test texture coherence computation with mock."""
        analyzer = TextureAnalyzer()
        
        # Create a mock image
        mock_image = MagicMock()
        
        # Mock numpy operations
        with patch('numpy.gradient') as mock_gradient:
            with patch('numpy.sqrt') as mock_sqrt:
                with patch('numpy.arctan2') as mock_arctan:
                    with patch('numpy.histogram') as mock_histogram:
                        mock_gradient.return_value = (np.ones((10, 10)), np.ones((10, 10)))
                        mock_sqrt.return_value = np.ones((10, 10))
                        mock_arctan.return_value = np.zeros((10, 10))
                        mock_histogram.return_value = (np.ones(8), np.linspace(0, 2*np.pi, 9))
                        
                        # This should work without errors
                        coherence = analyzer.compute_texture_coherence(mock_image)
                        
                        self.assertIsInstance(coherence, float)
                        self.assertGreaterEqual(coherence, 0)
                        self.assertLessEqual(coherence, 1)


class TestGCodeSilhouetteAnalyzer(unittest.TestCase):
    """Test G-code silhouette analysis."""

    def test_gcode_parsing(self):
        """Test G-code file parsing."""
        analyzer = GCodeSilhouetteAnalyzer()
        
        # Create temporary G-code file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gcode', delete=False) as f:
            f.write(";LAYER:0\n")
            f.write("G1 X10 Y20 Z0.2 E0 F1000\n")
            f.write("G1 X20 Y20 E5 F1000\n")
            f.write(";LAYER:1\n")
            f.write("G1 X10 Y20 Z0.4 E10 F1000\n")
            f.write("G1 X20 Y20 E15 F1000\n")
            gcode_path = f.name
        
        try:
            success = analyzer.parse_gcode(gcode_path)
            
            self.assertTrue(success)
            self.assertIn(0, analyzer.layer_geometries)
            self.assertIn(1, analyzer.layer_geometries)
            
            # Check that coordinates were extracted
            layer0 = analyzer.layer_geometries[0]
            self.assertGreater(len(layer0), 0)
            self.assertEqual(layer0[0], (10.0, 20.0))
        finally:
            Path(gcode_path).unlink()

    def test_gcode_parsing_empty_file(self):
        """Test G-code parsing with empty file."""
        analyzer = GCodeSilhouetteAnalyzer()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gcode', delete=False) as f:
            gcode_path = f.name
        
        try:
            success = analyzer.parse_gcode(gcode_path)
            self.assertTrue(success)
            self.assertEqual(len(analyzer.layer_geometries), 0)
        finally:
            Path(gcode_path).unlink()


class TestFrameAnalyzer(unittest.TestCase):
    """Test frame analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = MonitorConfig()
        self.analyzer = FrameAnalyzer(self.config)
        self.print_state = PrintState(is_printing=True, elapsed_sec=10)

    @patch('lib.print_monitor_enhanced.HAS_PIL', True)
    def test_frame_analysis_with_mock(self):
        """Test frame analysis with mocked image."""
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image.size = (640, 480)
        
        # Mock ImageStat
        with patch('lib.print_monitor_enhanced.ImageStat.Stat') as mock_stat:
            mock_stat_instance = MagicMock()
            mock_stat_instance.mean = [128]
            mock_stat_instance.stddev = [30]
            mock_stat.return_value = mock_stat_instance
            
            detections = self.analyzer.analyze(mock_image, self.print_state)
            
            # Should return list of detections
            self.assertIsInstance(detections, list)

    def test_set_reference_frame(self):
        """Test setting reference frame."""
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image.copy.return_value = mock_image
        mock_image.size = (640, 480)
        
        with patch('lib.print_monitor_enhanced.ImageStat.Stat') as mock_stat:
            mock_stat_instance = MagicMock()
            mock_stat_instance.mean = [128]
            mock_stat_instance.stddev = [30]
            mock_stat.return_value = mock_stat_instance
            
            self.analyzer.set_reference(mock_image)
            
            self.assertIn("default", self.analyzer.reference_frames)


class TestDecisionEngine(unittest.TestCase):
    """Test decision making."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = MonitorConfig(auto_pause_enabled=True)
        self.engine = DecisionEngine(self.config)
        self.print_state = PrintState()

    def test_high_confidence_spaghetti_triggers_pause(self):
        """Test that high-confidence spaghetti detection triggers pause."""
        detection = Detection(
            failure_type=FailureType.SPAGHETTI,
            severity=Severity.HIGH,
            confidence=0.95,
            description="High confidence spaghetti"
        )
        
        actions = self.engine.decide([detection], self.print_state)
        
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, ActionType.PAUSE)

    def test_low_confidence_detection_filtered(self):
        """Test that low-confidence detections are filtered."""
        self.config.confidence_threshold = 0.7
        
        detection = Detection(
            failure_type=FailureType.SPAGHETTI,
            severity=Severity.HIGH,
            confidence=0.5,  # Below threshold
            description="Low confidence"
        )
        
        actions = self.engine.decide([detection], self.print_state)
        
        self.assertEqual(len(actions), 0)

    def test_alert_cooldown(self):
        """Test alert cooldown mechanism."""
        detection = Detection(
            failure_type=FailureType.BLOB,
            severity=Severity.MEDIUM,
            confidence=0.8,
            description="Blob detected"
        )
        
        # First detection should generate action
        actions1 = self.engine.decide([detection], self.print_state)
        self.assertEqual(len(actions1), 1)
        
        # Second detection immediately after should be cooldownlocked
        actions2 = self.engine.decide([detection], self.print_state)
        self.assertEqual(len(actions2), 0)


class TestActionExecutor(unittest.TestCase):
    """Test action execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.config = MonitorConfig()
        self.executor = ActionExecutor(self.mock_client, self.config)
        self.print_state = PrintState()

    def test_alert_action(self):
        """Test alert action execution."""
        action = Action(
            action_type=ActionType.ALERT,
            reason="Test alert"
        )
        
        result = self.executor.execute(action, self.print_state)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "alert")

    def test_pause_action(self):
        """Test pause action execution."""
        self.mock_client.pause.return_value = {}
        
        action = Action(
            action_type=ActionType.PAUSE,
            reason="Pause requested"
        )
        
        result = self.executor.execute(action, self.print_state)
        
        self.assertTrue(result["success"])
        self.mock_client.pause.assert_called_once()

    def test_cancel_action(self):
        """Test cancel action execution."""
        self.mock_client.cancel.return_value = {}
        
        action = Action(
            action_type=ActionType.CANCEL,
            reason="Cancel requested"
        )
        
        result = self.executor.execute(action, self.print_state)
        
        self.assertTrue(result["success"])
        self.mock_client.cancel.assert_called_once()


class TestPrintHistory(unittest.TestCase):
    """Test print history recording."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.history_path = Path(self.temp_dir) / "history.json"

    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.temp_dir)

    def test_record_session(self):
        """Test recording a print session."""
        history = PrintHistory(str(self.history_path))
        
        detection = Detection(
            failure_type=FailureType.SPAGHETTI,
            severity=Severity.HIGH,
            confidence=0.8,
            description="Test"
        )
        
        action = Action(
            action_type=ActionType.PAUSE,
            reason="Pause due to spaghetti"
        )
        
        history.record_session(
            filename="test_print.gcode",
            detections=[detection],
            actions=[action],
            outcome="cancelled"
        )
        
        self.assertEqual(len(history.records), 1)
        record = history.records[0]
        self.assertEqual(record["filename"], "test_print.gcode")
        self.assertEqual(record["outcome"], "cancelled")
        self.assertEqual(len(record["detections"]), 1)

    def test_success_rate(self):
        """Test success rate calculation."""
        history = PrintHistory(str(self.history_path))
        
        # Add some records
        for i, outcome in enumerate(["success", "success", "failed"]):
            history.record_session(
                filename=f"test_{i}.gcode",
                detections=[],
                actions=[],
                outcome=outcome
            )
        
        # Should be 2/3
        self.assertAlmostEqual(history.get_success_rate(), 2/3, places=2)


class TestPrintMonitor(unittest.TestCase):
    """Test main print monitor."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = MonitorConfig(
            snapshot_dir=self.temp_dir,
            log_path=f"{self.temp_dir}/monitor.log",
            history_path=f"{self.temp_dir}/history.json"
        )
        self.monitor = PrintMonitor(self.config)
        self.monitor.client = MagicMock()
        self.monitor.client.get_printer_state.return_value = PrintState()

    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.temp_dir)

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        self.assertIsNotNone(self.monitor.client)
        self.assertIsNotNone(self.monitor.decision)
        self.assertIsNotNone(self.monitor.executor)
        self.assertIsNotNone(self.monitor.history)

    def test_pending_alerts(self):
        """Test alert queue management."""
        alert1 = {"action": "alert", "reason": "Test 1"}
        alert2 = {"action": "pause", "reason": "Test 2"}
        
        self.monitor.alerts_queue.extend([alert1, alert2])
        
        alerts = self.monitor.get_pending_alerts()
        
        self.assertEqual(len(alerts), 2)
        self.assertEqual(len(self.monitor.alerts_queue), 0)

    def test_generate_summary_report(self):
        """Test report generation."""
        report = self.monitor.generate_summary_report()
        
        self.assertIn("Print Monitor Session Report", report)
        self.assertIn("Total Snapshots", report)
        self.assertIn("Total Detections", report)

    def test_save_report(self):
        """Test saving report to file."""
        report_path = f"{self.temp_dir}/report.md"
        self.monitor.save_report(report_path)
        
        self.assertTrue(Path(report_path).exists())
        
        with open(report_path, 'r') as f:
            content = f.read()
            self.assertIn("Print Monitor Session Report", content)


class TestMonitorDaemon(unittest.TestCase):
    """Test monitor daemon."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = MonitorConfig(
            snapshot_dir=self.temp_dir,
            log_path=f"{self.temp_dir}/monitor.log"
        )

    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.temp_dir)

    def test_daemon_initialization(self):
        """Test daemon initialization."""
        daemon = MonitorDaemon(self.config)
        
        self.assertIsNotNone(daemon.monitor)
        self.assertFalse(daemon.is_running)

    def test_daemon_start_stop(self):
        """Test daemon start and stop."""
        daemon = MonitorDaemon(self.config)
        
        # Mock the monitoring loop to prevent long-running test
        daemon._monitoring_loop = MagicMock()
        
        daemon.start()
        self.assertTrue(daemon.is_running)
        
        daemon.stop()
        self.assertFalse(daemon.is_running)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = MonitorConfig(
            snapshot_dir=self.temp_dir,
            log_path=f"{self.temp_dir}/monitor.log",
            history_path=f"{self.temp_dir}/history.json"
        )

    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.temp_dir)

    @patch('lib.print_monitor_enhanced.OctoPrintClient')
    def test_full_monitoring_cycle(self, mock_client_class):
        """Test complete monitoring cycle."""
        mock_client = MagicMock()
        mock_client.get_printer_state.return_value = PrintState(
            is_printing=True,
            nozzle_temp=200,
            nozzle_target=205,
            progress_pct=50.0
        )
        mock_client_class.return_value = mock_client
        
        monitor = PrintMonitor(self.config)
        monitor.client = mock_client
        
        # Create detection
        detection = Detection(
            failure_type=FailureType.SPAGHETTI,
            severity=Severity.HIGH,
            confidence=0.9,
            description="Test spaghetti"
        )
        
        # Process through decision engine
        actions = monitor.decision.decide([detection], mock_client.get_printer_state())
        
        self.assertGreater(len(actions), 0)
        
        # Execute actions
        for action in actions:
            result = monitor.executor.execute(action, mock_client.get_printer_state())
            self.assertIsNotNone(result)
        
        # Record session
        monitor.record_outcome("cancelled", {})
        
        self.assertEqual(len(monitor.history.records), 1)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
