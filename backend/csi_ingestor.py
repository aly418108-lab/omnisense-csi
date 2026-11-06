"""
OmniSense CSI - Pure ESP32 Hardware Ingestion Engine (v2.1)
Module: backend/csi_ingestor.py

Features:
1. 100% Pure Hardware Mode: Zero simulation code or dummy stubs.
2. Auto-Connect on Startup: Auto-scans and connects to /dev/ttyACM0 (fallback /dev/ttyUSB0) at 115200 baud.
3. Fault-Tolerant Non-Blocking Serial Loop: Automatic background reconnection on disconnect.
4. Real-Time Packet Rate & Metrics: Tracks packets per second (PPS) and RSSI.
5. L2 AGC Normalization: Normalizes amplitudes to reject high-power modem AGC fluctuations.
"""

import os
import glob
import time
import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

logger = logging.getLogger("OmniSense.Ingestor")


@dataclass
class CSIPacket:
    """Represents a single Channel State Information packet from hardware."""
    timestamp: float
    mac: str
    rssi: int
    subcarriers: np.ndarray  # Complex OFDM subcarriers, shape: (52,) or (64,)
    raw_len: int
    source: str = "hardware"
    metadata: Dict[str, Any] = field(default_factory=dict)
    norm_amplitudes: Optional[np.ndarray] = None  # L2-normalized amplitudes (0 - 100 scale)

    def __post_init__(self):
        if self.norm_amplitudes is None and self.subcarriers is not None and len(self.subcarriers) > 0:
            amps = np.abs(self.subcarriers).astype(np.float32)
            l2 = float(np.linalg.norm(amps))
            self.norm_amplitudes = (amps / (l2 + 1e-6)) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        amps = np.abs(self.subcarriers).round(2).tolist()
        phases = np.angle(self.subcarriers).round(3).tolist()
        norm_amps = self.norm_amplitudes.round(2).tolist() if self.norm_amplitudes is not None else amps
        return {
            "timestamp": self.timestamp,
            "mac": self.mac,
            "rssi": self.rssi,
            "amplitudes": amps,
            "norm_amplitudes": norm_amps,
            "phases": phases,
            "num_subcarriers": len(self.subcarriers),
            "source": self.source,
            "metadata": self.metadata,
        }


class ESP32SerialReader:
    """
    Direct hardware interface for ESP32 microcontrollers running ESP32-CSI-Tool.
    Defaulting to /dev/ttyACM0 @ 115200 baud with automatic port discovery and reconnection.
    """

    DEFAULT_PORT = "/dev/ttyACM0"
    DEFAULT_BAUDRATE = 115200

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, default_baud: Optional[int] = None):
        self.port = port or self.DEFAULT_PORT
        self.baudrate = default_baud if default_baud is not None else baudrate
        self.serial_conn: Optional[serial.Serial] = None
        self.is_connected = False
        self._consecutive_errors = 0

    @classmethod
    def list_available_ports(cls) -> List[Dict[str, str]]:
        """Lists detected serial ports, prioritizing /dev/ttyACM0 and /dev/ttyACM*."""
        ports = []
        if serial and hasattr(serial, 'tools') and hasattr(serial.tools, 'list_ports'):
            try:
                for p in serial.tools.list_ports.comports():
                    ports.append({
                        "device": p.device,
                        "description": p.description,
                        "hwid": p.hwid
                    })
            except Exception as e:
                logger.error(f"Error enumerating serial ports: {e}")

        # Linux TTY candidates
        linux_candidates = ["/dev/ttyACM0", "/dev/ttyUSB0"] + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
        existing_devices = {p["device"] for p in ports}
        for dev in linux_candidates:
            if dev not in existing_devices and os.path.exists(dev):
                ports.append({
                    "device": dev,
                    "description": "ESP32 USB Serial Interface",
                    "hwid": "n/a"
                })
        return ports

    @classmethod
    def auto_detect_esp32_port(cls) -> str:
        """Automatically discovers active ESP32 device path, strictly prioritizing /dev/ttyACM0 and /dev/ttyUSB0."""
        # 1. Check primary /dev/ttyACM0
        if os.path.exists(cls.DEFAULT_PORT):
            return cls.DEFAULT_PORT
        # 2. Check /dev/ttyUSB0
        if os.path.exists("/dev/ttyUSB0"):
            return "/dev/ttyUSB0"
        # 3. Check any other ttyACM or ttyUSB
        usb_candidates = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
        if usb_candidates:
            return sorted(usb_candidates)[0]
        # Always default to /dev/ttyACM0
        return cls.DEFAULT_PORT

    def connect(self) -> bool:
        """Opens the serial connection with non-blocking timeouts."""
        if serial is None:
            logger.error("pyserial is not installed in the python environment.")
            return False

        # If port does not exist on disk, attempt auto-detect
        if not os.path.exists(self.port):
            detected = self.auto_detect_esp32_port()
            if detected:
                self.port = detected

        try:
            self.disconnect()
            logger.info(f"Connecting to ESP32 on {self.port} at {self.baudrate} baud...")
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                write_timeout=0.5,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                rtscts=False,
                dsrdtr=False
            )
            # Flush existing buffers
            self.serial_conn.reset_input_buffer()
            self.is_connected = True
            self._consecutive_errors = 0
            logger.info(f"Successfully opened serial port {self.port} @ {self.baudrate} baud.")
            return True
        except Exception as e:
            logger.debug(f"Could not open serial port {self.port}: {e}")
            self.serial_conn = None
            self.is_connected = False
            return False

    def disconnect(self):
        """Safely closes serial connection."""
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        self.serial_conn = None
        self.is_connected = False

    def read_packet(self) -> Optional[CSIPacket]:
        """Reads and parses one CSI packet from serial line."""
        if not self.serial_conn or not self.is_connected:
            return None

        try:
            raw_line = self.serial_conn.readline()
            if not raw_line:
                return None

            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                return None

            packet = self.parse_line(line)
            if packet:
                self._consecutive_errors = 0
                return packet

        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors > 15:
                logger.warning(f"Serial read errors on {self.port}: {e}. Triggering reconnect...")
                self.disconnect()
            return None

        return None

    def parse_line(self, line: str) -> Optional[CSIPacket]:
        """
        Parses standard ESP32-CSI-Tool CSV lines.
        Format:
        CSI_DATA,type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,"[data...]"
        """
        if not line or "CSI_DATA" not in line:
            return None

        try:
            # Extract bracketed subcarrier array
            data_start = line.find('"[')
            if data_start == -1:
                data_start = line.find('[')
                data_end = line.find(']')
            else:
                data_start += 1
                data_end = line.find(']"')

            if data_start == -1 or data_end == -1:
                return None

            raw_csi_str = line[data_start + 1 : data_end].strip()
            if not raw_csi_str:
                return None

            header_part = line[:data_start].strip().rstrip(',"')
            tokens = header_part.split(',')

            if len(tokens) < 5:
                return None

            # Parse MAC and RSSI
            mac = tokens[3].strip().lower()
            rssi = int(tokens[4].strip())

            # Parse integer pairs [imag, real, imag, real, ...]
            int_vals = [int(val.strip()) for val in raw_csi_str.split(',') if val.strip()]
            num_pairs = len(int_vals) // 2

            if num_pairs < 52:
                return None

            # Reconstruct complex numbers: (real + 1j * imag)
            complex_sc = np.zeros(num_pairs, dtype=np.complex64)
            for k in range(num_pairs):
                imag_part = float(int_vals[2 * k])
                real_part = float(int_vals[2 * k + 1])
                complex_sc[k] = complex(real_part, imag_part)

            # Standardize to active 52 subcarriers
            if num_pairs == 64:
                # Remove 12 pilot/null subcarriers: take 52 active subcarriers
                complex_sc = np.concatenate([complex_sc[6:32], complex_sc[33:59]])
            elif num_pairs > 52:
                complex_sc = complex_sc[:52]

            return CSIPacket(
                timestamp=time.time(),
                mac=mac,
                rssi=rssi,
                subcarriers=complex_sc,
                raw_len=num_pairs,
                source="hardware",
                metadata={"raw_subcarrier_count": num_pairs}
            )

        except Exception:
            return None


class CSIIngestorManager:
    """
    Manages background serial reading and feeds the asyncio packet queue.
    Automatically connects on startup and retries upon disconnection.
    """

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.serial_reader = ESP32SerialReader(port=port, baudrate=baudrate)
        self.is_running = False
        self._worker_task: Optional[asyncio.Task] = None

        # Telemetry metrics
        self._packet_timestamps = deque(maxlen=200)
        self.total_packets = 0
        self.current_pps = 0.0

    async def start(self):
        """Starts background hardware ingestion loop."""
        if self.is_running:
            return
        self.is_running = True
        self._worker_task = asyncio.create_task(self._serial_loop())
        logger.info("CSIIngestorManager started in 100% PURE HARDWARE mode.")

    async def stop(self):
        """Stops hardware ingestion loop."""
        self.is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self.serial_reader.disconnect()
        logger.info("CSIIngestorManager stopped.")

    def reconnect(self, port: Optional[str] = None, baudrate: Optional[int] = None) -> Dict[str, Any]:
        """Manually reconnects serial interface."""
        if port:
            self.serial_reader.port = port
        if baudrate:
            self.serial_reader.baudrate = baudrate
        ok = self.serial_reader.connect()
        return {
            "status": "ok" if ok else "error",
            "connected": ok,
            "port": self.serial_reader.port,
            "baudrate": self.serial_reader.baudrate
        }

    def get_hardware_status(self) -> Dict[str, Any]:
        """Returns live hardware metrics for HUD status bar."""
        # Calculate current packets per second
        now = time.time()
        while self._packet_timestamps and (now - self._packet_timestamps[0] > 1.0):
            self._packet_timestamps.popleft()
        self.current_pps = float(len(self._packet_timestamps))

        return {
            "connected": self.serial_reader.is_connected,
            "port": self.serial_reader.port,
            "baudrate": self.serial_reader.baudrate,
            "packets_per_sec": round(self.current_pps, 1),
            "total_packets": self.total_packets
        }

    async def _serial_loop(self):
        """Continuous non-blocking serial acquisition loop."""
        # Initial connect attempt
        self.serial_reader.connect()

        last_retry = time.time()

        while self.is_running:
            try:
                # If not connected, retry auto-connecting every 2.0s
                if not self.serial_reader.is_connected:
                    now = time.time()
                    if now - last_retry >= 2.0:
                        last_retry = now
                        self.serial_reader.connect()
                    await asyncio.sleep(0.05)
                    continue

                # Read packet in executor to avoid blocking event loop
                loop = asyncio.get_running_loop()
                packet = await loop.run_in_executor(None, self.serial_reader.read_packet)

                if packet:
                    self.total_packets += 1
                    t = time.time()
                    self._packet_timestamps.append(t)

                    # Push to queue (drop oldest if full to avoid lag)
                    if self.queue.full():
                        try:
                            self.queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await self.queue.put(packet)
                else:
                    await asyncio.sleep(0.005)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in serial worker loop: {e}", exc_info=False)
                await asyncio.sleep(0.05)
