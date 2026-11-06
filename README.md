# OmniSense v4.0: Unbounded Crowd Counting & Clinical-Grade Biometrics via COTS Wi-Fi

[![IEEE Format](https://img.shields.io/badge/Paper-IEEEtran_LaTeX-blue.svg)](paper/download.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hardware: ESP32](https://img.shields.io/badge/Hardware-ESP32_Xtensa-red.svg)](https://www.espressif.com/)

**OmniSense CSI v4.0 (IEEE Research Edition)** is a passive Wi-Fi radar platform capable of centimeter-level indoor tracking, unbounded crowd counting (10+ people), and clinical-grade PQRST vital sign extraction. Built entirely using sub-$15 Commercial Off-The-Shelf (COTS) ESP32 microcontrollers operating at 2.437 GHz (Channel 6).

---

## 🚀 Key Novelties & System Highlights
- **Unbounded Crowd Counting:** Replaces fragile FastICA pipelines with Random Matrix Theory (RMT) and Wax-Kailath MDL, guaranteeing zero person-splitting.
- **Dual-TX Bistatic Mesh:** Uses coprime injection rates (TX1 @ 100Hz on UDP 3333, TX2 @ 75Hz on UDP 3334) to completely eliminate CSMA/CA contention and blind tangents.
- **Specular Ghost Folding:** Employs a 2D Panoramic RF Digital Twin (Householder Reflection Operator) to fold multipath wall bounces, suppressing 98.5% of mirror ghosts.
- **Through-Wall CIR Range Gating:** Hann-windowed IFFT Channel Impulse Response rejects corridor interference exceeding 12.0 dB excess attenuation.
- **Clinical-Grade Biometrics:** Combines Variational Mode Decomposition (VMD) and Coiflet-4 Wavelets to resolve cardiac P-Q-R-S-T electrical morphology (0.85 BPM MAE).

---

## 🛠️ Hardware Setup & Spatial Geometry
- **Primary Injector (TX1):** ESP32 @ 100 Hz (`UDP Port 3333`)
- **Secondary Injector (TX2):** ESP32 @ 75 Hz (`UDP Port 3334`)
- **Radar CSI Receiver (RX):** ESP32 connected via UART @ 921,600 Baud
- **Access Point (AP):** SAM 4G Router (Channel 6, 2.437 GHz)

---

## ⚡ Quick Start & Deployment

### 1. Flash the ESP32 Nodes
Use the interactive flashing tool:

chmod +x flash_omnisense.sh
./flash_omnisense.sh

2. Start the Real-Time DSP Backend
Bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and start server
pip install -r requirements.txt
uvicorn backend.server:app --host 0.0.0.0 --port 8000
Access the tactical Cyber-HUD interface at http://localhost:8000.

📄 Research Paper
The full paper manuscript is available in the paper/ directory:

Title: OmniSense v4.0: Unbounded Crowd Counting and Clinical-Grade Biometrics via COTS Wi-Fi using Random Matrix Theory and Specular Digital Twins

Author: Ali Abdulkareem Ali Hadi Al-Fadhli (Al-Rasheed Smart University, Sana'a, Yemen)

Citation

@article{alfadhli2026omnisense,
  title={OmniSense v4.0: Unbounded Crowd Counting and Clinical-Grade Biometrics via COTS Wi-Fi using Random Matrix Theory and Specular Digital Twins},
  author={Al-Fadhli, Ali Abdulkareem Ali Hadi},
  journal={IEEE Research Preprints},
  year={2026}
}
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
