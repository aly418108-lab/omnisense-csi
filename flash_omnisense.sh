#!/usr/bin/env bash
# ==============================================================================
# OmniSense CSI v4.0 - Automated Hardware Flashing & Multi-Node Assistant
# Supports: Primary TX1 (100 Hz), Secondary TX2 (75 Hz), and RX Radar (921600 baud)
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IDF_EXPORT="/home/ali7fa/esp/esp-idf/export.sh"

echo "=============================================================================="
echo "    OmniSense CSI v4.0: أداة الحرق التلقائي للشريحة الثالثة والشبكة المتعددة"
echo "    Node 1: ESP32-TX1 (100 Hz @ Port 3333)"
echo "    Node 2: ESP32-TX2 (75 Hz @ Port 3334 - Secondary Injector)"
echo "    Node 3: ESP32-RX Radar (921600 baud Binary CSI Extractor)"
echo "    الشبكة المعمارية: SSID: _ali7fa | Pass: Ali2006ali211"
echo "=============================================================================="

# 1. تفعيل بيئة ESP-IDF
if [ -f "$IDF_EXPORT" ]; then
    echo "[+] تفعيل بيئة ESP-IDF v5.1..."
    source "$IDF_EXPORT" >/dev/null 2>&1
    echo "[✔] تم تجهيز بيئة ESP-IDF بنجاح."
else
    echo "[✘] خطأ: لم يتم العثور على export.sh في $IDF_EXPORT"
    exit 1
fi

# 2. فحص المنافذ المتاحة
find_esp_ports() {
    ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true
}

PORTS=($(find_esp_ports))

if [ ${#PORTS[@]} -eq 0 ]; then
    echo ""
    echo "------------------------------------------------------------------------------"
    echo "[!] تنبيه هام: لم يتم رصد أي منفذ تسلسلي (/dev/ttyUSB أو /dev/ttyACM) حتى الآن."
    echo "[!] السبب: كابل الـ USB الحالي يوصل كهرباء فقط (إضاءة اللمبة الحمراء)"
    echo "    ولا يحتوي على أسلاك نقل البيانات (D+ / D-)."
    echo ""
    echo "[⟳] الأداة الآن في وضع الاستماع المستمر (Waiting for Hardware)..."
    echo "    قم بتوصيل الشريحة بكابل نقل بيانات (Data Cable) في أي منفذ USB."
    echo "    بمجرد توصيل الكابل السليم سيتعرف النظام عليها ويبدأ الحرق فوراً!"
    echo "------------------------------------------------------------------------------"
    
    while [ ${#PORTS[@]} -eq 0 ]; do
        sleep 1
        PORTS=($(find_esp_ports))
    done
fi

echo ""
echo "[✔] تم اكتشاف منافذ نشطة: ${PORTS[*]}"

# إذا وُجد منفذ واحد
if [ ${#PORTS[@]} -eq 1 ]; then
    PORT1="${PORTS[0]}"
    echo ""
    echo "تم العثور على منفذ واحد متصل: $PORT1"
    echo "اختر ماذا تريد أن تحرق على هذه الشريحة:"
    echo "1) وحدة التحفيز الأساسية الأولى (ESP32-TX1 Injector @ 100 Hz)"
    echo "2) وحدة الاستقبال الراداري (ESP32-RX Radar @ 921600 baud)"
    echo "3) وحدة التحفيز الثانوية - الشريحة الثالثة (ESP32-TX2 Secondary @ 75 Hz)"
    read -p "أدخل رقم الخيار (1 أو 2 أو 3): " choice
    
    case $choice in
        1)
            echo "[*] جاري حرق وحدة التحفيز الأساسية (TX1) على $PORT1 بسرعة 921600 باود..."
            cd "$SCRIPT_DIR/firmware/esp32_tx_injector"
            idf.py -p "$PORT1" -b 921600 flash
            echo "[✔] تم حرق شريحة الإرسال الأساسية (TX1) بنجاح تام!"
            ;;
        2)
            echo "[*] جاري حرق وحدة الاستقبال الراداري (RX) على $PORT1 بسرعة 921600 باود..."
            cd "$SCRIPT_DIR/firmware/esp32_rx_radar"
            idf.py -p "$PORT1" -b 921600 flash
            echo "[✔] تم حرق شريحة الاستقبال (RX) بنجاح تام!"
            ;;
        3)
            echo "[*] جاري حرق وحدة التحفيز الثانوية للشريحة الثالثة (TX2) على $PORT1 بسرعة 921600 باود..."
            cd "$SCRIPT_DIR/firmware/esp32_tx_injector_secondary"
            idf.py -p "$PORT1" -b 921600 flash
            echo "[✔] تم حرق الشريحة الثالثة (TX2 Secondary) بنجاح تام!"
            ;;
    esac
fi

# إذا وُجد منفذان متصلان
if [ ${#PORTS[@]} -eq 2 ]; then
    TX_PORT="${PORTS[0]}"
    RX_PORT="${PORTS[1]}"
    echo ""
    echo "=============================================================================="
    echo "[✔] تم رصد شريحتين معاً: $TX_PORT و $RX_PORT"
    echo "=============================================================================="
    echo "1) حرق TX1 على المنفذ الأول و RX على المنفذ الثاني"
    echo "2) حرق TX2 (الشريحة الثالثة) على المنفذ الأول و RX على المنفذ الثاني"
    read -p "أدخل رقم الخيار (1 أو 2): " pair_choice
    
    if [ "$pair_choice" == "2" ]; then
        echo "[1/2] جاري حرق وحدة التحفيز الثانوية (TX2) على $TX_PORT..."
        cd "$SCRIPT_DIR/firmware/esp32_tx_injector_secondary"
        idf.py -p "$TX_PORT" -b 921600 flash
        echo "[✔] اكتمل حرق TX2!"
    else
        echo "[1/2] جاري حرق وحدة التحفيز الأساسية (TX1) على $TX_PORT..."
        cd "$SCRIPT_DIR/firmware/esp32_tx_injector"
        idf.py -p "$TX_PORT" -b 921600 flash
        echo "[✔] اكتمل حرق TX1!"
    fi
    
    echo "[2/2] جاري حرق وحدة الاستقبال الراداري (RX) على $RX_PORT..."
    cd "$SCRIPT_DIR/firmware/esp32_rx_radar"
    idf.py -p "$RX_PORT" -b 921600 flash
    echo "[✔] اكتمل حرق RX!"
fi

# إذا وُجد 3 منافذ متصلة (المنظومة ثلاثية العتاد كاملة)
if [ ${#PORTS[@]} -ge 3 ]; then
    P1="${PORTS[0]}"
    P2="${PORTS[1]}"
    P3="${PORTS[2]}"
    echo ""
    echo "=============================================================================="
    echo "🎉 تم رصد المنظومة الثلاثية الكاملة (3 Nodes Detected)!"
    echo "    1) Node 1 (TX1): $P1"
    echo "    2) Node 2 (TX2 Secondary): $P2"
    echo "    3) Node 3 (RX Radar): $P3"
    echo "=============================================================================="
    read -p "اضغط Enter لبدء الحرق الآلي للشرائح الثلاث معاً..." dummy
    
    echo "[1/3] جاري حرق TX1 على $P1..."
    cd "$SCRIPT_DIR/firmware/esp32_tx_injector"
    idf.py -p "$P1" -b 921600 flash
    
    echo "[2/3] جاري حرق TX2 على $P2..."
    cd "$SCRIPT_DIR/firmware/esp32_tx_injector_secondary"
    idf.py -p "$P2" -b 921600 flash
    
    echo "[3/3] جاري حرق RX Radar على $P3..."
    cd "$SCRIPT_DIR/firmware/esp32_rx_radar"
    idf.py -p "$P3" -b 921600 flash
    
    echo "=============================================================================="
    echo "  🎉 تهانينا! تم حرق وبرمجة الشرائح الثلاث بنجاح تام!"
    echo "=============================================================================="
fi
