#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_timer.h"
#include "driver/uart.h"

#define WIFI_SSID       "_ali7fa"
#define WIFI_PASS       "Ali2006ali211"
#define UART_PORT_NUM   UART_NUM_0
#define UART_BAUD_RATE  921600
#define NUM_SUBCARRIERS 52

typedef struct __attribute__((packed)) {
    uint16_t magic;         // 0x55AA
    uint32_t timestamp_us;
    uint8_t  source_mac[6];
    int8_t   rssi;
    int8_t   noise_floor;
    uint8_t  channel;
    uint8_t  reserved;
    int8_t   csi_iq[104];   // 52 subcarriers * 2 (Imaginary, Real)
    uint16_t crc16;
} omnisense_csi_frame_t;

static QueueHandle_t s_csi_queue = NULL;

static uint16_t calc_crc16(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= ((uint16_t)data[i] << 8);
        for (uint8_t j = 0; j < 8; j++) {
            crc = (crc & 0x8000) ? ((crc << 1) ^ 0x1021) : (crc << 1);
        }
    }
    return crc;
}

// دالة استدعاء استقبال CSI في طبقة العتاد (تعمل في سياق Wi-Fi Task)
static void IRAM_ATTR wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len < 128) {
        return;
    }

    omnisense_csi_frame_t frame;
    frame.magic        = 0x55AA;
    frame.timestamp_us = (uint32_t)esp_timer_get_time();
    memcpy(frame.source_mac, info->mac, 6);
    frame.rssi         = info->rx_ctrl.rssi;
    frame.noise_floor  = info->rx_ctrl.noise_floor;
    frame.channel      = info->rx_ctrl.channel;
    frame.reserved     = 0;

    // في ESP32 L-LTF (128 بايت = 64 ناقل)، النواقل النشطة الـ 52 هي:
    // الفهارس السالبة [-26..-1] تقع في الموضع (38..63) والفهارس الموجبة [+1..+26] في الموضع (1..26)
    const int8_t *raw = info->buf;
    int out_idx = 0;
    // استخراج النواقل السالبة [-26 .. -1]
    for (int sc = 38; sc <= 63; sc++) {
        frame.csi_iq[out_idx++] = raw[sc * 2];     // Imaginary
        frame.csi_iq[out_idx++] = raw[sc * 2 + 1]; // Real
    }
    // استخراج النواقل الموجبة [+1 .. +26]
    for (int sc = 1; sc <= 26; sc++) {
        frame.csi_iq[out_idx++] = raw[sc * 2];     // Imaginary
        frame.csi_iq[out_idx++] = raw[sc * 2 + 1]; // Real
    }

    frame.crc16 = calc_crc16((const uint8_t *)&frame, sizeof(omnisense_csi_frame_t) - 2);
    xQueueSendFromISR(s_csi_queue, &frame, NULL);
}

static void uart_stream_task(void *pvParameters) {
    omnisense_csi_frame_t frame;
    while (1) {
        if (xQueueReceive(s_csi_queue, &frame, portMAX_DELAY) == pdTRUE) {
            uart_write_bytes(UART_PORT_NUM, (const char *)&frame, sizeof(omnisense_csi_frame_t));
        }
    }
}

void app_main(void) {
    nvs_flash_init();
    s_csi_queue = xQueueCreate(64, sizeof(omnisense_csi_frame_t));

    // تهيئة UART0 بسرعة 921600 باود للبث الثنائي الفوري
    const uart_config_t uart_config = {
        .baud_rate  = UART_BAUD_RATE,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    uart_driver_install(UART_PORT_NUM, 4096, 4096, 0, NULL, 0);
    uart_param_config(UART_PORT_NUM, &uart_config);

    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    cfg.csi_enable = 1; // تفعيل دعم CSI في بنية التهيئة
    esp_wifi_init(&cfg);
    esp_wifi_set_mode(WIFI_MODE_STA);

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
        },
    };
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();
    esp_wifi_set_ps(WIFI_PS_NONE);
    esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20);
    esp_wifi_connect();

    // تفعيل الالتقاط المختلط لالتقاط كل من حزم المودم وحزم وحدة التحفيز TX
    esp_wifi_set_promiscuous(true);

    wifi_csi_config_t csi_config = {
        .lltf_en           = true,  // استخلاص 52 ناقلاً من حقل L-LTF عالي الاستقرار
        .htltf_en          = false,
        .stbc_htltf2_en    = false,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    xTaskCreatePinnedToCore(uart_stream_task, "uart_stream", 4096, NULL, 10, NULL, 0);
}
