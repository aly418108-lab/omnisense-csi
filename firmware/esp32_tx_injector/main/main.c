#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"

#define WIFI_SSID      "_ali7fa"
#define WIFI_PASS      "Ali2006ali211"
#define TARGET_UDP_IP  "255.255.255.255"
#define TARGET_PORT    3333
#define INJECT_RATE_HZ 100

static const char *TAG = "OMNISENSE_TX";
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

static void event_handler(void* arg, esp_event_base_t event_base,
                          int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void udp_traffic_injector_task(void *pvParameters) {
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdFALSE, portMAX_DELAY);
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    int broadcast = 1;
    setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));

    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(TARGET_UDP_IP);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(TARGET_PORT);

    uint32_t seq_id = 0;
    uint8_t payload[64];
    memset(payload, 0xA5, sizeof(payload));
    const TickType_t xFrequency = pdMS_TO_TICKS(1000 / INJECT_RATE_HZ);
    TickType_t xLastWakeTime = xTaskGetTickCount();

    ESP_LOGI(TAG, "Starting Active RF Excitation at %d PKT/S", INJECT_RATE_HZ);
    while (1) {
        memcpy(payload, &seq_id, sizeof(uint32_t));
        sendto(sock, payload, sizeof(payload), 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
        seq_id++;
        vTaskDelayUntil(&xLastWakeTime, xFrequency);
    }
}

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    s_wifi_event_group = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));

    xTaskCreatePinnedToCore(udp_traffic_injector_task, "udp_inject", 4096, NULL, 5, NULL, 1);
}
