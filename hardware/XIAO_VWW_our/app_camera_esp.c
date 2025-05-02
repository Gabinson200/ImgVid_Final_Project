/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "app_camera_esp.h" // Includes defines like CAMERA_PIXEL_FORMAT, CAMERA_FRAME_SIZE
#include "sdkconfig.h"      // Includes Kconfig options like CONFIG_TFLITE_USE_BSP, etc.

// Required for camera functions
#include "esp_camera.h"
// Required for logging
#include "esp_log.h"
// Required for GPIO config (only if using JTAG pin fixup)
#include "driver/gpio.h"

#if (CONFIG_TFLITE_USE_BSP) // Include BSP header only if BSP is actually used
#include "bsp/esp-bsp.h"
#endif

// Define TAG for logging
static const char *TAG = "app_camera";

// Function definition
int app_camera_init() {
    ESP_LOGI(TAG, "app_camera_init() called"); // Logging: Function start

#if ESP_CAMERA_SUPPORTED // Main check if camera support is enabled in ESP-IDF

// JTAG Pin Fixup (Only needed for specific boards, keep it for compatibility)
#if CONFIG_CAMERA_MODULE_ESP_EYE || CONFIG_CAMERA_MODULE_ESP32_CAM_BOARD
    // IO13, IO14 is designed for JTAG by default,
    // to use it as generalized input,
    // firstly declare it as pullup input
    gpio_config_t conf;
    conf.mode = GPIO_MODE_INPUT;
    conf.pull_up_en = GPIO_PULLUP_ENABLE;
    conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    conf.intr_type = GPIO_INTR_DISABLE;
    conf.pin_bit_mask = 1LL << 13;
    gpio_config(&conf);
    conf.pin_bit_mask = 1LL << 14;
    gpio_config(&conf);
#endif // CONFIG_CAMERA_MODULE_ESP_EYE || CONFIG_CAMERA_MODULE_ESP32_CAM_BOARD


    // Configure Camera based on BSP usage or manual pins
#if (CONFIG_TFLITE_USE_BSP)
    ESP_LOGI(TAG, "Using BSP for camera configuration.");
    bsp_i2c_init(); // Initialize I2C (might be needed by BSP)
    camera_config_t config = BSP_CAMERA_DEFAULT_CONFIG;
    // Note: If BSP_CAMERA_DEFAULT_CONFIG is wrong for target board, this path will fail.

#else // CONFIG_TFLITE_USE_BSP is disabled, configure manually
    ESP_LOGI(TAG, "Using manual pin configuration (from image).");
    camera_config_t config; // Declare config structure

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;

    // --- Pin Configuration based on provided image --- <<< UPDATED SECTION
    config.pin_d0 = 15;       // DVP_Y2
    config.pin_d1 = 17;       // DVP_Y3
    config.pin_d2 = 18;       // DVP_Y4
    config.pin_d3 = 16;       // DVP_Y5
    config.pin_d4 = 14;       // DVP_Y6
    config.pin_d5 = 12;       // DVP_Y7
    config.pin_d6 = 11;       // DVP_Y8
    config.pin_d7 = 48;       // DVP_Y9
    config.pin_xclk = 10;       // XMCLK
    config.pin_pclk = 13;       // DVP_PCLK
    config.pin_vsync = 38;      // DVP_VSYNC
    config.pin_href = 47;       // DVP_HREF
    config.pin_sccb_sda = 40;   // CAM_SDA (SIOD) <<< Use correct name
    config.pin_sccb_scl = 39;   // CAM_SCL (SIOC) <<< Use correct name
    config.pin_pwdn = -1;       // Assuming not specified/used
    config.pin_reset = -1;      // Assuming not specified/used
    // --- End of Pin Configuration ---

    // --- Other Camera Configurations ---
    // Try lower XCLK freq first for troubleshooting detection
    config.xclk_freq_hz = 10000000; // XCLK frequency (10MHz) <<< TRY THIS FIRST
    // config.xclk_freq_hz = 20000000; // XCLK frequency (20MHz is common)
    config.jpeg_quality = 12;       // JPEG quality (0-63) lower means higher quality (less relevant for non-JPEG)
    config.fb_count = 1;            // Number of frame buffers. Try 1 first if memory is tight or for troubleshooting.
                                    // Use 2 for smoother capture later if init works.
    config.fb_location = CAMERA_FB_IN_PSRAM; // Frame buffer location (PSRAM is necessary for larger images/multiple buffers)
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY; // Or CAMERA_GRAB_LATEST. WHEN_EMPTY is generally safer.

#endif // CONFIG_TFLITE_USE_BSP


    // --- Apply Model-Specific Settings (Likely from app_camera_esp.h via Kconfig) ---
    ESP_LOGI(TAG, "Setting Pixel Format: %d, Frame Size: %d", CAMERA_PIXEL_FORMAT, CAMERA_FRAME_SIZE);
    config.pixel_format = CAMERA_PIXEL_FORMAT; // Should be PIXFORMAT_GRAYSCALE or PIXFORMAT_RGB565
    config.frame_size = CAMERA_FRAME_SIZE;     // Should be FRAMESIZE_96X96


    // --- Initialize the Camera ---
    // The esp_camera_init function attempts to auto-detect the sensor (OV2640, OV5640, etc.)
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x (%s)", err, esp_err_to_name(err));
        // Specific check for detection failure
        if (err == ESP_ERR_CAMERA_NOT_DETECTED) {
             ESP_LOGE(TAG, "Camera not detected on I2C bus (Pins SDA=%d, SCL=%d). Check power, connections, XCLK, and sensor.", config.pin_sccb_sda, config.pin_sccb_scl); // <<< Use correct name
        }
        // Check for frame size issues
        if (err == ESP_ERR_CAMERA_FAILED_TO_SET_FRAME_SIZE) {
             ESP_LOGE(TAG, "Failed to set frame size %d.", config.frame_size);
        }
        // Check for pixel format issues (using corrected error code)
        if (err == ESP_ERR_CAMERA_FAILED_TO_SET_OUT_FORMAT) { // <<< Use correct error code
             ESP_LOGE(TAG, "Failed to set pixel format %d.", config.pixel_format);
        }
        return -1; // Indicate failure
    }
    ESP_LOGI(TAG, "esp_camera_init() successful.");


    // --- Get Sensor Handle and Apply Sensor-Specific Settings ---
    sensor_t *s = esp_camera_sensor_get();
    if (s == NULL) {
        ESP_LOGE(TAG, "Failed to get camera sensor handle after successful init.");
        // Attempt to deinitialize if init succeeded but sensor is NULL
        esp_camera_deinit();
        return -1;
    }

    // Log detected sensor PID
    ESP_LOGI(TAG, "Detected Camera PID=0x%02x", s->id.PID);

    // Common setting: Flip image vertically
    s->set_vflip(s, 1);
    ESP_LOGI(TAG, "Set vertical flip");

    // Example: Apply specific settings based on detected PID
    if (s->id.PID == OV2640_PID) {
        ESP_LOGI(TAG, "OV2640 detected.");
        // Add any OV2640 specific register writes or settings here if needed
    } else if (s->id.PID == OV5640_PID) {
        ESP_LOGI(TAG, "OV5640 detected.");
        // Add any OV5640 specific register writes or settings here if needed
    } else if (s->id.PID == OV3660_PID){
         ESP_LOGI(TAG, "OV3660 detected.");
         // Settings from original code
         s->set_brightness(s, 1);
         s->set_saturation(s, -2);
    } else {
        ESP_LOGI(TAG, "Detected camera PID=0x%02x, applying generic settings.", s->id.PID);
    }


    ESP_LOGI(TAG, "Camera Initialized Successfully");
    return 0; // Indicate success


#else // ESP_CAMERA_SUPPORTED is disabled
    ESP_LOGE(TAG, "Camera support is not enabled in Kconfig (CONFIG_ESP_CAMERA_SUPPORTED)");
    return -1; // Indicate failure
#endif // ESP_CAMERA_SUPPORTED

} // <<< Closing brace for the app_camera_init function