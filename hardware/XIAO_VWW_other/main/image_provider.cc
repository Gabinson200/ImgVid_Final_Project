
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
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_spi_flash.h"
#include "esp_system.h"
#include "esp_timer.h"

#include "app_camera_esp.h"
#include "esp_camera.h"
#include "model_settings.h"
#include "image_provider.h"
#include "esp_main.h"

//static const char* TAG = "app_camera";
static const char* TAG = "image_provider"; // Changed TAG slightly for clarity

static uint16_t *display_buf; // buffer to hold data to be sent to display

// Get the camera module ready
TfLiteStatus InitCamera() {
  #if CLI_ONLY_INFERENCE
    ESP_LOGI(TAG, "CLI_ONLY_INFERENCE enabled, skipping camera init");
    return kTfLiteOk;
  #endif
  // if display support is present, initialise display buf

  #if ESP_CAMERA_SUPPORTED
    int ret = app_camera_init();
    if (ret != 0) {
      MicroPrintf("Camera init failed\n");
      return kTfLiteError;
    }
    MicroPrintf("Camera Initialized\n");
  #else
    ESP_LOGE(TAG, "Camera not supported for this device");
  #endif
    return kTfLiteOk;
}

void *image_provider_get_display_buf()
{
  return (void *) display_buf;
}


// Get an image, downscale, and convert for UINT8 RGB 48x48 model
TfLiteStatus GetImage(int image_width, int image_height, int channels, uint8_t* image_data) {
  #if ESP_CAMERA_SUPPORTED
      // Request camera frame (e.g., 96x96 as defined in app_camera_esp.h)
      camera_fb_t* fb = esp_camera_fb_get();
      if (!fb) {
          ESP_LOGE(TAG, "Camera capture failed");
          return kTfLiteError;
      }
  
      // --- Start of Downscaling and RGB Conversion ---
  
      // Basic sanity checks
      if (fb->format != PIXFORMAT_RGB565) {
          ESP_LOGE(TAG, "Camera format is %d, expected PIXFORMAT_RGB565 (%d)", fb->format, PIXFORMAT_RGB565);
          esp_camera_fb_return(fb);
          return kTfLiteError;
      }
  
      // We expect image_width/height to be the *model's* input size (e.g., 48)
      // and fb->width/height to be the *camera's* output size (e.g., 96)
      int source_width = fb->width;
      int source_height = fb->height;
      int target_width = image_width;   // Should be kNumCols (48)
      int target_height = image_height;  // Should be kNumRows (48)
  
      if (target_width > source_width || target_height > source_height) {
           ESP_LOGE(TAG, "Model size (%dx%d) larger than camera frame (%dx%d)!",
                    target_width, target_height, source_width, source_height);
           esp_camera_fb_return(fb);
           return kTfLiteError;
      }
  
      ESP_LOGD(TAG, "Downscaling %dx%d -> %dx%d", source_width, source_height, target_width, target_height);
  
      uint16_t* source_buf = (uint16_t*)fb->buf;
      int input_index = 0;
  
      // Calculate scaling factors
      float width_scale = (float)source_width / target_width;
      float height_scale = (float)source_height / target_height;
  
      // Averaging Downscaling loop
      for (int ty = 0; ty < target_height; ty++) { // Target Y (0 to 47)
        int sy_start = ty * 2; // Source Y start (0, 2, 4...)
        for (int tx = 0; tx < target_width; tx++) { // Target X (0 to 47)
            int sx_start = tx * 2; // Source X start (0, 2, 4...)

            // Accumulate sum of 2x2 block
            uint32_t r_sum = 0, g_sum = 0, b_sum = 0;
            for (int y_offset = 0; y_offset < 2; ++y_offset) {
                for (int x_offset = 0; x_offset < 2; ++x_offset) {
                    int sy = sy_start + y_offset;
                    int sx = sx_start + x_offset;
                    uint16_t pixel = source_buf[sy * source_width + sx];

                    // Extract 5/6/5 bits
                    uint8_t r5 = (pixel >> 11) & 0x1F;
                    uint8_t g6 = (pixel >> 5)  & 0x3F;
                    uint8_t b5 = pixel         & 0x1F;

                    // Convert to 8-bit and add to sum
                    r_sum += (r5 * 255) / 31;
                    g_sum += (g6 * 255) / 63;
                    b_sum += (b5 * 255) / 31;
                }
            }

            // Calculate average and store (check RGB vs BGR order needed)
            image_data[input_index++] = (uint8_t)(r_sum / 4); // R avg
            image_data[input_index++] = (uint8_t)(g_sum / 4); // G avg
            image_data[input_index++] = (uint8_t)(b_sum / 4); // B avg
        }
      }
      // --- End of Downscaling and RGB Conversion ---
  
      esp_camera_fb_return(fb);
  
      #if DISPLAY_SUPPORT
          ESP_LOGW(TAG, "Display update logic needs adjustment for 48x48!");
      #endif
  
      // ESP_LOGI(TAG, "Image Captured & Downscaled (RGB uint8)"); // Optional log
      return kTfLiteOk;
  #else // !ESP_CAMERA_SUPPORTED
      ESP_LOGE(TAG, "Camera not supported");
      return kTfLiteError;
  #endif // ESP_CAMERA_SUPPORTED
  }