
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
#if DISPLAY_SUPPORT
  if (display_buf == NULL) {
    // Size of display_buf:
    // Frame 96x96 from camera is extrapolated to 192x192. RGB565 pixel format -> 2 bytes per pixel
    display_buf = (uint16_t *) heap_caps_malloc(96 * 2 * 96 * 2 * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (display_buf == NULL) {
    ESP_LOGE(TAG, "Couldn't allocate display buffer");
    return kTfLiteError;
  }
#endif // DISPLAY_SUPPORT

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


// Get an image from the camera module (MODIFIED FOR uint8 RGB)
TfLiteStatus GetImage(int image_width, int image_height, int channels, uint8_t* image_data) { // <<< CHANGED type of image_data
  //#if ESP_CAMERA_SUPPORTED
  
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
      ESP_LOGE(TAG, "Camera capture failed");
      return kTfLiteError;
  }

  // --- Start of RGB Conversion Logic for UINT8 Model ---

  // Verify dimensions and format
  if (fb->width != image_width || fb->height != image_height) {
      ESP_LOGE(TAG, "Camera frame size %dx%d doesn't match model input %dx%d",
                fb->width, fb->height, image_width, image_height);
      esp_camera_fb_return(fb);
      return kTfLiteError;
  }
  if (fb->format != PIXFORMAT_RGB565) {
      ESP_LOGE(TAG, "Camera format is %d, expected PIXFORMAT_RGB565 (%d)",
                fb->format, PIXFORMAT_RGB565);
      esp_camera_fb_return(fb);
      return kTfLiteError;
  }
  if (channels != 3) {
        ESP_LOGE(TAG, "Model expects %d channels, but we are providing 3 (RGB)", kNumChannels);
  }

  int input_index = 0;
  uint16_t* rgb565_buffer = (uint16_t*)fb->buf;

  // Loop through each pixel of the frame buffer
  for (int y = 0; y < image_height; y++) {
      for (int x = 0; x < image_width; x++) {
          uint16_t pixel = rgb565_buffer[y * image_width + x];

          // Extract 5-bit R, 6-bit G, 5-bit B
          uint8_t r5 = (pixel >> 11) & 0x1F;
          uint8_t g6 = (pixel >> 5)  & 0x3F;
          uint8_t b5 = pixel         & 0x1F;

          // Convert to 8-bit RGB (simple scaling)
          uint8_t r8 = (r5 * 255) / 31;
          uint8_t g8 = (g6 * 255) / 63;
          uint8_t b8 = (b5 * 255) / 31;

          // Place uint8_t values directly into the model input buffer (HWC format)
          // Ensure the order (R, G, B) matches what your specific model expects.
          image_data[input_index++] = r8; // <<< CHANGED: No subtraction
          image_data[input_index++] = g8; // <<< CHANGED: No subtraction
          image_data[input_index++] = b8; // <<< CHANGED: No subtraction
      }
  }

  // --- End of RGB Conversion Logic ---

  esp_camera_fb_return(fb);

  #if DISPLAY_SUPPORT
      ESP_LOGW(TAG, "Display update logic may need adjustment for RGB!");
  #endif

  // ESP_LOGI(TAG, "Image Captured (RGB uint8)"); // Optional log
  return kTfLiteOk;
  //#else // !ESP_CAMERA_SUPPORTED
  //    ESP_LOGE(TAG, "Camera not supported");
  //    return kTfLiteError;
  //#endif // ESP_CAMERA_SUPPORTED
  }