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

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"

#include <cmath> // Required for expf for logits calculation


// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

#if CONFIG_NN_OPTIMIZED
constexpr int scratchBufSize = 60 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
// An area of memory to use for input, output, and intermediate arrays.
// Keeping allocation on bit larger size to accomodate future needs.
constexpr int kTensorArenaSize = 100 * 1024 + scratchBufSize;
static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace

// Applies softmax function in-place to an array of scores
void apply_softmax(float* scores, int count) {
    if (count <= 0) return;

    // Find max score for numerical stability
    float max_score = scores[0];
    for (int i = 1; i < count; ++i) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    // Calculate sum of exponentials
    float sum_exp = 0.0f;
    for (int i = 0; i < count; ++i) {
        // Subtract max_score before exponentiating to avoid large numbers
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }

    // Normalize by dividing by the sum
    if (sum_exp > 0.0f) { // Avoid division by zero
         for (int i = 0; i < count; ++i) {
            scores[i] /= sum_exp;
        }
    }
}

// The name of this function is important for Arduino compatibility.
void setup() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_rgb_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<7> micro_op_resolver;
  //micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddMean();


  //micro_op_resolver.AddMaxPool2D();
  //micro_op_resolver.AddPad();
  //micro_op_resolver.AddDepthwiseConv2D();
  //micro_op_resolver.AddReshape();
  //micro_op_resolver.AddSoftmax();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

  // --- Print Input Tensor Details ---
  MicroPrintf("Input Tensor Details:");
  MicroPrintf("  Dimensions: %d", input->dims->size);
  for (int i = 0; i < input->dims->size; ++i) {
    MicroPrintf("    Dim %d: %d", i, input->dims->data[i]);
  }
  MicroPrintf("  Type: %s", TfLiteTypeGetName(input->type));
  // Verify this matches model expectations (e.g., [-1, 96, 96, 3] and uint8)


#ifndef CLI_ONLY_INFERENCE
  // Initialize Camera
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif
}

#ifndef CLI_ONLY_INFERENCE
// The name of this function is important for Arduino compatibility.
void loop() {
  // Get image from provider.
  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.uint8)) {
    MicroPrintf("Image capture failed.");
  }

  // --- DEBUG: Print first few input pixel values --- <<< ADD THIS BLOCK
  //MicroPrintf("Input Data Sample (first 5 RGB triplets):");
  //for (int i = 0; i < 5 * 3; i += 3) {
  //    MicroPrintf("  [%d] R:%d G:%d B:%d", i/3, input->data.uint8[i], input->data.uint8[i+1], input->data.uint8[i+2]);
  //}

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;
  //float output_scale = 0.02039806731045246f; // <<< NEW scale from Netron
  //int output_zero_point = 130;              // <<< NEW zero_point from Netron

  // Log parameters once to see what they are
  static bool params_logged = false;
  if (!params_logged) {
      MicroPrintf("Output tensor: scale=%.6f, zero_point=%d", output_scale, output_zero_point);
      params_logged = true;
  }
  // Dequantize using the correct formula
  uint8_t person_score_uint8 = output->data.uint8[kPersonIndex];
  uint8_t no_person_score_uint8 = output->data.uint8[kNotAPersonIndex];
  float person_logits = output_scale * (static_cast<float>(person_score_uint8) - output_zero_point);
  float no_person_logits = output_scale * (static_cast<float>(no_person_score_uint8) - output_zero_point);

  // --- Apply Softmax to convert logits to probabilities
  // Create an array of the dequantized logits
  float logits[2] = {no_person_logits, person_logits}; // Assuming kNotAPersonIndex=0, kPersonIndex=1

  // Apply softmax in-place
  apply_softmax(logits, 2);

  // Extract the final probabilities
  float person_score_f = logits[kPersonIndex];         // Get probability after softmax
  float no_person_score_f = logits[kNotAPersonIndex];   // Get probability after softmax
    

  // Respond to detection
  RespondToDetection(person_score_f, no_person_score_f);
  vTaskDelay(1); // to avoid watchdog trigger
}
#endif

#if defined(COLLECT_CPU_STATS)
  long long total_time = 0;
  long long start_time = 0;
  extern long long softmax_total_time;
  extern long long dc_total_time;
  extern long long conv_total_time;
  extern long long fc_total_time;
  extern long long pooling_total_time;
  extern long long add_total_time;
  extern long long mul_total_time;
#endif

void run_inference(void *ptr) {
  MicroPrintf("run_inference called");

  // --- INPUT HANDLING FOR CLI MODE --- <<< CHANGED
  // This now depends on what 'ptr' points to.
  // If ptr points to raw RGB888 uint8_t data compatible with the model input:
  memcpy(input->data.uint8, ptr, kNumCols * kNumRows * kNumChannels);
  // If ptr points to something else (like grayscale), you need conversion here.
  // Example: If ptr points to uint8_t grayscale:
  // uint8_t* gray_ptr = (uint8_t*)ptr;
  // for (int i = 0; i < kNumCols * kNumRows; ++i) {
  //   input->data.uint8[i*3 + 0] = gray_ptr[i]; // R = Gray
  //   input->data.uint8[i*3 + 1] = gray_ptr[i]; // G = Gray
  //   input->data.uint8[i*3 + 2] = gray_ptr[i]; // B = Gray
  // }
  // Choose the appropriate handling based on your CLI image data format.


#if defined(COLLECT_CPU_STATS)
  long long start_time = esp_timer_get_time();
#endif
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

#if defined(COLLECT_CPU_STATS)
  long long total_time = (esp_timer_get_time() - start_time);
  printf("Total time = %lld\n", total_time / 1000);
  //printf("Softmax time = %lld\n", softmax_total_time / 1000);
  printf("FC time = %lld\n", fc_total_time / 1000);
  printf("DC time = %lld\n", dc_total_time / 1000);
  printf("conv time = %lld\n", conv_total_time / 1000);
  printf("Pooling time = %lld\n", pooling_total_time / 1000);
  printf("add time = %lld\n", add_total_time / 1000);
  printf("mul time = %lld\n", mul_total_time / 1000);

  /* Reset times */
  total_time = 0;
  //softmax_total_time = 0;
  dc_total_time = 0;
  conv_total_time = 0;
  fc_total_time = 0;
  pooling_total_time = 0;
  add_total_time = 0;
  mul_total_time = 0;
#endif

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  //int8_t person_score = output->data.uint8[kPersonIndex];
  //int8_t no_person_score = output->data.uint8[kNotAPersonIndex];

  //float person_score_f =
  //    (person_score - output->params.zero_point) * output->params.scale;
  //float no_person_score_f =
  //    (no_person_score - output->params.zero_point) * output->params.scale;
  //RespondToDetection(person_score_f, no_person_score_f);

  // Access output data as uint8_t
  uint8_t person_score_uint8 = output->data.uint8[kPersonIndex];
  uint8_t no_person_score_uint8 = output->data.uint8[kNotAPersonIndex];

  // Convert to float scores (assuming output range 0-255 maps to probability 0.0-1.0)
  float person_score_f = person_score_uint8 / 255.0f;
  float no_person_score_f = no_person_score_uint8 / 255.0f;

  RespondToDetection(person_score_f, no_person_score_f);
}
