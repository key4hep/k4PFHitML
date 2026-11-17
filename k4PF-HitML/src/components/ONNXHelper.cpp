/*
 * Copyright (c) 2020-2024 Key4hep-Project.
 *
 * This file is part of Key4hep.
 * See https://key4hep.github.io/key4hep-doc/ for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "ONNXHelper.h"

#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>

// taken from https://github.com/key4hep/k4MLJetTagger/blob/main/k4MLJetTagger/src/components/ONNXRuntime.h

ONNXHelper::ONNXHelper(const std::string& model_path, int intraOpThreads, bool shrinkArenaAfterRun)
    : m_env(new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "onnx_runtime")), m_allocator(),
      m_cpu_mem_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  if (model_path.empty())
    throw std::runtime_error("Path to ONNX model cannot be empty!");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(intraOpThreads);
  if (shrinkArenaAfterRun) {
    // The CPU arena grows (in power-of-two chunks) to fit the largest activations seen so far and by default
    // keeps that memory for the lifetime of the session. With this run option every Run() hands fully free
    // chunks back to the system afterwards. It only affects this session's arena.
    m_runOptions.AddConfigEntry("memory.enable_memory_arena_shrinkage", "cpu:0");
  }
  std::string model{model_path};
  m_session = std::make_unique<Ort::Session>(*m_env, model.c_str(), options);

  // Get input names and shapes
  m_inputNodeStrings.clear();
  m_inputNodeDims.clear();

  for (size_t i = 0; i < m_session->GetInputCount(); ++i) {
    // get input names
    const auto input_name = m_session->GetInputNameAllocated(i, m_allocator).release();
    m_inputNodeStrings.emplace_back(input_name);

    // get input shapes
    const auto nodeInfo = m_session->GetInputTypeInfo(i);
    m_inputNodeDims[input_name] = nodeInfo.GetTensorTypeAndShapeInfo().GetShape();
  }

  // Get output names and shapes
  m_outputNodeStrings.clear();
  m_outputNodeDims.clear();
  for (size_t i = 0; i < m_session->GetOutputCount(); ++i) {
    // Get output names
    const auto output_name = m_session->GetOutputNameAllocated(i, m_allocator).release();
    m_outputNodeStrings.emplace_back(output_name);

    // get output shapes
    const auto nodeInfo = m_session->GetOutputTypeInfo(i);
    m_outputNodeDims[output_name] = nodeInfo.GetTensorTypeAndShapeInfo().GetShape();

    // the 0th dim depends on the batch size
    m_outputNodeDims[output_name].at(0) = -1;
  }
}

ONNXHelper::~ONNXHelper() {}

template <typename T>
ONNXHelper::Tensor<T> ONNXHelper::run(Tensor<T>& input, const Tensor<long>& input_shapes,
                                      unsigned long long batch_size) const {
  std::vector<Ort::Value> tensors_in;
  if (input.size() != m_inputNodeStrings.size()) {
    std::ostringstream msg;
    msg << "Number of provided inputs does not match model inputs. Provided " << input.size() << ", model expects "
        << m_inputNodeStrings.size() << ".";
    if (!m_inputNodeStrings.empty()) {
      msg << " Model inputs: ";
      for (size_t i = 0; i < m_inputNodeStrings.size(); ++i) {
        msg << m_inputNodeStrings[i];
        if (i + 1 != m_inputNodeStrings.size()) {
          msg << ", ";
        }
      }
    }
    throw std::runtime_error(msg.str());
  }
  for (std::size_t i = 0; i < m_inputNodeStrings.size(); ++i) {
    const auto& name = m_inputNodeStrings[i];
    auto& value = input[i];
    std::vector<int64_t> input_dims;
    if (input_shapes.empty()) {
      input_dims = m_inputNodeDims.at(name);
      input_dims[0] = batch_size;
    } else {
      input_dims = input_shapes[i];
    }
    // rely on the given input_shapes to set the batch size
    if (input_shapes.empty()) {
      input_dims = m_inputNodeDims.at(name);
      input_dims[0] = batch_size;

      if (input_dims[0] != static_cast<long>(batch_size)) {
        throw std::runtime_error("The first element of `input_shapes` (" + std::to_string(input_dims[0]) +
                                 ") does not match the given `batch_size` (" + std::to_string(batch_size) + ")");
      }
    } else {
      input_dims = input_shapes[i];
    }

    auto expected_len = std::accumulate(input_dims.begin(), input_dims.end(), 1, std::multiplies<int64_t>());
    if (expected_len != (int64_t)value.size())
      throw std::runtime_error("Input array '" + name + "' has a wrong size of " + std::to_string(value.size()) +
                               ", expected " + std::to_string(expected_len));

    auto input_tensor = Ort::Value::CreateTensor<float>(m_cpu_mem_info, value.data(), value.size(), input_dims.data(),
                                                        input_dims.size());
    if (!input_tensor.IsTensor())
      throw std::runtime_error("Failed to create an input tensor for variable '" + name + "'.");
    tensors_in.emplace_back(std::move(input_tensor));
  }

  // convert to char*
  std::vector<const char*> input_node_names;
  for (const auto& name_i : m_inputNodeStrings) {
    input_node_names.push_back(name_i.c_str());
  }

  std::vector<const char*> output_node_names;
  for (const auto& name_j : m_outputNodeStrings) {
    output_node_names.push_back(name_j.c_str());
  }

  // run the inference
  auto output_tensors = m_session->Run(m_runOptions, input_node_names.data(), tensors_in.data(), tensors_in.size(),
                                       output_node_names.data(), output_node_names.size());
  // convert output tensor to values
  Tensor<T> outputs;
  size_t i = 0;
  for (auto& output_tensor : output_tensors) {
    if (!output_tensor.IsTensor())
      throw std::runtime_error("(at least) inference output " + std::to_string(i) + " is not a tensor.");
    // get output shape
    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    auto length = tensor_info.GetElementCount();

    auto floatarr = output_tensor.GetTensorData<float>();
    outputs.emplace_back(floatarr, floatarr + length);
    ++i;
  }
  if (outputs.size() != m_session->GetOutputCount())
    throw std::runtime_error("Number of outputs differ from the expected one: got " + std::to_string(outputs.size()) +
                             ", expected " + std::to_string(m_session->GetOutputCount()));

  return outputs;
}

ONNXHelper::Tensor<float> ONNXHelper::runNamed(std::vector<ONNXInput>& inputs) const {
  std::vector<Ort::Value> tensors_in;

  for (const auto& expected_name : m_inputNodeStrings) {
    auto it =
        std::find_if(inputs.begin(), inputs.end(), [&](const ONNXInput& input) { return input.name == expected_name; });

    if (it == inputs.end()) {
      throw std::runtime_error("Missing ONNX input: " + expected_name);
    }

    const auto expected_len =
        std::accumulate(it->shape.begin(), it->shape.end(), int64_t{1}, std::multiplies<int64_t>());

    if (it->type == ONNXInput::Type::Float) {
      if (expected_len != static_cast<int64_t>(it->float_data.size())) {
        throw std::runtime_error("Wrong float input size for: " + expected_name);
      }

      tensors_in.emplace_back(Ort::Value::CreateTensor<float>(
          m_cpu_mem_info, it->float_data.data(), it->float_data.size(), it->shape.data(), it->shape.size()));
    } else {
      if (expected_len != static_cast<int64_t>(it->int64_data.size())) {
        throw std::runtime_error("Wrong int64 input size for: " + expected_name);
      }

      tensors_in.emplace_back(Ort::Value::CreateTensor<int64_t>(
          m_cpu_mem_info, it->int64_data.data(), it->int64_data.size(), it->shape.data(), it->shape.size()));
    }
  }

  std::vector<const char*> input_node_names;
  for (const auto& name : m_inputNodeStrings) {
    input_node_names.push_back(name.c_str());
  }

  std::vector<const char*> output_node_names;
  for (const auto& name : m_outputNodeStrings) {
    output_node_names.push_back(name.c_str());
  }

  auto output_tensors = m_session->Run(m_runOptions, input_node_names.data(), tensors_in.data(), tensors_in.size(),
                                       output_node_names.data(), output_node_names.size());

  Tensor<float> outputs;
  for (auto& output_tensor : output_tensors) {
    if (!output_tensor.IsTensor()) {
      throw std::runtime_error("runNamed output is not a tensor");
    }

    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto length = tensor_info.GetElementCount();
    auto floatarr = output_tensor.GetTensorData<float>();
    outputs.emplace_back(floatarr, floatarr + length);
  }

  return outputs;
}

template ONNXHelper::Tensor<float> ONNXHelper::run(Tensor<float>&, const Tensor<long>&, unsigned long long) const;