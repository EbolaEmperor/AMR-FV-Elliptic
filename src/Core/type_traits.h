#pragma once

// #include "Core/Tensor.h"
#include <concepts>
#include <iostream>
#include <type_traits>
#include <vector>

// typetraits for tensor
// template <typename T> struct is_tensor final : std::false_type {};
// template <typename T, int Dim>
// struct is_tensor<Tensor<T, Dim>> final : std::true_type {};

// template <typename T> struct is_real_tensor final : std::false_type {};
// template <int Dim>
// struct is_real_tensor<Tensor<Real, Dim>> final : std::true_type {};
// template <int Dim>
// struct is_real_tensor<Tensor<int, Dim>> final : std::true_type {};

// template <typename T> struct is_real_tensor2D final : std::false_type {};
// template <> struct is_real_tensor2D<Tensor<Real, 2>> final : std::true_type
// {}; template <> struct is_real_tensor2D<Tensor<int, 2>> final :
// std::true_type {};

template <typename T>
concept Number = std::is_scalar<T>::value;  // is a scalar number

template <typename T>
struct FixedSizeData : std::is_scalar<T> {};

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  for (auto &i : v) {
    std::cout << i << " ";
  }
  return os;
}
