/**
 * @file Linearization.h
 * @author {JiatuYan} ({2513630371@qq.com})
 * @brief General functions for linearIn and linearOut.
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once

#include "mpi.h"

#include <fstream>
#include <iostream>
#include <type_traits>
#include <vector>

template <class T>
concept need_manually_linearize = !std::is_scalar<T>::value;

class LinearizationHelper {
public:
  template <class T>
  static inline void linearIntoNewBuf(
      const T &input,
      std::vector<char> *buf) requires std::is_scalar<T>::value {
    buf->insert(buf->end(),
                reinterpret_cast<const char *>(&input),
                reinterpret_cast<const char *>(&input) + sizeof(T));
  };

  template <class T>
  static inline void linearIntoNewBuf(const T &input,
                                      std::vector<char> *buf) requires
      need_manually_linearize<T> {
    T::linearIntoNewBuf(input, buf);
  };

  template <class T>
  static inline void linearIntoNewBuf(const std::vector<T> &input,
                                      std::vector<char> *buf) {
    linearIntoNewBuf(input.size(), buf);
    for (auto &element : input) {
      linearIntoNewBuf(element, buf);
    }
  };

  /**
   * @brief linearize the data to a buf already set length.
   * Make sure buff have enough length.
   *
   * @tparam T given type
   * @param input
   * @param buf
   * @param pos
   */
  template <class T>
  static inline void linearIntoOldBuf(const T &input,
                                      std::vector<char> *buf,
                                      size_t *pos) {
    std::copy(reinterpret_cast<const char *>(&input),
              reinterpret_cast<const char *>(&input) + sizeof(T),
              buf->begin() + *pos);
    *pos += sizeof(T);
  };

  template <class T>
  static inline void linearIntoOldBuf(const T &input,
                                      std::vector<char> *buf,
                                      size_t *pos) requires
      need_manually_linearize<T> {
    T::linearIntoOldBuf(input, buf, pos);
  };

  template <class T>
  static inline void linearIntoOldBuf(const std::vector<T> &input,
                                      std::vector<char> *buf,
                                      size_t *pos) {
    size_t num_elements = input.size();
    std::copy(reinterpret_cast<const char *>(&num_elements),
              reinterpret_cast<const char *>(&num_elements) + sizeof(size_t),
              buf->begin() + *pos);
    *pos += sizeof(size_t);
    for (auto &element : input) {
      linearIntoOldBuf(element, buf, pos);
    }
  };

  template <class T>
  static inline void linearOut(const std::vector<char> &buf,
                               size_t *pos,
                               T *res) requires std::is_scalar<T>::value {
    std::copy(buf.begin() + *pos,
              buf.begin() + *pos + sizeof(T),
              reinterpret_cast<char *>(res));
    *pos += sizeof(T);
  };

  template <class T>
  static inline void linearOut(const std::vector<char> &buf,
                               size_t *pos,
                               T *res) requires need_manually_linearize<T> {
    T::linearOut(buf, pos, res);
  };

  template <class T>
  static inline void linearOut(const std::vector<char> &buf,
                               size_t *pos,
                               std::vector<T> *res) {
    size_t num_elements;
    linearOut(buf, pos, &num_elements);
    res->resize(num_elements);
    for (size_t i = 0; i != num_elements; ++i) {
      linearOut(buf, pos, &(res->at(i)));
    }
  };

  template <class T>
  static void writeData(const std::string &file_name,
                        const T &data,
                        int root) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == root) {
      std::vector<char> buffer;
      LinearizationHelper::linearIntoNewBuf(data, &buffer);
      std::ofstream fout(file_name, std::ios::out | std::ios::binary);
      fout.write(buffer.data(), buffer.size());
      fout.close();
    }
  }

  template <class T>
  static void readData(const std::string &file_name, T *data) {
    std::ifstream fin(file_name, std::ios::in | std::ios::binary);
    if (!fin) {
      std::cout << "open error!\n";
    }
    fin.seekg(0, fin.end);
    auto size = fin.tellg();
    std::vector<char> buffer(size);
    fin.seekg(0);
    fin.read(buffer.data(), size);
    size_t pos = 0;
    LinearizationHelper::linearOut(buffer, &pos, data);
    fin.close();
  }
};
