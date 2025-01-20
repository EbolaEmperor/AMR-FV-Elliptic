/**
 * @file ParallelDataBase.h
 * @author {JiatuYan} ({2513630371@qq.com})
 * @brief Base class for mpi data type
 * @version 0.1
 * @date 2024-03-22
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once
#include <Core/MPI.h>
#include <Core/type_traits.h>
#include <map>
#include <vector>

/**
 * @brief The basic class for types that need MPI communication.
 *
 * @tparam EnableRepeatComm Whether enable repeat communication
 * @tparam KnowRecvInfo Whether know the receiving information a priori,
 * or a MPI function should be called to set it.
 */
template <bool EnableRepeatComm, bool KnowRecvInfo>
class ParallelDataBase {
protected:
  /**
   * @brief a unique communicator for the object,
   * preventing tag colliding.
   *
   */
  Communicator comm_;

  /**
   * @brief The length of the data to send to each receivers.
   *
   */
  std::map<int, size_t> data_length_to_send_;

  /**
   * @brief The length of the data to receive from each senders.
   *
   */
  std::map<int, size_t> data_length_to_recv_;

  /**
   * @brief The buffer to store the data to send.
   *
   */
  std::map<int, std::vector<char>> send_buffer_;

  /**
   * @brief The buffer to store the data to receive.
   *
   */
  std::map<int, std::vector<char>> recv_buffer_;

  /**
   * @brief The requests for each sending operation.
   *
   */
  std::vector<MPI_Request> send_requests_;

  /**
   * @brief The requests for each receiving operation.
   *
   */
  std::vector<MPI_Request> recv_requests_;

  /**
   * @brief whether the communication is performing
   *
   */
  bool at_flight_;

public:
  /**
   * @brief Construct a new Parallel Data Base object
   *
   * @param comm the communicator copy from
   */
  ParallelDataBase(Communicator comm = MPI_COMM_WORLD);

  /**
   * @brief Construct a new Parallel Data Base object
   *
   * @param other
   */
  ParallelDataBase(const ParallelDataBase &other);

  ~ParallelDataBase() {
    if constexpr (EnableRepeatComm) {
      // free the persistent requests.
      for (auto &req : send_requests_)
        MPI_Request_free(&req);
      for (auto &req : recv_requests_)
        MPI_Request_free(&req);
    }
    MPI_Comm_free(&comm_);
  }

protected:
  /**
   * @brief Set the data length info already known by the sub-class
   *
   */
  virtual void setDataLengthInfo() {
    if (EnableRepeatComm)
      throw std::runtime_error("Should be implemented in the sub-class");
  };

  /**
   * @brief Sync the data length info and set all the member variables.
   *
   */
  void completeDataLengthInfo();

  /**
   * @brief Linearize the data to send into send_buffer_
   *
   */
  virtual void linearIn() = 0;

  /**
   * @brief Linearize out the data received from recv_buffer_
   *
   */
  virtual void linearOut() = 0;

  /**
   * @brief Initialze the repeat communication.
   *
   */
  void initialize() requires
      std::integral_constant<bool, EnableRepeatComm>::value;

  /**
   * @brief When the data length is changeable, set the data length from
   * the send_buffer_ which has already be filled by linearIn().
   *
   */
  void setDataLengthFromBuf() requires
      std::integral_constant<bool, !EnableRepeatComm>::value;
  /**
   * @brief allocate the persistent requests.
   *
   * @return std::enable_if<EnableRepeatComm>
   */
  void initializeCommunication() requires
      std::integral_constant<bool, EnableRepeatComm>::value;

  /**
   * @brief Start a temporary unblock communication.
   *
   */
  void startTempCommunication() requires
      std::integral_constant<bool, !EnableRepeatComm>::value;

  /**
   * @brief Allreduce the send_buffer[proc_id] to recv_buffer[proc_id].
   *
   */
  void beginAllReduce() requires
      std::integral_constant<bool, !EnableRepeatComm>::value;

  /**
   * @brief Start the communication
   *
   */
  void beginCommunication();

  /**
   * @brief End the communication
   *
   */
  void endCommunication();
};

template <bool EnableRepeatComm, bool KnowRecvInfo>
ParallelDataBase<EnableRepeatComm, KnowRecvInfo>::ParallelDataBase(
    Communicator comm) {
  // duplicate the communicator
  MPI_Comm_dup(comm, &comm_);
  at_flight_ = false;
}

template <bool EnableRepeatComm, bool KnowRecvInfo>
ParallelDataBase<EnableRepeatComm, KnowRecvInfo>::ParallelDataBase(
    const ParallelDataBase &other) :
    data_length_to_send_(other.data_length_to_send_),
    data_length_to_recv_(other.data_length_to_recv_),
    send_buffer_(other.send_buffer_),
    recv_buffer_(other.recv_buffer_),
    send_requests_(other.send_requests_),
    recv_requests_(other.recv_requests_) {
  MPI_Comm_dup(other.comm_, &comm_);
  if constexpr (EnableRepeatComm) {
    initializeCommunication();
  }
  at_flight_ = false;
};

template <bool EnableRepeatComm, bool KnowRecvInfo>
void ParallelDataBase<EnableRepeatComm, KnowRecvInfo>::initialize() requires
    std::integral_constant<bool, EnableRepeatComm>::value {
  setDataLengthInfo();
  completeDataLengthInfo();
  initializeCommunication();
};

template <bool EnableRepeatComm, bool KnowRecvInfo>
void ParallelDataBase<EnableRepeatComm,
                      KnowRecvInfo>::initializeCommunication() requires
    std::integral_constant<bool, EnableRepeatComm>::value {
  // allocate the persistent requests
  send_requests_.resize(data_length_to_send_.size());
  recv_requests_.resize(data_length_to_recv_.size());
  // set persistent communication.
  int cnt = 0;
  for (auto &send_info : data_length_to_send_) {
    if (send_info.first == ProcID(comm_)) {
      send_requests_.pop_back();
      continue;
    }
    MPI_Send_init(send_buffer_[send_info.first].data(),
                  send_info.second,
                  MPI_CHAR,
                  send_info.first,
                  0,
                  comm_,
                  &send_requests_[cnt]);
    cnt++;
  }
  cnt = 0;
  for (auto &recv_info : data_length_to_recv_) {
    if (recv_info.first == ProcID(comm_)) {
      recv_requests_.pop_back();
      continue;
    }
    MPI_Recv_init(recv_buffer_[recv_info.first].data(),
                  recv_info.second,
                  MPI_BYTE,
                  recv_info.first,
                  0,
                  comm_,
                  &recv_requests_[cnt]);
    cnt++;
  }
}

template <bool EnableRepeatComm, bool KnowRecvInfo>
void ParallelDataBase<EnableRepeatComm,
                      KnowRecvInfo>::beginAllReduce() requires
    std::integral_constant<bool, !EnableRepeatComm>::value {
  if (at_flight_)
    throw std::runtime_error("Communication is already at flight");
  linearIn();
  int world_size = numProcs(comm_);
  int proc_id = ProcID(comm_);
  std::vector<size_t> data_length(world_size, 0);
  std::vector<size_t> tmp_data_length(world_size, 0);
  tmp_data_length[proc_id] = send_buffer_[proc_id].size();
  MPI_Allreduce(tmp_data_length.data(),
                data_length.data(),
                data_length.size(),
                MPI_SIZE_T,
                MPI_SUM,
                comm_);
  recv_buffer_.clear();
  size_t total_length = 0, start_pole = 0;
  for (size_t i = 0; i != data_length.size(); ++i) {
    if ((size_t)proc_id == i) {
      start_pole = total_length;
    }
    total_length += data_length[i];
  }
  recv_buffer_[proc_id].resize(total_length, 0);
  std::vector<char> tmp_buffer(total_length, 0);
  std::copy(send_buffer_[proc_id].begin(),
            send_buffer_[proc_id].end(),
            tmp_buffer.begin() + start_pole);
  send_requests_.clear();
  recv_requests_.clear();
  MPI_Allreduce(tmp_buffer.data(),
                recv_buffer_[proc_id].data(),
                total_length,
                MPI_CHAR,
                MPI_SUM,
                comm_);
  at_flight_ = true;
};

template <bool EnableRepeatComm, bool KnowRecvInfo>
void ParallelDataBase<EnableRepeatComm, KnowRecvInfo>::beginCommunication() {
  if (at_flight_)
    throw std::runtime_error("Communication is already at flight");
  linearIn();
  if constexpr (!EnableRepeatComm) {
    setDataLengthFromBuf();
    completeDataLengthInfo();
    startTempCommunication();
  } else {
    if (send_requests_.size() != 0)
      MPI_Startall(send_requests_.size(), send_requests_.data());
    if (recv_requests_.size() != 0)
      MPI_Startall(recv_requests_.size(), recv_requests_.data());
  }
  at_flight_ = true;
}

template <bool EnableRepeatComm, bool KnowRecvInfo>
void ParallelDataBase<EnableRepeatComm, KnowRecvInfo>::endCommunication() {
  if (!at_flight_)
    throw std::runtime_error("Communication is not at flight");
  MPI_Waitall(send_requests_.size(), send_requests_.data(), MPI_STATUS_IGNORE);
  MPI_Waitall(recv_requests_.size(), recv_requests_.data(), MPI_STATUS_IGNORE);
  at_flight_ = false;
  linearOut();
}

template <bool EnableRepeatComm, bool KnowRecvInfo>
void ParallelDataBase<EnableRepeatComm,
                      KnowRecvInfo>::completeDataLengthInfo() {
  if constexpr (KnowRecvInfo) {
    return;
  }
  data_length_to_recv_.clear();
  recv_buffer_.clear();
  int world_size = numProcs(comm_);
  int proc_id = ProcID(comm_);
  std::vector<size_t> data_length(world_size * world_size, 0);
  for (auto &length : data_length_to_send_) {
    data_length[proc_id * world_size + length.first] = length.second;
  }
  MPI_Allreduce(data_length.data(),
                data_length.data(),
                data_length.size(),
                MPI_SIZE_T,
                MPI_SUM,
                comm_);
  for (int i = 0; i != world_size; ++i) {
    if (data_length[i * world_size + proc_id] > 0) {
      data_length_to_recv_[i] = data_length[i * world_size + proc_id];
      recv_buffer_[i].resize(data_length[i * world_size + proc_id]);
    }
  }
};

template <bool EnableRepeatComm, bool KnowRecvInfo>
void ParallelDataBase<EnableRepeatComm,
                      KnowRecvInfo>::startTempCommunication() requires
    std::integral_constant<bool, !EnableRepeatComm>::value {
  send_requests_.resize(data_length_to_send_.size());
  recv_requests_.resize(data_length_to_recv_.size());
  int cnt = 0;
  for (auto &send_info : data_length_to_send_) {
    if (send_info.first == ProcID(comm_)) {
      send_requests_.pop_back();
      continue;
    }
    MPI_Isend(send_buffer_[send_info.first].data(),
              send_info.second,
              MPI_CHAR,
              send_info.first,
              0,
              comm_,
              &send_requests_[cnt]);
    cnt++;
  }
  cnt = 0;
  for (auto &recv_info : data_length_to_recv_) {
    if (recv_info.first == ProcID(comm_)) {
      recv_requests_.pop_back();
      continue;
    }
    MPI_Irecv(recv_buffer_[recv_info.first].data(),
              recv_info.second,
              MPI_CHAR,
              recv_info.first,
              0,
              comm_,
              &recv_requests_[cnt]);
    cnt++;
  }
};
