#ifndef MESSAGE_QUEUE_DEFS_H_
#define MESSAGE_QUEUE_DEFS_H_

#include <iostream>
#include <queue>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

template<typename T>
class message_queue
{
private:
  std::queue<T> queue;
  boost::mutex mutex;
  boost::condition_variable cond_variable;
  bool state;

public:

  message_queue() {
    state = true;
  }

  void push(T const &data) {
    boost::mutex::scoped_lock lock(mutex);
    queue.push(data);
    lock.unlock();
    cond_variable.notify_one();
  }

  bool empty() const
  {
    if (queue.empty() && state)
      return true;

    return false;
  }

  bool read_message(T &data) {
    boost::mutex::scoped_lock lock(mutex);
    if (empty())
      return false;

    data = queue.back();
    clear();
    lock.unlock();
    return true;
  }

  void read_message_block(T &data) {
    boost::mutex::scoped_lock lock(mutex);
    while(empty()) {
      cond_variable.wait(lock);
    }
    if (!state)
      return;

    data = queue.back();
    clear();
    lock.unlock();
  }

  void close() {
    boost::mutex::scoped_lock lock(mutex);
    state = false;
    clear();
    lock.unlock();
    cond_variable.notify_one();
  }

  void clear() {
    // std::queue<T> aux_queue;
    // std::swap(queue, aux_queue);
    queue.pop();
  }

  size_t size() {
    return queue.size();
  }
};


#endif /* MESSAGE_QUEUE_DEFS_H_ */