#ifndef CIRCULAR_BUFFER
#define CIRCULAR_BUFFER

#include <type_traits>

template <typename item_t>
class CircularBuffer {
  private:
   template<bool isConst>
   class Iterator;

  public:
   CircularBuffer(size_t capacity);
   ~CircularBuffer() = default;

   CircularBuffer(const CircularBuffer&) = default;
   CircularBuffer& operator=(const CircularBuffer&) = default;

   const item_t& front() const;
   const item_t& back() const;

   void push_back(const item_t& item);

   void pop_front(int numToPop=1);
   void pop_back(int numToPop=1);

   item_t& operator[](size_t idx);
   const item_t& operator[](size_t idx) const;

   size_t size() const;
   bool empty() const;

   using iterator = Iterator<false>;
   using const_iterator = Iterator<true>;

   iterator begin();
   iterator end();

   const_iterator begin() const;
   const_iterator end() const;

  private:
   size_t capacity_;
   vector<item_t> arr_;
   int first_;
   int last_;

   void circInc(int& x, int numToInc=1) const;
   void circDec(int& x, int numToDec=1) const;

   template <bool isConst>
   class Iterator {
     public:
      using value_type = item_t;
      using reference = typename std::conditional<isConst, const value_type&, value_type&>::type;
      using pointer = typename std::conditional<isConst, const value_type*, value_type*>::type;
      using difference_type = ptrdiff_t;
      using iterator_category = std::forward_iterator_tag;

      Iterator() = default;
      ~Iterator() = default;
      Iterator(const Iterator&) = default;
      Iterator& operator=(const Iterator&) = default;

      reference operator*() const;
      pointer operator->() const;
      Iterator& operator++();
      bool operator==(const Iterator& rhs) const;
      bool operator!=(const Iterator& rhs) const;

      friend class CircularBuffer<item_t>;

     private:
      using container_ref = typename std::conditional<isConst, const vector<item_t>&, vector<item_t>
&>::type;

      Iterator(container_ref arr, int idx, int last, size_t capacity);

      container_ref arr_;
      int idx_;
      int last_;
      size_t capacity_;
   };


};

#include "CircularBuffer-private.hpp"

#endif
