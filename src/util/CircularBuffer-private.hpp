template <typename item_t>
CircularBuffer<item_t>::CircularBuffer(size_t capacity) :
   capacity_{capacity},
   first_{-1}, // There is no first or last item yet
   last_{-1} {
}

template <typename item_t>
const item_t& CircularBuffer<item_t>::front() const {
   return arr_[first_];
}

template <typename item_t>
const item_t& CircularBuffer<item_t>::back() const {
   return arr_[last_];
}

template <typename item_t>
void CircularBuffer<item_t>::push_back(const item_t& item) {
   if (empty()) {
      first_ = 0;
      last_ = 0;
   } else {
      circInc(last_);
      if (first_ == last_) {
	 circInc(first_);
      }
   }

   if (arr_.size() < capacity_ and last_ == int(arr_.size())) {
      arr_.push_back(item);
      if (arr_.size() == capacity_) {
	 arr_.shrink_to_fit();
      }
   } else {  
      arr_[last_] = item;
   }
}

template <typename item_t>
void CircularBuffer<item_t>::pop_front(int numToPop) {
   if (numToPop >= size()) { // list is now empty
      first_ = -1;
      last_ = -1;
   } else {
      circInc(first_, numToPop);
   }
}

template <typename item_t>
void CircularBuffer<item_t>::pop_back(int numToPop) {
   if (numToPop >= (int)size()) { // list is now empty
      first_ = -1;
      last_ = -1;
   } else {
      circDec(last_, numToPop);
   }
}

template <typename item_t>
void CircularBuffer<item_t>::circInc(int& x, int numToInc) const {
   x += numToInc;
   if ((size_t)x >= capacity_) {
      x -= capacity_;
   }
}

template <typename item_t>
void CircularBuffer<item_t>::circDec(int& x, int numToDec) const {
   x -= numToDec;
   if (x < 0) {
      x += capacity_;
   }   
}

template <typename item_t>
item_t& CircularBuffer<item_t>::operator[](size_t idx) {
   size_t circIdx = idx + first_;
   if (circIdx >= capacity_) {
      circIdx -= capacity_;
   }

   return arr_[circIdx];
}

template <typename item_t>
const item_t& CircularBuffer<item_t>::operator[](size_t idx) const {
   size_t circIdx = idx + first_;
   if (circIdx >= capacity_) {
      circIdx -= capacity_;
   }

   return arr_[circIdx];
}

template <typename item_t>
size_t CircularBuffer<item_t>::size() const {
   if (first_ < 0) {
      return 0;
   } else {
      int size = last_ - first_;
      if (size < 0) {
	 size += capacity_;
      }
      ++size;

      return size;
   }
}   

template <typename item_t>
bool CircularBuffer<item_t>::empty() const {
   return (first_ < 0);
}

template <typename item_t>
typename CircularBuffer<item_t>::iterator CircularBuffer<item_t>::begin() {
   return {arr_, first_, last_, capacity_};
}

template <typename item_t>
typename CircularBuffer<item_t>::iterator CircularBuffer<item_t>::end() {
   return {arr_, -1, last_, capacity_};
}

template <typename item_t>
typename CircularBuffer<item_t>::const_iterator CircularBuffer<item_t>::begin() const {
   return {arr_, first_, last_, capacity_};
}

template <typename item_t>
typename CircularBuffer<item_t>::const_iterator CircularBuffer<item_t>::end() const {
   return {arr_, -1, last_, capacity_};
}

template <typename item_t>
template <bool isConst>
CircularBuffer<item_t>::Iterator<isConst>::Iterator(container_ref arr, int idx, int last, size_t capacity) :
   arr_{arr},
   idx_{idx},
   last_{last},
   capacity_{capacity} {}

template <typename item_t>
template <bool isConst>
typename CircularBuffer<item_t>::template Iterator<isConst>::reference CircularBuffer<item_t>::Iterator<isConst>::operator*() const {
   return arr_[idx_];
}

template <typename item_t>
template <bool isConst>
typename CircularBuffer<item_t>::template Iterator<isConst>::pointer CircularBuffer<item_t>::Iterator<isConst>::operator->() const {
   return &**this; // A pointer to the item at the position of *this
}

template <typename item_t>
template <bool isConst>
typename CircularBuffer<item_t>::template Iterator<isConst>& CircularBuffer<item_t>::Iterator<isConst>::operator++() {
   if (idx_ == last_) {
      idx_ = -1;
   } else {
      ++idx_;
      if ((size_t)idx_ >= capacity_) {
	 idx_ -= capacity_;
      }
   }

   return *this;
}

template <typename item_t>
template <bool isConst>
bool CircularBuffer<item_t>::Iterator<isConst>::operator==(const Iterator& rhs) const {
   return idx_ == rhs.idx_;      
}

template <typename item_t>
template <bool isConst>
bool CircularBuffer<item_t>::Iterator<isConst>::operator!=(const Iterator& rhs) const {
   return !(*this == rhs);
}
