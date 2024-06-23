#include "Params.hpp"
#include "dout.hpp"
#include <iostream>

using namespace std;

void Params::setFloat(const string& key, double f) {
   floatMap[key] = f;
}

void Params::setInt(const string& key, long long i) {
   intMap[key] = i;
}

void Params::setStr(const string& key, const string& s) {
   strMap[key] = s;
}

template <class ValT>
const ValT& Params::get(const map<string, ValT>& map, const string& key) const {
   auto iter = map.find(key);
   if (iter != map.end()) {
      return iter->second;
   } else {
      cerr << "Key " << key << " not found in params!";
      exit(1);
   }
}

double Params::getFloat(const string& key) const {
   return get(floatMap, key);
}

long long Params::getInt(const string& key) const {
   return get(intMap, key);
}

const string& Params::getStr(const string& key) const {
   return get(strMap, key);
}

bool Params::isSet(const string& key) const {
   return strMap.find(key) != strMap.end() or
          floatMap.find(key) != floatMap.end() or
          intMap.find(key) != intMap.end();   
}
