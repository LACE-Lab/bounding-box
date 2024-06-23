#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <string>
#include <map>

class Params
{
  public:
   Params() = default;
   ~Params() = default;

   void setFloat(const std::string& key, double f);
   void setInt(const std::string& key, long long i);
   void setStr(const std::string& key, const std::string& s);

   double getFloat(const std::string& key) const;
   long long getInt(const std::string& key) const;
   const std::string& getStr(const std::string& key) const;

   bool isSet(const std::string& key) const;   
   
  private:
   std::map<std::string, double> floatMap;
   std::map<std::string, long long> intMap;
   std::map<std::string, std::string> strMap;

   template <class ValT>
   const ValT& get(const std::map<std::string, ValT>& map, const std::string& key) const;
};

#endif
