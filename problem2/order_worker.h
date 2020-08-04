#include "../vendor/csv.h"
#include "../vendor/timer.hpp"
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>

class OrderWorker {
 public:
  OrderWorker() = default;
  virtual ~OrderWorker() = default;
  void Run(const std::string & file_path);
  void GenReport();

 private:
  void LoadData(const std::string & file_path);
  void RegisterType();
  Timer tm;

  std::unordered_set<std::string> ticker_set_;
  std::unordered_map<std::string, std::unordered_set<std::string> > ticker_buy_orderid_map_;
  std::unordered_map<std::string, std::unordered_set<std::string> > ticker_sell_orderid_map_;

  std::unordered_map<std::string, double> buy_turnover_map_;
  std::unordered_map<std::string, double> sell_turnover_map_;

  std::unordered_map<std::string, int> buy_type_count_map_ = {{"Small", 0}, {"Middle", 0}, {"Large", 0}, {"Exlarge", 0}};
  std::unordered_map<std::string, int> sell_type_count_map_ = {{"Small", 0}, {"Middle", 0}, {"Large", 0}, {"Exlarge", 0}};

  std::string ticker, TranID, Time, SaleOrderID, BuyOrderID, Type;
  double Price, SaleOrderPrice, BuyOrderPrice;
  int Volume, SaleOrderVolume, BuyOrderVolume;
};

