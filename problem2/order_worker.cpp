#include "order_worker.h"

void OrderWorker::Run(const std::string & file_path) {
  tm.StartTimer();
  LoadData(file_path);
  // RegisterType();
  tm.EndTimer("handled oneday trans data");
}

void OrderWorker::LoadData(const std::string & file_path) {
  io::CSVReader<12> in(file_path.c_str());
  // in.read_header(io::ignore_extra_column, "ticker", "TranID", "Time", "Price", "Volume", "SaleOrderVolume", "BuyOrderVolume", "Type", "SaleOrderID", "SaleOrderPrice", "BuyOrderID", "BuyOrderPrice");
  while(in.read_row(ticker,TranID,Time,Price,Volume,SaleOrderVolume,BuyOrderVolume,Type,SaleOrderID,SaleOrderPrice,BuyOrderID,BuyOrderPrice)){
    // printf("ticker=%s, transid=%s, time=%s, price=%lf, vol=%d, saleordervol=%d, buyordervol=%d, type=%s, saleorderid=%s, saleorderprice=%lf, buyorderid=%s, buyorderprice=%lf\n", ticker.c_str(), TranID.c_str(), Time.c_str(), Price, Volume, SaleOrderVolume, BuyOrderVolume, Type.c_str(), SaleOrderID.c_str(), SaleOrderPrice, BuyOrderID.c_str(), BuyOrderPrice);
    std::string buy_id = ticker + BuyOrderID;
    std::string sell_id = ticker + SaleOrderID;
    if (sell_turnover_map_.find(sell_id) == sell_turnover_map_.end()) {
      sell_turnover_map_[sell_id] = 0.0;
    }
    if (buy_turnover_map_.find(buy_id) == buy_turnover_map_.end()) {
      buy_turnover_map_[buy_id] = 0.0;
    }
    // if (Type == "S") {
      sell_turnover_map_[sell_id] += Price * Volume;
      ticker_sell_orderid_map_[ticker].insert(sell_id);
    // } else if (Type == "B") {
      buy_turnover_map_[buy_id] += Price * Volume;
      ticker_buy_orderid_map_[ticker].insert(buy_id);
    // } else {
      // continue;
    // }
    ticker_set_.insert(ticker);
  }
}

void OrderWorker::RegisterType() {
  for (auto m : buy_turnover_map_) {
    int t = m.second;
    std::string label="";
    if (t <= 50000) {
      label = "Small";
    } else if (t <= 200000) {
      label = "Middle";
    } else if (t <= 1000000) {
      label = "Large";
    } else {
      label = "Exlarge";
    }
    buy_type_count_map_[label] += 1;
    printf("buyorderid %s label: %s\n", m.first.c_str(), label.c_str());
  }
  for (auto m : sell_turnover_map_) {
    int t = m.second;
    std::string label="";
    if (t <= 50000) {
      label = "Small";
    } else if (t <= 200000) {
      label = "Middle";
    } else if (t <= 1000000) {
      label = "Large";
    } else {
      label = "Exlarge";
    }
    sell_type_count_map_[label] += 1;
    printf("sellorderid %s label: %s\n", m.first.c_str(), label.c_str());
  }
}

void OrderWorker::GenReport() {
  for (auto m : ticker_set_) {
    std::string ticker = m;
    printf("[%s]", ticker.c_str());
    auto buy_set = ticker_buy_orderid_map_[ticker];
    auto sell_set = ticker_sell_orderid_map_[ticker];
    int small=0, middle=0, large=0, exlarge = 0;
    for (auto orderid : buy_set) {
      double t = buy_turnover_map_[orderid];
      (t <= 50000) ? small++ : (t <= 200000) ? middle++ : (t <= 1000000) ? large++ : exlarge++;
      // printf("%s=%lf\n", orderid.c_str(), t);
    }
    printf("BUY:Small=%d, Middle=%d, large=%d, exlarge=%d;", small, middle, large, exlarge);
    small=0, middle=0, large=0, exlarge = 0;
    for (auto orderid : sell_set) {
      double t = sell_turnover_map_[orderid];
      (t <= 50000) ? small++ : (t <= 200000) ? middle++ : (t <= 1000000) ? large++ : exlarge++;
      // printf("%s=%lf\n", orderid.c_str(), t);
    }
    printf("SELL:Small=%d, Middle=%d, large=%d, exlarge=%d\n", small, middle, large, exlarge);
  }
}
