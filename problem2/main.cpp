#include <vector>
#include <string>
#include "order_worker.h"

int main() {
  std::string file_path = "../data/trans";
  OrderWorker ow;
  ow.Run(file_path);
  ow.GenReport();
}
