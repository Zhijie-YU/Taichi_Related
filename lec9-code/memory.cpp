// g++ memory.cpp -o memory -std=c++17 -O3 -Wall && ./memory 4

#include <sys/time.h>
#include <iostream>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

constexpr int n = 256 * 1024 * 1024;
int a[n];

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: ./memory [stride]\n"); 
    return -1;
  }
  int stride = std::atoi(argv[1]);
  for (int K = 0; K < 32; K++) {
    auto t = get_time();
    for (int i = 0; i < n; i += stride) {
      a[i] = i;
    }
    printf("%f\n", get_time() - t);
  }
  return 0;
}
