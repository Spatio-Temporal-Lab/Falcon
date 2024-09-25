// easy coding benchmark
#include <cstdio>
#include <iostream>
#include <ostream>

#include "Serf/test/baselines/gorilla/gorilla_compressor.h"
#include "Serf/test/baselines/gorilla/gorilla_decompressor.h"

int main() {
    GorillaCompressor compressor(10001);
    GorillaDecompressor decompressor;
    compressor.addValue(3.14);
    compressor.addValue(3.15);
    compressor.close();
    auto pack = compressor.get_compress_pack();
    auto decompressed = decompressor.decompress(pack);
    for (double decompressed1 : decompressed) {
        std::cout << decompressed1 << std::endl;
    }
}