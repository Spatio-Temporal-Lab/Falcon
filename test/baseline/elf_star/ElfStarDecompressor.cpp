#include <cstdint>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <iostream>

#include "elf_star.h"
#include "defs.h"
#include "BitStream/BitWriter.h"
#include "BitStream/BitReader.h"
#include "utils/array.h"
#include "utils/post_office_solver.h"

class ElfStarXORDecompressor {
  private:
   DOUBLE storedVal = {.i = 0};
   int storedLeadingZeros = __INT32_MAX__;
   int storedTrailingZeros = __INT32_MAX__;
   bool first = true;
   bool endOfStream = false;
   BitReader reader;
 
   Array<int> leadingRepresentation;
   Array<int> trailingRepresentation;
   int leadingBitsPerValue;
   int trailingBitsPerValue;
 
   int read_int(int length) { return readInt(&reader, length); }
   int read_bit() { return readInt(&reader, 1); }
   uint64_t read_long(int length) { return readLong(&reader, length); }
 
   void initLeadingRepresentation() { 
     int num = read_int(5);
     if (num ==0) {
       num = 32;
     }
     leadingBitsPerValue = PostOfficeSolver::kPositionLength2Bits[num];
     leadingRepresentation = Array<int>(num);
     for (int i = 0; i < num; i++){
       leadingRepresentation[i] = read_int(6);
     }
   }
 
   void initTrailingRepresentation() { 
     int num = read_int(5);
     if (num ==0) {
       num = 32;
     }
     trailingBitsPerValue = PostOfficeSolver::kPositionLength2Bits[num];
     trailingRepresentation = Array<int>(num);
     for (int i = 0; i < num; i++){
       trailingRepresentation[i] = read_int(6);
     }
   }
 
   void next() {
     if (first) {
       initLeadingRepresentation();
       initTrailingRepresentation();
       first = false;
       int trailingZeros = read_int(7);
       if (trailingZeros < 64) {
         storedVal.i = ((read_long(63 - trailingZeros) << 1) + 1) << trailingZeros;
       } else {
         storedVal.i = 0;
       }
       if (isnan(storedVal.d)) {
         endOfStream = true;
       }
     } else {
       nextValue();
     }
   }
 
   void nextValue() {
     DOUBLE value;
     int centerBits;
 
     if (read_bit() == 1) {
       // case 1
       centerBits = 64 - storedLeadingZeros - storedTrailingZeros;
       value.i = read_long(centerBits) << storedTrailingZeros;
       value.i = storedVal.i ^ value.i;
       if (isnan(value.d)){
         endOfStream = true;
       } else {
         storedVal = value;
       }
     } else if (read_bit() == 0) {
       // case  00
       int leadAndTrail = read_int(leadingBitsPerValue + trailingBitsPerValue);
       int lead = leadAndTrail >> trailingBitsPerValue;
       int trail = ~(0xffffffff << trailingBitsPerValue) & leadAndTrail;

       storedLeadingZeros = leadingRepresentation[lead];
       storedTrailingZeros = trailingRepresentation[trail];
       centerBits = 64 - storedLeadingZeros - storedTrailingZeros;
 
       value.i = read_long(centerBits) << storedTrailingZeros;
       value.i = storedVal.i ^ value.i;
       if (isnan(value.d)){
         endOfStream = true;
       } else {
         storedVal = value;
       }
     }
   }
  public:
   size_t length = 0;
 
   void init(uint32_t *in, size_t len) {
     initBitReader(&reader, in + 1, len - 1);
     length = in[0];
   }
 
   double *getValues() {
     double *res = (double *) malloc(sizeof(double) * length);
     for (int i = 0; i < length; i++) {
       res[i] = readValue();
     }
     return res;
   }
 
   BitReader *getReader() {
     return &reader;
   }
 
   double readValue() {
     next();
     if (endOfStream) {
       return -1;
     }
     return storedVal.d;
   }
 
   void refresh() {
     storedVal = {.i = 0};
     storedLeadingZeros = __INT32_MAX__;
     storedTrailingZeros = __INT32_MAX__;
     first = true;
     endOfStream = false;
 
     leadingBitsPerValue = 0;
     trailingBitsPerValue = 0;
     // NOTE: 似乎没有必要 需要考虑这个数组的空间能否自己释放掉
     for (int i = 0; i < leadingRepresentation.length(); i++) {
       leadingRepresentation[i] = 0;
     }
     for (int i = 0; i < trailingRepresentation.length(); i++) {
       trailingRepresentation[i] = 0;
     }
   }
 };

class ElfStarDecompressor {
 private:
  ElfStarXORDecompressor xorDecompressor;
  int lastBetaStar = __INT32_MAX__;

  double nextValue() {
    double v;
    if (read_int(1) == 0) {
      // case 0
      v = recoverVByBetaStar();
    } else if (read_int(1) == 0) {
      // case 10
      v = xorDecompressor.readValue();
    } else {
      // case 11
      lastBetaStar = read_int(4);
      v = recoverVByBetaStar();
    }
    return v;
  }

  double recoverVByBetaStar() {
    double v;
    double vPrime = xorDecompressor.readValue();
    int sp = getSP(abs(vPrime));
    if (lastBetaStar == 0) {
      v = get10iN(-sp - 1);
      if (vPrime < 0) {
        v = -v;
      }
    } else {
      int alpha = lastBetaStar - sp - 1;
      v = roundUp(vPrime, alpha);
    }
    return v;
  }
 protected:
  int read_int(int len) {
    int res = readInt(xorDecompressor.getReader(), len);
    return res;
  }
  int getLength() {
    return xorDecompressor.length;
  }
 public:
  int decompress(double *output) {
    int len = getLength();
    for (int i = 0; i < len; i++) {
      if (i == 4219) {
        asm("nop");
      }
      output[i] = nextValue();
    }
    // 更新，开始新的解压循环
    return len;
  }

  void refresh() { 
    lastBetaStar = __INT32_MAX__;
    xorDecompressor.refresh();
  }

  void init(uint32_t *in, size_t len) { xorDecompressor.init(in, len); }
};

// len data len data 
// len data 

// len = dataset_size =1000000,data1,data2,data3(data1.size = 1024)
//len = dataset_size =1000000,data1 = 1024,datan(100000000-1024*(n-1))

ssize_t elf_star_decode(uint8_t *in, ssize_t len, double *out) {
  ElfStarDecompressor decompressor;
  decompressor.init((uint32_t *) in, len / 4);
  return decompressor.decompress(out);
}