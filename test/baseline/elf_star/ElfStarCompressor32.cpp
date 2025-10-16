 #include <cstdint>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <bits/stdc++.h>

#include "elf_star.h"
#include "defs.h"
#include "BitStream/BitWriter.h"
#include "BitStream/BitReader.h"
#include "utils/post_office_solver_32.h"
#include "utils/output_bit_stream.h"
#include "utils/array.h"
using namespace std;

class ElfStarXORCompressor32 {
  private:
    Array<int> leading_representation_ = Array<int>(32);
    Array<int> trailing_representation_ = Array<int>(32);
    Array<int> leading_round_ = Array<int>(32);
    Array<int> trailing_round_ = Array<int>(32);
    int storedLeadingZeros = __INT32_MAX__;
    int storedTrailingZeros = __INT32_MAX__;
    uint32_t storedVal = 0;
    bool first = true;
    Array<int> lead_distribution_ = Array<int>(32);
    Array<int> trail_distribution_ = Array<int>(32);
    int leading_bits_per_value_;
    int trailing_bits_per_value_;

    size_t length = 0;
    BitWriter writer;
    uint32_t *output;

    int initLeadingRoundAndRepresentation(Array<int> lead_distribution_){
      Array<int> lead_positions = PostOfficeSolver32::InitRoundAndRepresentation(
          lead_distribution_, leading_representation_, leading_round_);
      leading_bits_per_value_ =
          PostOfficeSolver32::kPositionLength2Bits[lead_positions.length()];
      return PostOfficeSolver32::WritePositions(lead_positions, &writer);
    }

    int initTrailingRoundAndRepresentation(Array<int> trail_distribution_){
      Array<int> trail_positions = PostOfficeSolver32::InitRoundAndRepresentation(trail_distribution_,
        trailing_representation_,
        trailing_round_);
        trailing_bits_per_value_ =
          PostOfficeSolver32::kPositionLength2Bits[trail_positions.length()];
      return PostOfficeSolver32::WritePositions(trail_positions, &writer);
    }

    int writeFirst(int value) {
      first = false;
      storedVal = value;
      int trailingZeros = __builtin_ctz(value);
      write(&writer, trailingZeros, 6);
      // if (trailingZeros < 64) { optimized-out somehow, assuming __builtin_ctzl
      // always < 64 ?
      if (value != 0) {
        writeLong(&writer, storedVal >> (trailingZeros + 1), 31 - trailingZeros);
        // size += 70 - trailingZeros;
        return 37 - trailingZeros;
      } else {
        // size += 7;
        return 6;
      }
    }

    int compressValue(int value) {
      int thisSize = 0;
      uint32_t _xor = storedVal ^ value;
      if (_xor == 0) {
        write(&writer, 1, 2);
        thisSize += 2;
      } else {
        int leading_count = __builtin_clz(_xor);
        int trailing_count = __builtin_ctz(_xor);

        int leadingZeros = leading_round_[leading_count];
        int trailingZeros = trailing_round_[trailing_count];
        // int leadingZeros = leading_round_[__builtin_clzl(_xor)];
        // int trailingZeros = trailing_round_[__builtin_ctzl(_xor)];

        if (leadingZeros >= storedLeadingZeros && trailingZeros >= storedTrailingZeros &&
          (leadingZeros - storedLeadingZeros) + (trailingZeros - storedTrailingZeros) < 1 + leading_bits_per_value_ + trailing_bits_per_value_) {
          // case 1
          int centerBits = 32 - storedLeadingZeros - storedTrailingZeros;
          int len = 1 + centerBits;
          if (len > 32) {
            write(&writer, 1, 1);
            writeLong(&writer, _xor >> storedTrailingZeros, centerBits);
          } else {
            writeLong(&writer, (1L << centerBits) | (_xor >> storedTrailingZeros), 1 + centerBits);
          }

          thisSize += len;
        } else {
          storedLeadingZeros = leadingZeros;
          storedTrailingZeros = trailingZeros;
          int centerBits = 32 - storedLeadingZeros - storedTrailingZeros;

          // case 00
          int len = 2 + leading_bits_per_value_ + trailing_bits_per_value_ +
                    centerBits;
          if(len > 32) {
            write(&writer,
                  (leading_representation_[storedLeadingZeros]
                   << trailing_bits_per_value_) |
                      trailing_representation_[storedTrailingZeros],
                  2 + leading_bits_per_value_ + trailing_bits_per_value_);
            writeLong(&writer, _xor >> storedTrailingZeros, centerBits);
          } else {
            long tmp = ((((uint32_t)leading_representation_[storedLeadingZeros]
                          << trailing_bits_per_value_) |
                         trailing_representation_[storedTrailingZeros])
                        << centerBits) |
                       (_xor >> storedTrailingZeros);
            writeLong(&writer,
                      ((((uint32_t)leading_representation_[storedLeadingZeros]
                         << trailing_bits_per_value_) |
                        trailing_representation_[storedTrailingZeros])
                       << centerBits) |
                          (_xor >> storedTrailingZeros),
                      len);
          }
          thisSize += len;
        }
        storedVal = value;
      }
      return thisSize;
    }

  public:
    BitWriter *getWriter() {
      return &writer;
    }

    void init(size_t length) {
      length *= 12;
      output = (uint32_t *) malloc(length + 4);
      initBitWriter(&writer, output + 1, length / sizeof(uint32_t));
    }

    int addValue(int value) {
      if (first) {
        return initLeadingRoundAndRepresentation(lead_distribution_) +
               initTrailingRoundAndRepresentation(trail_distribution_) +
               writeFirst(value);
      } else {
        return compressValue(value);
      }
    }

    // int addValue(float value) {
    //   DOUBLE data = {.d = value};
    //   if (first) {
    //     return initLeadingRoundAndRepresentation(lead_distribution_) +
    //            initTrailingRoundAndRepresentation(trail_distribution_) +
    //            writeFirst(value);
    //   } else {
    //     return compressValue(value);
    //   }
    // }

    void close() {
      // *output = length;
      flush(&writer);
    }

    // size_t getSize() {
    //   return size;
    // }

    uint32_t *getOut() {
      return output;
    }

    void setDistribution(Array<int> lead_distribution, Array<int> trail_distribution) {
      this->lead_distribution_ = lead_distribution;
      this->trail_distribution_ = trail_distribution;
    }

    void refresh(){
      output = (uint32_t *) malloc(length + 4);
      storedLeadingZeros = __INT32_MAX__;
      storedTrailingZeros = __INT32_MAX__;
      storedVal = 0;
      first = true;
      for (int i = 0; i < 32; i++){
        leading_representation_[i] = 0;
        trailing_representation_[i] = 0;
        leading_round_[i] = 0;
        trailing_round_[i] = 0;
      } 
    }
};

// template<ssize_t N>
class ElfStarCompressor32 {
  private:
    size_t size = 32;
    int lastBetaStar = __INT32_MAX__;
    int numberOfValues = 0;
    ElfStarXORCompressor32 xorCompressor;
    // int betaStarList[N];
    // uint64_t vPrimeList[N];
    int* betaStarList;
    uint32_t* vPrimeList;
    Array<int> leadDistribution = Array<int>(32);
    Array<int> trailDistribution = Array<int>(32);

   protected:
    int writeInt(int n, int len){
        write(xorCompressor.getWriter(), n, len);
        return len;
      }

    int writeBit(bool bit){
      write(xorCompressor.getWriter(), bit, 1);
      return 1;
    }

    int xorCompress(int vPrimeInt){
      return xorCompressor.addValue(vPrimeInt);
    }

  public:
    void addValue(float v) {
      FLOAT data = {.f = v};
      int vPrimeInt;
      if (v == 0.0 or isinf(v)) {
        vPrimeList[numberOfValues] = data.i;
        betaStarList[numberOfValues] = __INT32_MAX__;
      } else if (isnan(v)){
        vPrimeList[numberOfValues] = 0x7fc00000 & data.i;
        betaStarList[numberOfValues] = __INT32_MAX__;
      } else {
        int *alphaAndBetaStar = getAlphaAndBetaStar_32(v, lastBetaStar);
        int e = ((int) (data.i >> 23)) & 0xff;
        int gAlpha = getFAlpha(alphaAndBetaStar[0]) + e - 127;
        int eraseBits = 23 - gAlpha;
        long mask = 0xffffffffL << eraseBits;
        long delta = (~mask) & data.i;
        if (delta != 0 && eraseBits > 3) {
          if (alphaAndBetaStar[1] != lastBetaStar) {
            lastBetaStar = alphaAndBetaStar[1];
          }
          betaStarList[numberOfValues] = lastBetaStar;
          vPrimeList[numberOfValues] = mask & data.i;
        } else {
          betaStarList[numberOfValues] = __INT32_MAX__;
          vPrimeList[numberOfValues] = data.i;
        }

        delete[] alphaAndBetaStar;
      }
      numberOfValues++;
    }

    void calculateDistribution() { 
      uint32_t lastValue = vPrimeList[0];
      for (int i = 1; i < numberOfValues; i++){
        uint32_t xor_ = lastValue ^ vPrimeList[i];
        if (xor_ != 0){
          leadDistribution[__builtin_clz(xor_)]++;
          trailDistribution[__builtin_ctz(xor_)]++;
          lastValue = vPrimeList[i];
        }
      }
    }

    void compress() {
      xorCompressor.setDistribution(leadDistribution, trailDistribution);
      lastBetaStar = __INT32_MAX__;
      for (int i = 0; i < numberOfValues; i++){
        if (betaStarList[i] == __INT32_MAX__) {
          // case 10
          size += writeInt(2, 2);
        } else if (betaStarList[i] == lastBetaStar) {
          // case 0
          size += writeBit(false);
        } else {
          // case 11, 2 + 3 = 5
          size += writeInt(betaStarList[i] | 0x18, 5);
          lastBetaStar = betaStarList[i];
        }
        size += xorCompressor.addValue(vPrimeList[i]);
      }
    }

    double getRatio() { return size / (numberOfValues * 32.0); }

    int getSize() { return size; }

    void init(int length) {
      xorCompressor.init(length);
      betaStarList = new int[length];
      vPrimeList = new uint32_t[length];
    }

    uint32_t *getBytes() {
      return xorCompressor.getOut();
    }
  
    void close() {
      calculateDistribution();
      compress();
      // we write one more bit here, for marking an end of the stream.
      // size += writeInt(2, 2);
      xorCompressor.close();
      // 这一步的size已在初始化时预留32bit
      *getBytes() = numberOfValues;
      delete[] betaStarList;
      delete[] vPrimeList;
    }

    void refresh() { 
      xorCompressor.refresh();
      size = 0;
      lastBetaStar = __INT32_MAX__;
      numberOfValues = 0;
      leadDistribution = Array<int>(32);
      trailDistribution = Array<int>(32);
    }
};

ssize_t elf_star_encode_32(float *in, ssize_t len, uint8_t **out) {
  ElfStarCompressor32 compressor;
  compressor.init(len);//bitwrite
  for (int i = 0; i < len; i++) {
    if (i == 4219) {
      asm("nop");
    }
    compressor.addValue(in[i]);//够了blocksize close()
  }
  compressor.close();
  *out = (uint8_t *) compressor.getBytes();
  // std::cout << "ratio: " << compressor.getRatio() << std::endl;
  return (compressor.getSize() + 31) / 32 * 4;
}