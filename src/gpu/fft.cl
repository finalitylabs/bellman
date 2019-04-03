typedef unsigned int uint32;
typedef unsigned long long uint64;

typedef struct {
  uint32 val[8];
} uint256;

__constant uint256 P = {0x00000001,0xffffffff,0xfffe5bfe,0x53bda402,
                        0x09a1d805,0x3339d808,0x299d7d48,0x73eda753};

__constant uint256 ONE = {0x00000001,0x00000000,0x00000000,0x00000000,
                          0x00000000,0x00000000,0x00000000,0x00000000};

__constant uint256 GEN = {0x00000007,0x00000000,0x00000000,0x00000000,
                          0x00000000,0x00000000,0x00000000,0x00000000};

__constant uint256 R2 = {0xf3f29c6d,0xc999e990,0x87925c23,0x2b6cedcb,
                         0x7254398f,0x05d31496,0x9f59ff11,0x0748d9d9};

__constant uint32 P0INV = 4294967295;


__constant uint32 S = 32;

// 2^32th root of unity
__constant uint256 ROOT = {0x439f0d2b,0x3829971f,0x8c2280b9,0xb6368350,
                           0x22c813b4,0xd09b6819,0xdfe81f20,0x16a2a19e};

uint256 create(uint32 v) {
  uint256 ret = {v,0,0,0,0,0,0,0};
  return ret;
}

void add_digit(uint32 *res, uint32 index, uint32 num) {
  while(true) {
    uint32 old = res[index];
    res[index] += num;
    if(res[index] < old) {
      num = 1;
      index++;
    } else break;
  }
}

void sub_digit(uint32 *res, uint32 index, uint32 num) {
  while(true) {
    uint32 old = res[index];
    res[index] -= num;
    if(res[index] > old) {
      num = 1;
      index++;
    } else break;
  }
}

bool gte(uint256 a, uint256 b) {
  for(int i = 7; i >= 0; i--){
    if(a.val[i] > b.val[i])
      return true;
    if(a.val[i] < b.val[i])
      return false;
  }
  return true;
}

uint256 add(uint256 a, uint256 b) {
  for(int i = 0; i < 8; i++)
    add_digit(a.val, i, b.val[i]);
  return a;
}

uint256 sub(uint256 a, uint256 b) {
  for(int i = 0; i < 8; i++)
    sub_digit(a.val, i, b.val[i]);
  return a;
}

uint256 mul_reduce(uint256 a, uint256 b) {
  uint32 res[16] = {0};
  for(int i = 0; i < 8; i++) {
    for(int j = 0; j < 8; j++) {
      uint64 total = (uint64)a.val[i] * (uint64)b.val[j];
      uint32 lo = total & 0xffffffff;
      uint32 hi = total >> 32;
      add_digit(res, i + j, lo);
      add_digit(res, i + j + 1, hi);
    }
  }
  for (int i = 0; i < 8; i++)
  {
    uint64 u = ((uint64)P0INV * (uint64)res[i]) & 0xffffffff;
    for(int j = 0; j < 8; j++) {
      uint64 total = u * (uint64)P.val[j];
      uint32 lo = total & 0xffffffff;
      uint32 hi = total >> 32;
      add_digit(res, i + j, lo);
      add_digit(res, i + j + 1, hi);
    }
  }
  uint256 result;
  for(int i = 0; i < 8; i++) result.val[i] = res[i+8];
  if(gte(result, P))
    result = sub(result, P);
  return result;
}

uint256 mulmod(uint256 a, uint256 b) {
  return mul_reduce(mul_reduce(mul_reduce(a, R2), mul_reduce(b, R2)), ONE);
}

uint256 negmod(uint256 a) {
  return sub(P, a);
}

uint256 submod(uint256 a, uint256 b) {
  uint256 res = sub(a, b);
  if(!gte(a, b)) res = add(res, P);
  return res;
}

uint256 addmod(uint256 a, uint256 b) {
  return submod(a, negmod(b));
}

uint256 powmod(uint256 b, uint64 p) {
  if (p == 0)
    return ONE;
  uint256 t = powmod(b, p / 2);
  t = mulmod(t, t);
  if (p % 2 == 0)
    return t;
  else
    return mulmod(b, t);
}

// FFT

uint32 bitreverse(uint32 n, uint32 bits) {
  uint32 r = 0;
  for(int i = 0; i < bits; i++) {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  return r;
}

void swap(uint256 *a, uint256 *b) {
  uint256 tmp = *a;
  *a = *b;
  *b = tmp;
}

void FFT(uint256 *elems, uint32 n, uint32 lg, uint256 omega) {

  for(uint32 k = 0; k < n; k++) {
    uint32 rk = bitreverse(k, lg);
    if(k < rk)
      swap(&elems[k], &elems[rk]);
  }

  uint32 m = 1;
  for(int i = 0; i < lg; i++) {
    uint256 w_m = powmod(omega, n / (2*m));
    uint32 k = 0;
    while(k < n) {
      uint256 w = ONE;
      for(int j = 0; j < m; j++) {
        uint256 t = elems[k+j+m];
        t = mulmod(t, w);
        uint256 tmp = elems[k+j];
        tmp = submod(tmp, t);
        elems[k+j+m] = tmp;
        elems[k+j] = addmod(elems[k+j], t);
        w = mulmod(w, w_m);
      }
      k += 2*m;
    }
    m *= 2;
  }
}

__kernel void fft(__global int* buffer) {
  buffer[get_global_id(0)] *= 5;
}
