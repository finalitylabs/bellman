__kernel void custom_radix2_fft(__global ulong4* buffer,
                  uint n,
                  uint lgn,
                  ulong4 om,
                  uint lgm) {

  int index = get_global_id(0);

  __global uint256 *elems = buffer;
  uint256 omega = *(uint256*)&om;

  uint works = n >> (lgm + 1);
  uint m = 1 << lgm;

  uint32 k = index >> lgm << (lgm + 1);
  uint32 j = index & (m - 1);
  uint256 w = powmod(omega, (n >> lgm >> 1) * j);

  uint256 t = elems[k+j+m];
  t = mulmod(t, w);
  uint256 tmp = elems[k+j];
  tmp = submod(tmp, t);
  elems[k+j+m] = tmp;
  elems[k+j] = addmod(elems[k+j], t);
}

__kernel void bealto_radix2_fft(__global ulong4* src,
                  __global ulong4* dst,
                  uint n,
                  ulong4 om,
                  uint lgm,
                  uint p) {

  __global uint256 *x = src;
  __global uint256 *y = dst;
  uint256 omega = *(uint256*)&om;

  uint i = get_global_id(0);
  uint t = n >> 1;

  uint k = i & (p - 1);
  x += i;

  uint256 u0;
  u0 = x[0];
  uint256 u1;
  u1 = x[t];

  uint256 twiddle = powmod(omega, (n >> lgm >> 1) * k);
  u1 = mulmod(u1, twiddle);

  uint256 tmp = submod(u0, u1);
  u0 = addmod(u0, u1);
  u1 = tmp;

  uint j = ((i-k)<<1) + k;
  y += j;
  y[0] = u0;
  y[p] = u1;
}

__kernel void bealto_radix4_fft(__global ulong4* src,
                  __global ulong4* dst,
                  uint n,
                  ulong4 om,
                  uint lgm,
                  uint p) {
  __global uint256 *x = src;
  __global uint256 *y = dst;
  uint256 omega = *(uint256*)&om;

  uint i = get_global_id(0);
  uint t = n >> 2; //256
  uint k = i & (p - 1);

  x += i;
  y += ((i-k)<<2) + k;
  
  uint256 u0 = x[0];
  uint256 twiddle = powmod(omega, (n >> (lgm << 1) >> 1) * k);
  uint256 u1 = mulmod(x[t], twiddle);
  uint256 u2 = mulmod(x[2*t], powmod(twiddle, 2));
  uint256 u3 = mulmod(x[3*t], powmod(twiddle, 3));

  // dft2(a,c)
  uint256 tmp = submod(u0, u2);
  u0 = addmod(u0, u2);
  u2 = tmp;

  // dft2(b,d)
  uint256 tmp2 = submod(u1, u3);
  u1 = addmod(u1, u3);
  u3 = tmp2;

  // unsure about this part
  // twiddle d
  u3 = mulmod(u3, twiddle);

  // dft2(a,b)
  uint256 tmp3 = submod(u0, u1);
  u0 = addmod(u0, u1);
  u1 = tmp3;

  // dft2(c,d)
  uint256 tmp4 = submod(u2, u3);
  u2 = addmod(u2, u3);
  u3 = tmp2;  

  y[0] = u0;
  y[p] = u2;
  y[2*p] = u1;
  y[3*p] = u3;
}