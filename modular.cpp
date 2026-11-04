/* Copyright (C) 2019 Paul Zimmermann, LORIA/INRIA
 * Copyright (C) 2026 Alexander Jones
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; see the file COPYING.  If not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02111-1307, USA.
 */

#include <limits>
#include <version>

#include "modular.h"

#if __cpp_lib_int_pow2 >= 202002L

#include <bit>
#define nbits std::bit_width

#else

constexpr static int _nbits(unsigned long p) {
	int bit_count = 0;
	while (p) {
		bit_count++;
		p >>= 1;
	}
	return bit_count;
}

#define nbits _nbits

#endif

unsigned long mulmod(const unsigned long a, const unsigned long b, const unsigned long modulus) {
	if (modulus <= std::numeric_limits<unsigned int>::max()) {
		return (a * b) % modulus;
	} else {
		const __uint128_t product = (__uint128_t) a * (__uint128_t) b;
		return product % modulus;
	}
}

/* powmod = b^n mod m */
unsigned long powmod(const unsigned long base, const unsigned long exponent, const unsigned long modulus) {
	if (exponent == 0) {
		return 1;
	} else if (exponent == 1) {
		return base % modulus;
	}
	int bit_count = nbits(exponent) - 1;
	unsigned long result = base > modulus ? base % modulus : base;
	while (bit_count--) {
		result = mulmod(result, result, modulus);
		if ((exponent >> bit_count) & 1) {
			result = mulmod(result, base, modulus);
		}
	}
	return result;
}
