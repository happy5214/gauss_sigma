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

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "complex.h"
#include "modular.h"

std::ostream& operator<<(std::ostream& out, const complex_t z) {
	out << "(" << z.x << "," << z.y << ")";
	return out;
}

static unsigned long norm_real_imag(const unsigned long real, const unsigned long imag) {
	const unsigned long real2 = real * real;
	const unsigned long imag2 = imag * imag;
	const unsigned long norm = real2 + imag2;
	if (norm < real2) {
		throw std::overflow_error("Norm does not fit an unsigned long.");
	}
	return norm;
}

/* return non-zero if norm(z) <= bound */
bool norm_fits_p(const complex_t z, const unsigned long bound) {
	/* first check if the norm fits an unsigned long */
	const unsigned long x = std::abs(z.x);
	const unsigned long y = std::abs(z.y);
	if (x >= std::numeric_limits<unsigned int>::max() || y >= std::numeric_limits<unsigned int>::max()) {
		return false;
	}
	/* now x, y < 2^32 thus x^2+y^2 < 2^65 */
	try {
		const unsigned long n = norm_real_imag(x, y);
		return n <= bound;
	} catch (std::overflow_error const& ex) {
		// Does not fit an unsigned long
		return false;
	}
}

/* return norm(z), assuming norm(z) <= bound */
unsigned long get_norm(const complex_t z) {
	return norm_real_imag(z.x, z.y);
}

/* Check whether u+iv divides z, i.e., of z * (u-iv)/(u^2+v^2) has
   integer coefficients.
   If so, divides z by u+iv.
   We always have p = u^2+v^2 <= sqrt(norm(z)) <= bound.
   No overflow can occur. */
bool try_divide(complex_t & z, const long u, const long v) {
	if (u == 0 && v == 0) {
		return false;
	}
	assert(u != 0); /* if v*i divides z, also does v, which we prefer */
	const long n = u * u + v * v; /* no overflow since u^2+v^2 = p < 2^32 */
	/* we have |x| <= 2*sqrt(norm(z))*sqrt(p) <= 2*bound^(3/4) < 2^63
	   since bound < 2^64 */
	const long x = z.x * u + z.y * v;
	if ((x % n) != 0) {
		return false;
	}
	const long y = z.x * (-v) + z.y * u; /* same bound for |y| */
	if ((y % n) != 0) {
		return false;
	}
	z.x = x / n;
	z.y = y / n;
	return true;
}

/* return k such that k^2 = -1 mod p for p = 1 mod 4:
   let n >= 2 such that n^((p-1)/2) = -1, then take k = n^((p-1)/4) */
unsigned long square_root_of_minus_one(const unsigned long p) {
	assert((p % 4) == 1);
	for (unsigned long n = 2; n < p; n++) {
		const unsigned long k = powmod(n, (p - 1) / 4, p);
		if (mulmod(k, k, p) == p - 1) {
			return k;
		}
	}
	/* we should never get there */
	std::cerr << "square_root_of_minus_one failed for p=" << p << std::endl;
	std::exit(3);
}
