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

#ifndef COMPLEX_H
#define COMPLEX_H 1

#include <iostream>

typedef struct {
	long x;
	long y;
} complex_t;

std::ostream& operator<<(std::ostream& out, complex_t z);

/* return non-zero if norm(z) <= bound */
bool norm_fits_p(complex_t z, unsigned long bound);

/* return norm(z), assuming norm(z) <= bound */
unsigned long get_norm(complex_t z);

/* Check whether u+iv divides z, i.e., of z * (u-iv)/(u^2+v^2) has
   integer coefficients.
   If so, divides z by u+iv.
   We always have p = u^2+v^2 <= sqrt(norm(z)) <= bound.
   No overflow can occur. */
bool try_divide(complex_t & z, long u, long v);

/* return k such that k^2 = -1 mod p for p = 1 mod 4:
   let n >= 2 such that n^((p-1)/2) = -1, then take k = n^((p-1)/4) */
unsigned long square_root_of_minus_one(unsigned long p);

#endif // COMPLEX_H
