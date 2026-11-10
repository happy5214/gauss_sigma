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
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <omp.h>

#include "complex.h"
#include "factor.h"
#include "primes.h"

#if DEBUG
std::ostream& operator<<(std::ostream& out, const Factor factor) {
	const auto [z, e] = factor;
	out << z << '^' << e;
	return out;
}

std::ostream& operator<<(std::ostream& out, const factor_t factors) {
	for (const auto factor : factors) {
		out << factor << ' ';
	}
	out << '\n';
	return out;
}
#endif

/* initializes all primes < bound */
GaussianCycleFinder::GaussianCycleFinder(const unsigned long bound, std::unordered_set<complex_t> known_cycles) : known_cycles(known_cycles) {
	primes = sievePrimes(bound);
	const auto nprimes = primes.size();
	std::cout << "found " << nprimes << " primes <= " << bound - 1 << std::endl;
	complex_primes.reserve(nprimes);
#pragma omp parallel for schedule(dynamic,1)
	for (unsigned long i = 0; i < nprimes; i++) {
		const unsigned long p = primes[i];
		if ((p % 4) == 1) {
			/* case p = 1 mod 4:
				(1) first find k such that k^2 = 1 mod p
				(2) then compute the gcd of p and k+I */
			const unsigned long k = square_root_of_minus_one(p);
			const complex_t z = find_divisor(p, k);
			assert(z.x > 0 && z.y >= 0);
			complex_primes[i] = z;
		}
	}
	std::cout << "finished computing complex primes" << std::endl;
}

/* Return x+I*y = unit * gcd(p, k+I) over the Gaussian integers
   with x+I*y in the first quadrant, thus ax > 0 and ay >= 0.
   Assumes p < 2^32 and 0 < k < p. */
complex_t GaussianCycleFinder::find_divisor(const long p, const long k) const {
	long ax = p, ay = 0, bx = k, by = 1, cx, cy, qx, qy, nb;
	while (bx != 0 || by != 0) {
		/* compute a/b = a*conj(b)/b^2 */
		cx = ax * bx + ay * by;
		cy = - ax * by + ay * bx;
		nb = bx * bx + by * by;
		qx = div_round(cx, nb);
		qy = div_round(cy, nb);
		cx = qx * bx - qy * by;
		cy = qx * by + qy * bx;
		cx = ax - cx;
		cy = ay - cy;
		ax = bx; ay = by;
		bx = cx; by = cy;
		}
	if (ax < 0) {
		ax = -ax;
		ay = -ay;
	}
	/* now ax >= 0 */
	if (ay < 0) {
		ay = -ay; /* complex conjugate */
	}
	if (ax == 0) {
		assert(ay > 0);
		/* return -I*z */
		ax = ay;
		ay = 0;
	}
	complex_t z = {ax, ay};
	return z;
}

/* return round(a/b), assuming b > 0 */
long GaussianCycleFinder::div_round(const long a, const long b) const {
	assert(b > 0);
	const long q = a / b;
	const long r = a - q * b;
	if (2 * r > b) { /* r > b/2 */
		return q + 1;
	} else if (2 * r < -b) { /* r < -b/2 */
		return q - 1;
	} else {
		return q;
	}
}
/* we should have x > 0 and y >= 0 */
void GaussianCycleFinder::factor_add(factor_t & factors, const long x, const long y) const {
	assert(x > 0 && y >= 0);
	const complex_t z = {x, y};
	const Factor newFactor = std::make_pair(z, 1);
	factors.push_back(newFactor);
}

/* Assuming p divides n, look for all prime divisors of norm p.
   Assumes p < 2^32. */
unsigned long GaussianCycleFinder::factor_aux(
	factor_t & factors,
	complex_t & z,
	unsigned long norm,
	const unsigned long p,
	const unsigned long idx
) const {
	bool ok;
	if (p == 2) {
		/* from [1] page 17 (step 3c): if e is the 2-valuation of n, then
			(1+k)^e divides z */
		norm /= 2;
		factor_add(factors, 1, 1);
		ok = try_divide(z, 1, 1);
		assert(ok);
		while ((norm % 2) == 0) {
			norm /= 2;
			factors.back().second++;
			ok = try_divide(z, 1, 1);
			assert(ok);
		}
		return norm;
	} else if ((p % 4) == 3) {
		/* if p = 4k+3, then (x,y) = (p,0) and the exponent should
			be even, see [1] top of page 17 */
		assert((norm % (p * p)) == 0);
		norm /= (p * p);
		factor_add(factors, p, 0);
		ok = try_divide(z, p, 0);
		assert(ok);
		while ((norm % (p * p)) == 0) {
			norm /= (p * p);
			factors.back().second++;
			ok = try_divide(z, p, 0);
			assert(ok);
		}
		return norm;
	}
	assert((p % 4) == 1);
	/* we use precomputed values of x,y, with x > 0 and y >= 0 */
	const long x = complex_primes[idx].x;
	const long y = complex_primes[idx].y;
	/* either x+I*y divides z, or its conjugate,
		in fact both can divide, as shown for
		-40-5*I = (-1) * (-I - 2) * (2*I + 1)^2 * (2*I + 3) */
#if DEBUG
	std::cout << "p=" << p << " k=" << k << " x=" << x << " y=" << y << " z=" << z << std::endl;
#endif
	/* we always have norm divisible by p here */
	/* here x^2+y^2 = p, and p < sqrt(bound) */
	if (try_divide(z, x, y)) {
		norm /= p;
		factor_add(factors, x, y);
		while ((norm % p) == 0 && try_divide(z, x, y)) {
			norm /= p;
			factors.back().second++;
		}
	}
	/* now try with I*conj(z) = y+I*x */
	if ((norm % p) == 0 && try_divide(z, y, x)) {
		norm /= p;
		factor_add(factors, y, x);
		while ((norm % p) == 0 && try_divide(z, y, x)) {
			norm /= p;
			factors.back().second++;
		}
	}
	return norm;
}

/* assert z0 <> 0 */
factor_t GaussianCycleFinder::factor(const complex_t z0) const {
	assert(z0.x != 0 || z0.y != 0);
	factor_t factors;
	complex_t z = z0;
	unsigned long norm = get_norm(z);
	if (isPrime(norm)) {
		goto last_factor;
	}
	for (unsigned long i = 0; i < primes.size(); i++) {
		const unsigned long p = primes[i];
		if (p * p > norm) {
			break;
		}
		if ((norm % p) == 0) {
			norm = factor_aux(factors, z, norm, p, i);
			if (isPrime(norm)) {
				break;
			}
		}
	}
	/* if n > 1, then n is prime, thus the only divisor is z */
last_factor:
	if (norm > 1) {
		long x = z.x, y = z.y;
		if (x < 0) {
			x = -x, y = -y;
		}
		if (y < 0) { /* change into -I*z */
			std::swap(x, y);
			x = -x;
		}
		if (x > 0) {
			factor_add(factors, x, y);
		} else {
			factor_add(factors, y, 0);
		}
	}
	return factors;
}

/* z1 <- sigma(z) */
complex_t GaussianCycleFinder::sigma(const complex_t z0) const {
	complex_t z1 = {1, 0};
	const factor_t factors = factor(z0);
#if DEBUG
	std::cout << factors << std::endl;
#endif
	for (auto [t, e] : factors)
		{
		complex_t u = {1, 0};
		long x;
		/* compute 1 + t + ... + t^e */
		while (e-- > 0)
			{
			/* u <- 1 + t*u */
			x = u.x * t.x - u.y * t.y;
			u.y = u.x * t.y + u.y * t.x;
			u.x = 1 + x;
			}
		/* z1 <- z1 * u */
		x = z1.x * u.x - z1.y * u.y;
		z1.y = z1.x * u.y + z1.y * u.x;
		z1.x = x;
		}
	/* subtract z */
	z1.x -= z0.x;
	z1.y -= z0.y;
	return z1;
}

void GaussianCycleFinder::find_cycle(const complex_t z, const unsigned long bound, const unsigned long mod) {
	/* we avoid (0,0) and (1,0) */
	if (z.y == 0 && (z.x == 0 || z.x == 1)) {
		return;
	}
	if (!norm_fits_p(z, bound)) {
		return;
	}
	const unsigned long n0 = get_norm(z);
	assert((n0 % mod) == 0);
	std::vector<complex_t> l;
	l.push_back(z);
	for (int i = 1;; i++) {
		/* we have norm(l[i-1]) <= bound here */
		const complex_t nextZ = sigma(l.back());
		l.push_back(nextZ);
		/* we stop if the first element is not the "smallest" one
		   for some order function C -> R, here we consider the initial
		   element should have the largest x */
		if (nextZ.x > z.x) {
			break;
		}
		/* check loop */
		for (int j = i - 1; j >= 0; j--) {
			if (l[j] == nextZ) {
				const int k = i - j;
				if (j < k || (j >= k && l[j - k] != nextZ)) {
					/* check known loops */
					bool ok;
					{
						const std::shared_lock lock(known_cycles_mutex);
						ok = !known_cycles.contains(nextZ);
					}
					if (ok) {
						std::ofstream cycle_file("gauss_sigma_C_file_cycles", std::ios::out | std::ios::app);
						if (!cycle_file) {
							std::cerr << "Could not open file for writing!" << std::endl;
							std::exit(2);
						} else {
							cycle_file << "loop " << nextZ << ' ' << k << " from " << z << std::endl;
							cycle_file.close();
						}
						std::cout << "loop " << nextZ << ' ' << k << " from " << z << std::endl;
						/* add in known_cycles to avoid duplicate reports */
						const std::unique_lock lock(known_cycles_mutex);
						known_cycles.insert(nextZ);
					}
				}
				/* if a loop is found, we can exit, since no independent loop
				   can be found */
				return;
			}
		}
		/* we stop when reaching 1 */
		if (nextZ.x == 1 && nextZ.y == 0) {
			return;
		}
		/* we also stop when norm(nextZ) does not fit an unsigned long */
		if (!norm_fits_p(nextZ, bound)) {
			return;
		}
	}
}
