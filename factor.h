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

#ifndef FACTOR_H
#define FACTOR_H 1

#include <shared_mutex>
#include <unordered_set>
#include <utility>
#include <vector>

#include "complex.h"

typedef std::pair<complex_t, int> Factor;
typedef std::vector<Factor> factor_t;

#if DEBUG
#include <iostream>
std::ostream& operator<<(std::ostream& out, Factor f);

std::ostream& operator<<(std::ostream& out, factor_t factors);
#endif

class GaussianCycleFinder {
public:
	GaussianCycleFinder(unsigned long bound, std::unordered_set<complex_t> known_cycles);

	void find_cycle(const complex_t z, const unsigned long bound, const unsigned long mod);

private:
	std::vector<unsigned long> primes;
	std::vector<complex_t> complex_primes;
	std::unordered_set<complex_t> known_cycles;
	std::shared_mutex known_cycles_mutex;

	long div_round(long a, long b) const;
	complex_t find_divisor(long p, long k) const;

	complex_t sigma(complex_t z0) const;
	factor_t factor(complex_t z0) const;
	void factor_add(factor_t & factors, long x, long y) const;
	unsigned long factor_aux(
		factor_t & factors,
		complex_t & z,
		unsigned long norm,
		unsigned long p,
		unsigned long idx
	) const;
};

#endif // FACTOR_H
