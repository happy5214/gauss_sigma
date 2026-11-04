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

#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "modular.h"
#include "primes.h"

std::vector<unsigned long> sievePrimes(const unsigned long pMax) {
	std::vector<unsigned long> primes;
	std::vector<bool> numbers(pMax, true);

	// Sieve
	for (unsigned long i = 0; i < pMax - 1; i++) {
		if (numbers[i]) {
			for (unsigned long j = 2 * i + 2; j < pMax; j += (i + 2)) {
				numbers[j] = false;
			}
			primes.push_back(i + 2);
		}
	}
	return primes;
}

/* https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test */
bool millerRabinTest(const unsigned long n, const unsigned long a) {
	unsigned long d = n - 1, s = 1;
	while ((d % 2) == 0) {
		d = d / 2, s++;
	}
	/* now n - 1 = 2 ^ s * d with d odd */
	unsigned long b = powmod(a, d, n);
	if (b == 1) {
		return true; /* other powers will be 1 too */
	}
	if (b == n - 1) {
		return true;
	}
	for (unsigned long r = 1; r < s; r++) {
		b = mulmod(b, b, n);
		if (b == n - 1) {
			return true;
		}
	}
	return false;
}

/* https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test */
bool isPrime(const unsigned long n) {
	if (n <= 2) {
		return n == 2;
	} else if ((n % 2) == 0) {
		return false;
	}

	/* now n is odd and n >= 3 */
	if (n < 341531UL) {
		return millerRabinTest(n, 9345883071009581737UL);
	} else if (n < 1050535501UL) {
		return millerRabinTest(n, 336781006125UL) &&
			millerRabinTest(n, 9639812373923155UL);
	} else if (n < 350269456337UL) {
		return millerRabinTest(n, 4230279247111683200UL) &&
			millerRabinTest(n, 14694767155120705706UL) &&
			millerRabinTest(n, 16641139526367750375UL);
	} else if (n < 55245642489451UL) {
		return millerRabinTest(n, 2UL) &&
			millerRabinTest(n, 141889084524735UL) &&
			millerRabinTest(n, 1199124725622454117UL) &&
			millerRabinTest(n, 11096072698276303650UL);
	} else if (n < 7999252175582851UL) {
		return millerRabinTest(n, 2UL) &&
			millerRabinTest(n, 4130806001517UL) &&
			millerRabinTest(n, 149795463772692060UL) &&
			millerRabinTest(n, 186635894390467037UL) &&
			millerRabinTest(n, 3967304179347715805UL);
	} else if (n < 585226005592931977UL) {
		return millerRabinTest(n, 2UL) &&
			millerRabinTest(n, 123635709730000UL) &&
			millerRabinTest(n, 9233062284813009UL) &&
			millerRabinTest(n, 43835965440333360UL) &&
			millerRabinTest(n, 761179012939631437UL) &&
			millerRabinTest(n, 1263739024124850375UL);
	} else if (n < std::numeric_limits<unsigned long>::max()) {
		return millerRabinTest(n, 2UL) &&
			millerRabinTest(n, 365UL) &&
			millerRabinTest(n, 9375UL) &&
			millerRabinTest(n, 28178UL) &&
			millerRabinTest(n, 450775UL) &&
			millerRabinTest(n, 9780504UL) &&
			millerRabinTest(n, 1795265022UL);
	} else {
		std::cerr << "Error, could not find isPrime implementation." << std::endl;
		std::exit(3);
		return false;
	}
}

#if DEBUG

static bool isPrimeSlow(const unsigned long n) {
	if (n <= 2) {
		return n == 2;
	}

	for (unsigned long p = 2; p * p <= n; p++) {
		if ((n % p) == 0) {
			return false;
		}
	}

	return true;
}

#include <omp.h>

void checkIsPrime(const unsigned long bound) {
#pragma omp parallel for schedule(dynamic,1)
	for (unsigned long n = 0; n <= bound; n++) {
		const bool fastResult = isPrime(n);
		const bool slowResult = isPrimeSlow(n);
		if (fastResult != slowResult) {
			std::cerr << "Error, isPrime and isPrimeSlow disagree for n=" << n << "\n";
			std::cerr << "isPrime returns " << fastResult << "\n";
			std::cerr << "isPrimeSlow returns " << slowResult << std::endl;
			std::exit(3);
		}
	}
}

#endif // DEBUG
