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

#ifndef PRIMES_H
#define PRIMES_H 1

#include <vector>

std::vector<unsigned long> sievePrimes(const unsigned long pMax);

bool millerRabinTest(const unsigned long n, const unsigned long a);

bool isPrime(const unsigned long n);

#if DEBUG

void checkIsPrime(const unsigned long bound);

#endif // DEBUG

#endif // PRIMES_H
