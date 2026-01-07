/* Copyright (C) 2019 Paul Zimmermann, LORIA/INRIA

  This program is free software; you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by the
  Free Software Foundation; either version 2 of the License, or (at your
  option) any later version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
  more details.

  You should have received a copy of the GNU General Public License along
  with this program; see the file COPYING.  If not, write to the Free
  Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  02111-1307, USA.
*/

/* compile with:
   gcc -ftrapv -O3 -g gauss_sigma.c -lm -lgmp -fopenmp
*/

/* Meilleur programme : celui à lancer
*/

/* references:
   [1] https://encompass.eku.edu/etd/158
   [2] https://www.mersenneforum.org/showthread.php?t=21068
   [3] https://stackoverflow.com/questions/2269810/whats-a-nice-method-to-factor-gaussian-integers
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <omp.h>
#include <gmp.h>
#include <signal.h>

#include "carg_parser.h"

#define UWtype unsigned long
#define UDItype unsigned long
#define W_TYPE_SIZE 64
#define ASSERT assert

static const char * invocation_name = "gauss_sigma";	/* default value */

/* known loops (will not be printed) */
#if 1
long known[][2] = {/* bound=10000000: */
                   {-2639,-1228},{-1105,1020},
                   /* bound=20000000: */
                   {-1895,2060},{-3433,-2356},
                   /* bound=50000000: */
                   {-4694,467},{-3235,1020},{-4478,-5471},{-3970,2435},
                   {-3549,-4988},{-766,-6187},
                   /* bound=100000000: */
                   {-6468,-5251},{4232,-8280},{5356,-6133},{8008,3960},
                   /* bound=200000000: */
                   {-8733,-10366},{-8547,4606},{-5068,-11551},{11636,-4473},
                   /* bound=500000000: */
                   {736,-16560},{-14612,-7159},{-13056,-9187},{-12449,-2978},
                   {-9004,-12573},{-6880,4275},{-991,-16702},{4212,-19241},
                   {13484,-10787},{17648,768},
                   /* bound=1000000000: */
                   {-24877,-15664},{-22233,-7876},{-21246,-8807},{3033,-28124},
                   {5166,-26953},{11092,20564},{9877,-27536},{10028,23596},
                   {15128,-17112},{16072,14712},
                   /* bound=2000000000: */
                   {-29258,-8631},{29281,-19358},{-25407,-2404},
                   {-24766,-12687},{-22843,9474},{-22043,-22476},
                   {-18557,-27474},{-13681,-29842},{2367,-36956},{4104,-43720},
                   {5978,-33729},{7766,-33313},{20243,-33324},{42696,4120},
                   /* bound=4294967295=2^32-1: */
                   {-42529,-11098},{-30638,-951},{-36769,-17538},
                   {-31606,-9787},{-22681,-53392},{-18322,-44169},
                   {20732,48724},{-15832,-52632},{-10688,59384},{-6074,-47573},
                   {-2584,52512},{5953,-51974},{7969,-64062},{9912,-45641},
                   {10336,54064},{20868,34476},{46436,36252},{43528,9161},
                   {44764,38148},{49481,-33408},{50632,-14568},{53312,-10800},
                   /* bound=8589934592=2^33: */
                   {-15952,65248},{64000,15248},
                   {-41942,-47569},{-61678,-28721},{-59756,-3027},
                   {-51323,-35586},{-46517,-38374},{-37428,-47141},
                   {-5844,-82733},{-1796,69428},{-1324,64732},
                   {14518,-81999},{18848,70928},{24923,-72414},
                   {45557,-62426},{57942,-44431},{61908,-42619},{69760,-16432},
                   /* bound=17179869184=2^34: */
                   {-62731,8168},{-71738,-41641},{-67144,96192},
                   {-66793,-32466},{-63656,111408},{-44789,-84008},
                   {-37103,-50084},{15673,-105774},{40058,-105719},
                   {52751,-87580},
                   /* bound=34359738368=2^35: */
                   {-157567,-36594},{-125473,-111986},{-121359,-98788},
                   {-118256,70544},{151661,-66148},{-110035,18430},
                   {132848,66836},{-100792,-118979},{-95718,18549},
                   {-92319,-75308},{-74727,62586},{-74576,-121232},
                   {-71785,-86970},{-69658,-131531},{102120,136712},
                   {-66224,-129168},{-65837,-61294},{-61776,69368},
                   {-58861,-135452},{-53357,-155774},{-47962,-130309},
                   {-37344,152392},{-33276,-134107},{-32139,-125938},
                   {-16024,132832},{-9443,-166326},{3856,123992},
                   {12127,-162286},{23036,154052},{40964,149948},
                   {65357,-119346},{67184,120560},{82599,-121532},
                   {104319,-140692},{105592,-148221},{108433,-105094},
                   {110512,76684},{111098,-81589},{122836,104152},
                   {129276,-57893},{137739,-69262},{142680,-115112},
                   {150764,95048},{155043,-26474},
                   /* bound=68719476736=2^36: */
                   {234886,106357},{-164899,-98468},{-153616,110888},
                   {-130384,164960},{-119184,45912},{-99879,-213128},
                   {-48616,211088},{-67197,-128050},{170176,125296},
                   {-46984,242912},{-41452,168796},{-30388,202884},
                   {-29016,153888},{-8744,257392},{12772,-161191},
                   {50104,212856},{176920,-78152},{65480,166952},
                   {114874,-196277},{118879,-231472},{167037,-107630},
                   {187868,-8729},{193863,-157784},{228296,-61656},
                   /* bound=137438953472=2^37: */
                   {-338904,-121528},{54226,-337943},{101034,-353887},
                   {-11457,-300424},{-287496,-116072},{-265626,-51257},
                   {-259222,-59439},{-256434,-112313},{-215014,-121723},
                   {307424,-152407},{246392,198968},{-220570,-46565},
                   {-211024,114576},{-203672,-30529},{-184168,220624},
                   {-183996,-36447},{-181232,129176},{154424,-298440},
                   {-153561,1548},{207094,-166027},{-142839,-241828},
                   {-108484,-289463},{-98624,-301193},{-98374,-219893},
                   {-48328,-270471},{-37959,-261388},{-35864,-298440},
                   {12022,-309201},{21394,-328067},{32796,-269953},
                   {67416,298912},{68664,332048},{71236,-235783},
                   {101888,-277968},{109624,252024},{105164,344748},
                   {126436,330252},{137914,-248177},{140892,254644},
                   {248280,-178216},{151080,289816},{157320,240616},
                   {157908,216956},{181032,274360},{205144,250333},
                   {215608,-268568},{265256,-173533},{272776,159240},
                   {279548,37191},{282168,-204760},{287864,-25560},
                   {292528,103008},{293124,-106057},{301559,-146012},
                   {305400,-176776},{345057,-13976},
                   /* bound=274877906944=2^38: */
                   /* length 6: */
                   {-507253,70523},{-225727,335537},{-155165,433407},
                   {243117,130729},{112387,373843},{139123,294287},
                   /* length 2: */
                   {-14529,-341098},{-461624,-95768},{-452888,232384},
                   {-115387,-493534},{-408636,-127027},{-392801,-15432},
                   {-385192,214016},{-369976,-109432},{-365173,-221936},
                   {-349539,-233248},{-343912,174016},{-339374,-103253},
                   {-338885,17630},{-334180,5815},{-152092,324956},
                   {-316458,-29381},{-309764,-174233},{-303496,247328},
                   {-297368,186944},{-294413,-125726},{130504,418904},
                   {-238142,125971},{-266853,-240896},{-242818,-359251},
                   {-251422,39081},{-168578,-331081},{-234812,377716},
                   {-228504,347792},{-226424,428032},{-220776,343888},
                   {181312,422784},{-211300,18485},{-205188,262284},
                   {466072,-155160},{-184979,-394578},{50352,-398816},
                   {-151588,419684},{-149084,-342773},{274360,248216},
                   {-130287,-397834},{382794,-130217},{384088,-171032},
                   {-121194,-373783},{-109529,-330378},{-98768,-290351},
                   {-83452,-488439},{255240,-329624},{-36502,-481339},
                   {-5479,-396528},{3374,-388747},{8136,424120},
                   {40569,-409442},{53676,-460493},{69773,-413794},
                   {95568,-430624},{101612,414384},{102388,437616},
                   {132968,435240},{146312,362792},{158339,-410352},
                   {168373,-417664},{168568,324600},{178111,-345898},
                   {220916,281837},{236804,-467047},{237888,399616},
                   {244072,267848},{262040,-309416},{271448,-273608},
                   {295353,-318604},{306360,250424},{307084,-217837},
                   {336632,-183000},{361049,-154582},{372289,181098},
                   {387419,-207342},{401848,-199089},{416577,-118166},
                   {431247,-155846},{439016,-161864},{459384,-25480},
                   {471431,-6558},
                   /* bound=2^39: */
                   {304072,391848},{-551488,368912},{-530888,-347416},
                   {428088,-471752},{263902,-493981},{-504968,-153576},
                   {-503402,18861},{-499526,-185907},{-498059,-299648},
                   {-487208,307544},{-461728,35429},{-458247,-391454},
                   {-452528,-233821},{-442001,-234332},{-438464,375280},
                   {-435944,510592},{-434392,249256},{-421856,544008},
                   {-403292,-490569},{-397583,-301806},{-396247,-318254},
                   {-381134,-431113},{-185562,-570259},{-363178,-177571},
                   {-363032,487776},{-347415,-88430},{-346942,-312179},
                   {-345568,544824},{70352,-552561},{-300131,-404924},
                   {-267234,-354103},{97712,685084},{420072,420632},
                   {-239241,-406402},{-237398,-628261},{-215223,-417336},
                   {-192804,-453653},{-178284,482372},{-143846,-718847},
                   {-120364,525752},{-104756,570908},{-103312,-602709},
                   {45752,-512411},{-91636,518248},{-51664,668800},
                   {-28058,-662131},{15191,-544338},{37588,478316},
                   {146372,565404},{155248,-563619},{160898,-551989},
                   {202288,702916},{205726,-559893},{221579,-675712},
                   {229613,-659284},{261201,-516068},{300461,-451348},
                   {303594,-439367},{343103,-463554},{353720,584136},
                   {359394,-440777},{390416,467744},{397888,544688},
                   {398648,-347208},{423086,301727},{426334,-387687},
                   {436087,-447666},{455171,-476356},{461481,-442718},
                   {491539,235348},{499047,-546946},{522248,64411},
                   {528764,-302627},{551223,-310664},{576442,-179581},
                   {582448,34161},{607480,-411336},{636256,48656},
                   {649178,-27629},{657787,241084},{687846,-97153},
                   {-642712,-336584},{-632632,52776},{-355460,201705},
                   /* bound=2^40: */
                   {-1043216,42488},
                   {-972304,44872},
                   {-721368,-250336},
                   {-710128,359704},
                   {-703600,313936},
                   {-668517,-217294}, /* found by JLG, not found by Andrew */
                   {-657187,-231884},
                   {-645712,573808},
                   {-639186,157823},
                   {-626472,-223904},
                   {-583631,-441812},
                   {-573949,26268},
                   {-572228,829004},
                   {-551912,784896},
                   {-537556,-455477},
                   {-527456,709608},
                   {-519772,574996},
                   {-500408,-597281},
                   {-499192,290081},
                   {-497344,732792},
                   {-496272,347496},
                   {-473749,-564848},
                   {-428088,877504},
                   {-424344,-457983},
                   {-388928,-637121},
                   {-371214,-896023},
                   {-353435,-969156},
                   {-241386,-929827},
                   {-131113,-583896},
                   {-37172,812796},
                   {-36792,756968},
                   {14387,-928516},
                   {29842,-970281},
                   {52608,-691344},
                   {63172,955204},
                   {116906,-966683}, /* not found by Andrew */
                   {170504,-856297},
                   {188012,846284},
                   {196992,-889456},
                   {200088,811672},
                   {211971,-659978},
                   {258064,-702992},
                   {263517,-882506}, /* found by JLG, not found by Andrew */
                   {277913,-754084},
                   {281576,635944},
                   {283631,-750028},
                   {288528,701168},
                   {300559,980488},
                   {363188,945116},
                   {368116,-754603},
                   {391696,-737328},
                   {398869,-736912},
                   {433928,729960},
                   {466701,-823918},
                   {480149,645868},
                   {489851,-425868},
                   {511672,773448},
                   {578954,-766097},
                   {587873,691136},
                   {590856,746120},
                   {610527,-694336},
                   {615112,774392},
                   {637144,-612417},
                   {678424,-323944},
                   {708608,-455119},
                   {716246,602097},
                   {743113,-354504},
                   {766087,358084},
                   {768744,-612920},
                   {775672,-470760},
                   {781299,487918},
                   {782509,77338},
                   {799928,-561048},
                   {810792,-6968},
                   {855912,-259672},
                   {884696,89897},
                   {923788,168316},
                   {942574,-13357}, /* not found by Andrew */
                   {946241,-292888},
                   {959758,5481},
                   {960306,-235733},
                   {1007012,146084},
                   /* bound=2^41: */
                   {-1329840,303056},
                   {-1179512,-320184},
                   {-1131784,-713288},
                   {-1047166,-409887},
                   {-1036624,495520},
                   {-934556,-436467},
                   {-926488,-381816},
                   {-902011,-684452},
                   {-893396,-475147},
                   {-886771,-709472},
                   {-885682,-916649}, /* not found by Andrew */
                   {-864140,20345}, /* not found by Andrew */
                   {-855881,-803372}, /* not found by Andrew */
                   {-851888,-1069091},
                   {-821034,-813713},
                   {-793533,92844},
                   {-790616,-672712},
                   {-781594,189167},
                   {-779326,-277267}, /* not found by Andrew */
                   {-743656,1003408}, /* not found by Andrew */
                   {-724697,-661554},
                   {-695843,-557526},
                   {-629476,-461397}, /* not found by Andrew */
                   {-594811,-1194502}, /* not found by Andrew */
                   {-554709,-559788},
                   {-543575,274380}, /* not found by Andrew */
                   {-498887,-785784},
                   {-430233,-723036},
                   {-263476,-1216457}, /* not found by Andrew */
                   {30612,1301884},
                   {59376,-1164943},
                   {64852,873964},
                   {166087,-947796}, /* not found by Andrew */
                   {190316,1145212},
                   {190608,1214156},
                   {193764,1431548},
                   {200032,-1104528},
                   {255418,-1171549},
                   {260368,1316848},
                   {394789,-950652},
                   {422652,1149364}, /* not found by Andrew */
                   {423792,1166644},
                   {485800,1025352},
                   {549067,-1109506},
                   {589416,-1109800},
                   {631097,-1085646},
                   {663201,976082},
                   {665806,-1060733},
                   {665856,-893008},
                   {703051,-972268},
                   {712838,798941},
                   {718243,-879274},
                   {719424,-954032},
                   {721834,-1107087},
                   {738771,-1125728},
                   {757720,-1213864},
                   {778709,-928212},
                   {785480,973272},
                   {799508,968956},
                   {817162,-654941},
                   {818539,-995802},
                   {864092,1081844},
                   {871287,-901016},
                   {896400,696944},
                   {982840,-889032},
                   {1014879,-605522},
                   {1015537,369884},
                   {1019304,612760},
                   {1053996,-84353},
                   {1059800,-520152},
                   {1064760,209192},
                   {1065794,646333},
                   {1090879,-477472},
                   {1093211,374652},
                   {1093368,-456088},
                   {1096582,187549},
                   {1107892,777044},
                   {1120688,-898909},
                   {1120781,358192}, /* not found by Andrew */
                   {1121480,767464},
                   {1128011,-92548},
                   {1134233,-679844},
                   {1192073,121716}, /* not found by Andrew */
                   {1207531,-732458}, /* not found by Andrew */
                   {1213526,-688593},
                   {1219073,-684414},
                   {1220933,479506},
                   {1263308,676356},
                   {1281928,468021},
                   {1303384,-40987},
                   {131326,-1090733}, /* not found by Andrew */
                   {1315288,-544909},
                   {1350072,31879},
                   {1417424,-145457},
                   {1434254,167493},
                   {0,0}};
#else
long known[][2] = {{0,0}};
#endif

typedef struct {
  long x;
  long y;
} complex_t;

unsigned long *Primes = NULL;
unsigned long nprimes = 0;
complex_t *Cprimes = NULL;

typedef struct {
  int size;
  int alloc;
  complex_t *z;
  int *e;
} factor_struct;
typedef factor_struct factor_t[1];

FILE *ffff = NULL;
FILE *fff = NULL;

complex_t *Known = NULL;
unsigned long known_size = 0;
long *lastx;

static void
signal_handler (int sig)
{
  int nthreads = omp_get_max_threads ();
  long last = LONG_MAX;
  for (int i = 0; i < nthreads; i++)
    if (lastx[i] < last)
      last = lastx[i];
  printf ("killed (completed up to %ld included)\n", last);
  fflush (stdout);
  exit (1);
}

static void
init_known (void)
{
  int i, j;
  for (i = 0;; i++)
    {
      for (j = 0; j < i; j++)
        if (known[j][0] == known[i][0] && known[j][1] == known[i][1])
          {
            fprintf (stderr, "Error, duplicate known entry (%ld,%ld)\n",
                     known[j][0], known[j][1]);
            exit (1);
          }
      if (known[i][0] == 0 && known[i][1] == 0)
        {
          known_size = i;
          break;
        }
    }
  printf ("found %lu value(s) in known table\n", known_size);
  Known = malloc (known_size * sizeof (complex_t));
  for (i = 0; i < known_size; i++)
    {
      Known[i].x = known[i][0];
      Known[i].y = known[i][1];
    }
}

static void
factor_init (factor_t f)
{
  f->alloc = f->size = 0;
  f->z = NULL;
  f->e = NULL;
}

/* we should have x > 0 and y >= 0 */
static void
factor_add (factor_t f, long x, long y)
{
  assert (x > 0 && y >= 0);
  if (f->size + 1 > f->alloc)
    {
      f->alloc += 1 + f->alloc / 2;
      f->z = realloc (f->z, f->alloc * sizeof (complex_t));
      f->e = realloc (f->e, f->alloc * sizeof (int));
    }
  f->z[f->size].x = x;
  f->z[f->size].y = y;
  f->e[f->size] = 1;
  f->size++;
}

static void
factor_print (factor_t f)
{
  for (int i = 0; i < f->size; i++)
    printf ("(%ld,%ld)^%d ", f->z[i].x, f->z[i].y, f->e[i]);
  printf ("\n");
}

static void
factor_clear (factor_t f)
{
  free (f->z);
  free (f->e);
}

/* return non-zero if norm(z) <= bound */
int
norm_fits_p (complex_t z, unsigned long bound)
{
  /* first check if the norm fits an unsigned long */
  unsigned long x = (z.x >= 0) ? z.x : -z.x;
  unsigned long y = (z.y >= 0) ? z.y : -z.y;
  if (x >= 4294967296UL || y >= 4294967296UL)
    return 0;
  /* now x, y < 2^32 thus x^2+y^2 < 2^65 */
  x = x * x;
  y = y * y;
  unsigned long n = x + y;
  if (n < x)
    return 0; /* does not fit an unsigned long */
  return n <= bound;
}

/* return norm(z), assuming norm(z) <= bound */
unsigned long
get_norm (complex_t z)
{
  return (unsigned long) z.x * (unsigned long) z.x
    + (unsigned long) z.y * (unsigned long) z.y;
}

/* Check whether u+iv divides z, i.e., of z * (u-iv)/(u^2+v^2) has
   integer coefficients.
   If so, divides z by u+iv.
   We always have p = u^2+v^2 <= sqrt(norm(z)) <= bound.
   No overflow can occur. */
int
try_divide (complex_t *z, long u, long v)
{
  if (u == 0 && v == 0)
    return 0;
  assert (u != 0); /* if v*i divides z, also does v, which we prefer */
  long n = u * u + v * v; /* no overflow since u^2+v^2 = p < 2^32 */
  /* we have |x| <= 2*sqrt(norm(z))*sqrt(p) <= 2*bound^(3/4) < 2^63
     since bound < 2^64 */
  long x = z->x * u + z->y * v;
  if ((x % n) != 0)
    return 0;
  long y = z->x * (-v) + z->y * u; /* same bound for |y| */
  if ((y % n) != 0)
    return 0;
  z->x = x / n;
  z->y = y / n;
  return 1;
}

static unsigned long
mulmod (unsigned long a, unsigned long b, unsigned long p)
{
  if (p <= 4294967295UL)
    return (a * b) % p;
  else
    {
      __uint128_t c = (__uint128_t) a * (__uint128_t) b;
      return c % p;
    }
}

int
nbits (unsigned long p)
{
  int n = 0;
  while (p)
    {
      n ++;
      p = p >> 1;
    }
  return n;
}

/* return a^e mod p */
static unsigned long
powmod (unsigned long a, unsigned long e, unsigned long p)
{
  unsigned long r;
  if (e == 0)
    return 1;
  int n = nbits (e) - 1;
  r = a;
  while (n--)
    {
      r = mulmod (r, r, p);
      if ((e >> n) & 1)
        r = mulmod (r, a, p);
    }
  return r;
}

/* https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test */
static int
miller_rabin (unsigned long n, unsigned long a)
{
  unsigned long d = n-1, s = 1;
  while ((d % 2) == 0)
    d = d / 2, s ++;
  /* now n-1 = 2^s*d with d odd */
  unsigned long b = powmod (a, d, n);
  if (b == 1)
    return 1; /* other powers will be 1 too */
  if (b == n-1)
    return 1;
  for (unsigned long r = 1; r < s; r++)
    {
      b = mulmod (b, b, n);
      if (b == n-1)
        return 1;
    }
  return 0;
}

/* https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test */
static int
isprime (unsigned long n)
{
  if (n <= 2)
    return n == 2;
  if ((n % 2) == 0)
    return 0;
  /* now n is odd and n >= 3 */
  if (n < 2047)
    return miller_rabin (n, 2);
  if (n < 9080191)
    return miller_rabin (n, 31) && miller_rabin (n, 73);
  if (n < 4759123141UL)
    return miller_rabin (n, 2) && miller_rabin (n, 7) && miller_rabin (n, 61);
  if (n < 1122004669633UL)
    return miller_rabin (n, 2) && miller_rabin (n, 13) && miller_rabin (n, 23)
      && miller_rabin (n, 1662803);
  if (n < 2152302898747)
    return miller_rabin (n, 2) && miller_rabin (n, 3) && miller_rabin (n, 5)
      && miller_rabin (n, 7) && miller_rabin (n, 11);
  if (n < 3474749660383)
    return miller_rabin (n, 2) && miller_rabin (n, 3) && miller_rabin (n, 5)
      && miller_rabin (n, 7) && miller_rabin (n, 11) && miller_rabin (n, 13);
  if (n < 341550071728321)
    return miller_rabin (n, 2) && miller_rabin (n, 3) && miller_rabin (n, 5)
      && miller_rabin (n, 7) && miller_rabin (n, 11) && miller_rabin (n, 13)
      && miller_rabin (n, 17);
  if (n < 3825123056546413051)
    return miller_rabin (n, 2) && miller_rabin (n, 3) && miller_rabin (n, 5)
      && miller_rabin (n, 7) && miller_rabin (n, 11) && miller_rabin (n, 13)
      && miller_rabin (n, 17) && miller_rabin (n, 19) && miller_rabin (n, 23);
  fprintf (stderr, "Error, isprime not yet implemented for n >= 3825123056546413051\n");
  exit (1);
}

static int
isprime_slow (unsigned long n)
{
  if (n <= 2)
    return n == 2;
  for (unsigned long p = 2; p * p <= n; p++)
    if ((n % p) == 0)
      return 0;
  return 1;
}

static void
check_isprime (unsigned long bound)
{
#pragma omp parallel for schedule(dynamic,1)
  for (unsigned long n = 0; n <= bound; n++)
    if (isprime (n) != isprime_slow (n))
      {
        fprintf (stderr, "Error, isprime and isprime_slow disagree for n=%lu\n", n);
        fprintf (stderr, "isprime returns %d\n", isprime (n));
        fprintf (stderr, "isprime_slow returns %d\n", isprime_slow (n));
        exit (1);
      }
}

/* return k such that k^2 = -1 mod p for p = 1 mod 4:
   let n >= 2 such that n^((p-1)/2) = -1, then take k = n^((p-1)/4) */
static unsigned long
square_root_of_minus_one (unsigned long p)
{
  unsigned long n, k;
  assert ((p % 4) == 1);
  for (n = 2; n < p; n++)
    {
      k = powmod (n, (p - 1) / 4, p);
      if (mulmod (k, k, p) == p - 1)
        return k;
    }
  /* we should never get there */
  printf ("square_root_of_minus_one failed for p=%lu\n", p);
  assert (0);
}

/* return round(a/b), assuming b > 0 */
long
div_round (long a, long b)
{
  assert (b > 0);
  long q = a / b;
  long r = a - q * b;
  if (2 * r > b) /* r > b/2 */
    return q + 1;
  else if (2 * r < -b) /* r < -b/2 */
    return q - 1;
  else
    return q;
}

/* Return x+I*y = unit * gcd(p, k+I) over the Gaussian integers
   with x+I*y in the first quadrant, thus ax > 0 and ay >= 0.
   Assumes p < 2^32 and 0 < k < p. */
static void
find_divisor (long *x, long *y, long p, long k)
{
  long ax = p, ay = 0, bx = k, by = 1, cx, cy, qx, qy, nb;
  while (bx != 0 || by != 0)
    {
      /* compute a/b = a*conj(b)/b^2 */
      cx = ax * bx + ay * by;
      cy = - ax * by + ay * bx;
      nb = bx * bx + by * by;
      qx = div_round (cx, nb);
      qy = div_round (cy, nb);
      cx = qx * bx - qy * by;
      cy = qx * by + qy * bx;
      cx = ax - cx;
      cy = ay - cy;
      ax = bx; ay = by;
      bx = cx; by = cy;
    }
  if (ax < 0)
    {
      ax = -ax;
      ay = -ay;
    }
  /* now ax >= 0 */
  if (ay < 0)
    ay = -ay; /* complex conjugate */
  if (ax == 0)
    {
      assert (ay > 0);
      /* return -I*z */
      ax = ay;
      ay = 0;
    }
  *x = ax;
  *y = ay;
}

/* initializes all primes < B */
static void
initPrimes (unsigned long B)
{
  unsigned long i, j, p;
  mpz_t T;
  mpz_init (T); /* T[i] = 1 when i is not prime */
  for (i = 2; i * i < B; i++)
    {
      if (mpz_tstbit (T, i) == 0)
        for (j = i * i; j < B; j += i)
          mpz_setbit (T, j);
    }
  for (nprimes = 0, i = 2; i < B; i++)
    if (mpz_tstbit (T, i) == 0)
      nprimes++;
  Primes = malloc ((nprimes + 1) * sizeof (unsigned long));
  for (j = 0, i = 2; i < B; i++)
    if (mpz_tstbit (T, i) == 0)
      Primes[j++] = i;
  /* add a sentinel */
  Primes[nprimes] = B;
  mpz_clear (T);
  assert (j == nprimes);
  printf ("found %lu primes <= %lu\n", nprimes, B-1);
  fflush (stdout);
  Cprimes = malloc (nprimes * sizeof (complex_t));
#pragma omp parallel for schedule(dynamic,1)
  for (i = 0; i < nprimes; i++)
    {
      unsigned long p = Primes[i];
      if ((p % 4) == 1)
        {
          /* case p = 1 mod 4:
             (1) first find k such that k^2 = 1 mod p
             (2) then compute the gcd of p and k+I */
          unsigned long k = square_root_of_minus_one (p);
          long x, y;
          find_divisor (&x, &y, p, k);
          assert (x > 0 && y >= 0);
          Cprimes[i].x = x;
          Cprimes[i].y = y;
        }
    }
  printf ("finished computing complex primes\n");
  fflush (stdout);
}

static void
print_complex (complex_t z)
{
  printf ("(%ld,%ld)\n", z.x, z.y);
}

/* Assuming p divides n, look for all prime divisors of norm p.
   Assumes p < 2^32. */
static void
factor_aux (factor_t f, complex_t *z, unsigned long *n, unsigned long p,
            unsigned long idx)
{
  int ok;
  if (p == 2)
    {
      /* from [1] page 17 (step 3c): if e is the 2-valuation of n, then
         (1+k)^e divides z */
      *n = *n / 2;
      factor_add (f, 1, 1);
      ok = try_divide (z, 1, 1);
      assert (ok);
      while ((*n % 2) == 0)
        {
          *n = *n / 2;
          f->e[f->size-1] ++;
          ok = try_divide (z, 1, 1);
          assert (ok);
        }
      return;
    }
  else if ((p % 4) == 3)
    {
      /* if p = 4k+3, then (x,y) = (p,0) and the exponent should
         be even, see [1] top of page 17 */
      assert ((*n % (p * p)) == 0);
      *n = *n / (p * p);
      factor_add (f, p, 0);
      ok = try_divide (z, p, 0);
      assert (ok);
      while ((*n % (p * p)) == 0)
        {
          *n = *n / (p * p);
          f->e[f->size-1] ++;
          ok = try_divide (z, p, 0);
          assert (ok);
        }
      return;
    }
  assert ((p % 4) == 1);
  /* we use precomputed values of x,y, with x > 0 and y >= 0 */
  long x = Cprimes[idx].x;
  long y = Cprimes[idx].y;
  /* either x+I*y divides z, or its conjugate,
     in fact both can divide, as shown for
     -40-5*I = (-1) * (-I - 2) * (2*I + 1)^2 * (2*I + 3) */
  // printf ("p=%lu k=%lu x=%ld y=%ld z=(%ld,%ld)\n", p, k, x, y, z,x, z.y);
  /* we always have *n divisible by p here */
  /* here x^2+y^2 = p, and p < sqrt(bound) */
  if (try_divide (z, x, y))
    {
      *n = *n / p;
      factor_add (f, x, y);
      while ((*n % p) == 0 && try_divide (z, x, y))
        {
          *n = *n / p;
          f->e[f->size-1] ++;
        }
    }
  /* now try with I*conj(z) = y+I*x */
  if ((*n % p) == 0 && try_divide (z, y, x))
    {
      *n = *n / p;
      factor_add (f, y, x);
      while ((*n % p) == 0 && try_divide (z, y, x))
        {
          *n = *n / p;
          f->e[f->size-1] ++;
        }
    }
}

#define TRACE_X -3160
#define TRACE_Y -113

/* assert z0 <> 0 */
static void
factor (factor_t f, const complex_t z0)
{
  assert (z0.x != 0 || z0.y != 0);
  complex_t z = z0;
  f->size = 0;
  unsigned long n = get_norm (z);
  unsigned long p;
  if (isprime (n))
    goto last_factor;
  for (unsigned long i = 0; i < nprimes; i++)
    {
      p = Primes[i];
      if (p * p > n)
        break;
      if ((n % p) == 0)
        {
          factor_aux (f, &z, &n, p, i);
          if (isprime (n))
            break;
        }
    }
  /* if n > 1, then n is prime, thus the only divisor is z */
 last_factor :
  if (n > 1)
    {
      long x = z.x, y = z.y;
      if (x < 0)
        x = -x, y = -y;
      if (y < 0) /* change into -I*z */
        { long t = x; x = -y; y = t; }
      if (x > 0)
        factor_add (f, x, y);
      else
        factor_add (f, y, 0);
    }
}

/* z1 <- sigma(z) */
static complex_t
sigma (complex_t z)
{
  factor_t f;
  complex_t z1;
  factor_init (f);
  factor (f, z);
  // factor_print (f);
  z1.x = 1;
  z1.y = 0;
  for (int i = 0; i < f->size; i++)
    {
      complex_t t = f->z[i];
      int e = f->e[i];
      complex_t u;
      long x;
      /* compute 1 + t + ... + t^e */
      u.x = 1;
      u.y = 0;
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
  factor_clear (f);
  /* subtract z */
  z1.x -= z.x;
  z1.y -= z.y;
  return z1;
}

static void
find_cycle (complex_t z, unsigned long bound, unsigned long mod)
{
  complex_t *l = NULL;
  /* we avoid (0,0) and (1,0) */
  if (z.y == 0 && (z.x == 0 || z.x == 1))
    return;
  if (!norm_fits_p (z, bound))
    return;
  unsigned long n0 = get_norm (z);
  assert ((n0 % mod) == 0);
  unsigned long alloc = 8;
  l = malloc (alloc * sizeof (complex_t));
  l[0] = z;
  for (int i = 1;; i++)
    {
      if (i >= alloc)
        {
          alloc *= 2;
          l = realloc (l, alloc * sizeof (complex_t));
        }
      /* we have norm(l[i-1]) <= bound here */
      l[i] = sigma (l[i-1]);
      /* we stop if the first element is not the "smallest" one
         for some order function C -> R, here we consider the initial
         element should have the largest x */
      if (l[i].x > z.x)
        break;
      /* check loop */
      for (int j = i-1; j >= 0; j--)
        if (l[j].x == l[i].x && l[j].y == l[i].y)
          {
            int k = i - j;
            if (j < k || (j >= k && (l[j-k].x != l[j].x || l[j-k].y != l[j].y)))
              {
                /* check known loops */
                int ok = 1;
                for (int s = 0; s < known_size; s++)
                  if (Known[s].x == l[j].x && Known[s].y == l[j].y)
                    ok = 0;
                if (ok)
                  {
                    ffff = fopen ("gauss_sigma_C_file_cycles", "a");
                    if (ffff == NULL)
                       printf("Impossible d'ouvrir le fichier en écriture !");
                    else
                       {
                       fprintf (ffff, "loop (%ld,%ld) %d from (%ld,%ld)\n",
                            l[j].x, l[j].y, k, z.x, z.y);
                       fclose (ffff);
                       };
                    printf ("loop (%ld,%ld) %d from (%ld,%ld)\n",
                            l[j].x, l[j].y, k, z.x, z.y);
                    fflush (stdout);
                    /* add in Known to avoid duplicate reports */
                    known_size ++;
                    Known = realloc (Known, known_size * sizeof (complex_t));
                    Known[known_size-1].x = l[j].x;
                    Known[known_size-1].y = l[j].y;
                  }
              }
            /* if a loop is found, we can exit, since no independent loop
               can be found */
            goto end;
          }
      /* we stop when reaching 1 */
      if (l[i].x == 1 && l[i].y == 0)
        break;
      /* we also stop when norm(l[i]) does not fit an unsigned long */
      if (!norm_fits_p (l[i], bound))
        break;
    }
 end:
  free (l);
}

/* Return the smallest y >= y0 such that x^2 + y^2 is divisible by 'mod',
   and put in toggle[0] and toggle[1] values such that the next solution
   is y0 + toggle[0], then y0 + toggle[0] + toggle[1], then
   y0 + toggle[0] + toggle[1] + toggle[0], ...
   Return LONG_MAX if no such y exists.
   Assume 'mod' is prime. */
static long
find_y0 (long y0, long x, long mod, long *toggle)
{
  long s, t;
  if ((x % mod) == 0) /* it suffices to have y = 0 mod 'mod' */
    {
      t = (-y0) % mod;
      if (t < 0)
        t = t + mod;
      assert (0 <= t < mod);
      y0 += t;
      toggle[0] = toggle[1] = mod;
      return y0;
    }
  if (mod == 2)
    {
      if ((x % 2) == 0)
        y0 += y0 & 1; /* y should be even */
      else
        y0 += (y0+1) & 1; /* y should be odd */
      toggle[0] = toggle[1] = 2;
      return y0;
    }
  /* now mod is an odd prime, which does not divide x */
  /* x^2 + y^2 = 0 mod m iff y/x is a root of -1 mod m,
     and there is a root of -1 mod m iff m = 1 mod 4 */
  if ((mod % 4) != 1)
    return LONG_MAX;
  unsigned long r = square_root_of_minus_one (mod);
  r = (x > 0) ? mulmod (r, x, mod) : mod - mulmod (r, -x, mod);
  /* let y = y0 + t = r mod 'mod', thus t = r - y0 mod 'mod' */
  t = (r - y0) % mod;
  if (t < 0)
    t = t + mod;
  r = mod - r;
  s = (r - y0) % mod;
  if (s < 0)
    s = s + mod;
  /* y0 = y0 + min(s,t), y1 = y0 + max(s,t) */
  if (s < t)
    y0 = y0 + s, toggle[0] = t - s;
  else
    y0 = y0 + t, toggle[0] = s - t;
  toggle[1] = mod - toggle[0];
  return y0;
}

static void
internal_error (const char * const msg)
{
  fprintf (stderr, "%s: Internal error: %s\n", invocation_name, msg);
  exit (3);
}

int
main (const int argc, const char * const argv[])
{
  unsigned long mod = 1;
  long xmin = LONG_MIN;
  long xmax = LONG_MAX;
  FILE *fp = NULL;

  /* parse options */
  const ap_Option options[] =
    {
    /* code, long_name, has_arg (no/yes/maybe/yme) */
    { 'm', "mod",  ap_yes },
    { 'x', "xmin", ap_yes },
    { 'X', "xmax", ap_yes },
    { 's', "save", ap_yes },
    { 0, 0,        ap_no  } };

  Arg_parser parser;
  int argind = 0;
  if (argc > 0)
    invocation_name = argv[0];

  if (!ap_init (&parser, argc, argv, options, 0))
    {
      fputs ("Not enough memory.", stderr);
      return 1;
    }
  if (ap_error (&parser))
    {
      fputs (ap_error (&parser), stderr);
      return 1;
    }

  for (; argind < ap_arguments (&parser); ++argind)
    {
    const int code = ap_code (&parser, argind);
    if (!code)
      break;					/* no more options */
    const char * const arg = ap_argument (&parser, argind);
    switch (code)
      {
      case 'm':
        mod = strtoul (arg, NULL, 10);
        printf ("using mod=%lu\n", mod);
        break;
      case 'x':
        xmin = strtol (arg, NULL, 10);
        break;
      case 'X':
        xmax = strtol (arg, NULL, 10);
        break;
      case 's':
        fp = fopen (arg, "w");
        break;
      default:
        internal_error ("uncaught option.");
        return 1;
      }
    } /* end process options */

  const char * const bound_arg = ap_argument (&parser, ++argind);
  const unsigned long bound = strtoul (bound_arg, NULL, 10);

  ap_free (&parser);

  signal (SIGINT, &signal_handler);
  signal (SIGTERM, &signal_handler);

  long xbound = (long) sqrt ((double) bound);
  xmin = (-xbound < xmin) ? xmin : -xbound;
  xmax = (xmax < xbound+1) ? xmax : xbound+1;
  printf ("xbound=%lu xmin=%ld xmax=%ld\n", xbound, xmin, xmax);
  fflush (stdout);

  int nthreads = omp_get_max_threads ();
  lastx = malloc (nthreads * sizeof (long));
  for (int i = 0; i < nthreads; i++)
    lastx[i] = xmin - 1;
  long old_last = xmin - 1;

  assert (sizeof (unsigned long) == 8); /* only works for 64-bit word */
  //  check_isprime (bound);
  init_known ();
  initPrimes (xbound + 1);
#pragma omp parallel for schedule(dynamic,1)
  for (long x = xmin; x < xmax; x++)
    {
      // printf ("starting x=%ld\n", x);
      /* we want x^2 + y^2 <= bound */
      long ymax = (long) sqrt ((double) (bound - x * x)), y0;
      unsigned long t = 0;
      long toggle[2] = {1,1};
      if (mod == 1)
        y0 = -ymax;
      else /* mod > 1 */
        {
          assert (isprime (mod));
          y0 = find_y0 (-ymax, x, mod, toggle);
          // printf ("mod=%ld x=%ld -ymax=%ld y0=%ld toggle[0]=%ld toggle[1]=%ld\n", mod, x, -ymax, y0, toggle[0], toggle[1]);
        }
      // printf ("y0=%ld ymax=%ld\n", y0, ymax);
      for (long y = y0; y <= ymax; y += toggle[t&1], t++)
        {
          complex_t z;
          z.x = x;
          z.y = y;
          find_cycle (z, bound, mod);
        }
      lastx[omp_get_thread_num ()] = x;
#pragma omp critical
      if (fp != NULL)
        {
          long last = LONG_MAX;
          for (int i = 0; i < nthreads; i++)
            if (lastx[i] < last)
              last = lastx[i];
          if (last > old_last)
            {
              fprintf (fp, "%ld\n", old_last = last);
              fflush (fp);
            }
        }
    if (x%100==0)
       {
       fff = fopen ("gauss_sigma_C_file_x", "a");
       if (fff == NULL)
          printf("Impossible d'ouvrir le fichier en écriture !");
       else
       {
        fprintf (fff, "%ld\n", x);
        fclose (fff);
        };
       }
    if (x%1000==0)
       {printf("x=%ld\n",x);};
    }
  printf ("completed xmin=%ld xmax=%ld\n", xmin, xmax);
  fflush (stdout);
  free (Known);
  free (Primes);
  free (Cprimes);
  free (lastx);
  if (fp != NULL)
    fclose (fp);
  return 0;
}
