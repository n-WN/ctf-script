// MITM + DP for Knapsack (Go)
//
// This program demonstrates two classic approaches to the subset-sum/knapsack
// problem using the given 48×96 dataset:
// - meet-in-the-middle (MITM) with 128-bit integer sums
// - DP (modulo bitset) as a feasibility filter and teaching aid
//
// Notes
// - MITM enumerates 2^(n/2) sums per side. For n=48, each side has 2^24≈16.7M
//   subsets. A full in-memory run may require ~400–500MB RAM for left sums
//   (struct ~24 bytes × 16.7M ≈ 400MB) plus overhead. It is doable on many
//   machines, but you can reduce n (via -n) to run a quick demo, then try n=48.
// - We store 128-bit sums in two uint64 (Hi,Lo), which is enough because the
//   sum of up to 24 ~96-bit numbers is < 2^101.
// - We enumerate subsets in Gray code order so each next sum differs by ±a[i]
//   (one add/sub per step), reducing the number of 128-bit operations.
// - DP modulo (power-of-two M) is provided as a teaching filter; it cannot
//   recover the exact solution alone for large instances but is useful to learn
//   the idea.
//
// Usage examples
//   go run mitm_dp_knapsack.go -mode mitm-demo -n 32
//   # Attempt the full instance (memory intensive):
//   go run mitm_dp_knapsack.go -mode mitm -n 48
//   # Modulo-DP feasibility check (fast):
//   go run mitm_dp_knapsack.go -mode dp-mod -modbits 20
//
// Dataset
// The constants below are from the provided DEBUG sample. You can replace them
// or adapt the program to read external input as needed.
package main

import (
	"errors"
	"flag"
	"fmt"
	"math/big"
	"math/bits"
	"os"
	"runtime"
	"sort"
	"time"
)

type Uint128 struct{ Hi, Lo uint64 }

type Entry struct {
	Sum  Uint128
	Mask uint32 // store BINARY mask for the half (not Gray)
}

// ---- 128-bit helpers ----

func add128(a, b Uint128) Uint128 {
	lo, c := bits.Add64(a.Lo, b.Lo, 0)
	hi, _ := bits.Add64(a.Hi, b.Hi, c)
	return Uint128{Hi: hi, Lo: lo}
}

func sub128(a, b Uint128) (Uint128, bool) {
	lo, c := bits.Sub64(a.Lo, b.Lo, 0)
	hi, brr := bits.Sub64(a.Hi, b.Hi, c)
	return Uint128{Hi: hi, Lo: lo}, brr != 0 // underflow if borrow out
}

func cmp128(a, b Uint128) int {
	if a.Hi < b.Hi {
		return -1
	}
	if a.Hi > b.Hi {
		return 1
	}
	if a.Lo < b.Lo {
		return -1
	}
	if a.Lo > b.Lo {
		return 1
	}
	return 0
}

func fromBigInt(x *big.Int) (Uint128, error) {
	if x.Sign() < 0 {
		return Uint128{}, errors.New("negative not supported")
	}
	// Extract low/high 64 bits
	lo := new(big.Int).And(x, big.NewInt(0).Sub(new(big.Int).Lsh(big.NewInt(1), 64), big.NewInt(1)))
	hi := new(big.Int).Rsh(x, 64)
	if hi.BitLen() > 64 {
		return Uint128{}, errors.New("value exceeds 128 bits")
	}
	return Uint128{Hi: hi.Uint64(), Lo: lo.Uint64()}, nil
}

func fromDecString(s string) (Uint128, error) {
	x, ok := new(big.Int).SetString(s, 10)
	if !ok {
		return Uint128{}, fmt.Errorf("invalid decimal: %s", s)
	}
	return fromBigInt(x)
}

// grayToBinary converts a Gray-coded 32-bit value to binary.
func grayToBinary32(g uint32) uint32 {
	// Classic loop
	for shift := uint(1); shift < 32; shift <<= 1 {
		g ^= g >> shift
	}
	return g
}

// enumerateSumsGray enumerates subset sums in Gray order but stores BINARY masks.
func enumerateSumsGray(vals []Uint128) []Entry {
	m := len(vals)
	N := 1 << m
	res := make([]Entry, N)
	var sum Uint128
	var prevG uint32 = 0
	var maskBin uint32 = 0
	res[0] = Entry{Sum: sum, Mask: 0}
	for k := 1; k < N; k++ {
		g := uint32(k) ^ uint32(k>>1) // Gray code
		delta := prevG ^ g             // toggled bit
		idx := uint(bits.TrailingZeros32(delta))
		// toggle presence bit in binary mask
		maskBin ^= (1 << idx)
		if ((g >> idx) & 1) == 1 {
			// bit turned on => add
			sum = add128(sum, vals[idx])
		} else {
			// bit turned off => subtract
			var under bool
			sum, under = sub128(sum, vals[idx])
			if under {
				panic("underflow in Gray enumeration (should not happen)")
			}
		}
		prevG = g
		res[k] = Entry{Sum: sum, Mask: maskBin}
	}
	return res
}

// searchEntry finds target in sorted entries by Sum.
func searchEntry(entries []Entry, target Uint128) int {
	idx := sort.Search(len(entries), func(i int) bool {
		return cmp128(entries[i].Sum, target) >= 0
	})
	if idx < len(entries) && cmp128(entries[idx].Sum, target) == 0 {
		return idx
	}
	return -1
}

// mitmSolve returns (found, fullMask) where fullMask has n bits (LSB-first). The
// left/right masks returned are converted from Gray to binary before composing.
func mitmSolve(a []Uint128, bag Uint128) (bool, uint64) {
	n := len(a)
	m := n / 2
	left := a[:m]
	right := a[m:]

	start := time.Now()
	L := enumerateSumsGray(left)
	sort.Slice(L, func(i, j int) bool { return cmp128(L[i].Sum, L[j].Sum) < 0 })
	fmt.Printf("[MITM] Left sums: %d entries, built+sorted in %v\n", len(L), time.Since(start))

	// Enumerate right in Gray order, track BINARY mask on the fly
	R := right
	Rlen := len(R)
	var sumR Uint128
	var prevG uint32 = 0
	var rightMaskBin uint32 = 0

	for k := 0; k < (1 << Rlen); k++ {
		if k == 0 {
			// sumR=0, rightMaskBin=0
		} else {
			g := uint32(k) ^ uint32(k>>1)
			delta := prevG ^ g
			idx := uint(bits.TrailingZeros32(delta))
			// toggle binary mask and update sum
			rightMaskBin ^= (1 << idx)
			if ((g >> idx) & 1) == 1 { // on
				sumR = add128(sumR, R[idx])
			} else { // off
				var under bool
				sumR, under = sub128(sumR, R[idx])
				if under {
					panic("underflow in Gray enumeration (right)")
				}
			}
			prevG = g
		}
		// need = bag - sumR
		need, under := sub128(bag, sumR)
		if under {
			continue
		}
		idx := searchEntry(L, need)
		if idx >= 0 {
			leftMaskBin := L[idx].Mask
			fullMask := uint64(leftMaskBin) | (uint64(rightMaskBin) << uint(m))
			return true, fullMask
		}
	}
	return false, 0
}

// dpResidueModPow2 computes reachable residues modulo M=2^modBits using a bitset.
// It returns whether bag%M is reachable.
func dpResidueModPow2(a []Uint128, bag Uint128, modBits uint) bool {
	if modBits == 0 || modBits > 30 {
		panic("modBits must be in 1..30 for this demo")
	}
	M := uint32(1) << modBits
	mask := M - 1
	// bitset size M bits => M/64 words
	words := (uint64(M) + 63) / 64
	dp := make([]uint64, words)
	dp[0] = 1 // residue 0 reachable

	for _, v := range a {
		// residue = value % M. Since M is power-of-two and value is 128-bit hi<<64+lo,
		// residue depends only on low modBits bits of Lo (2^64 is divisible by 2^modBits).
		r := uint32(v.Lo & uint64(mask))
		if r == 0 {
			continue
		}
		// rotate-left dp by r positions (modulo M) and OR back (classic convolution on Z/MZ)
		rot := rotateBitset(dp, uint64(r), uint64(M))
		for i := range dp {
			dp[i] |= rot[i]
		}
	}
	// target residue
	tr := uint32(bag.Lo & uint64(mask))
	word := tr / 64
	bit := tr % 64
	return ((dp[word] >> bit) & 1) == 1
}

// rotateBitset performs a rotate-left by k over a ring of N bits, where dp has N bits.
func rotateBitset(dp []uint64, k, N uint64) []uint64 {
	if k == 0 {
		res := make([]uint64, len(dp))
		copy(res, dp)
		return res
	}
	res := make([]uint64, len(dp))
	// Normalize k
	k %= N
	if k == 0 {
		copy(res, dp)
		return res
	}
	words := uint64(len(dp))
	wordShift := k / 64
	bitShift := k % 64
	for i := uint64(0); i < words; i++ {
		// source word index
		j := (i + words - wordShift) % words
		lo := dp[j]
		var hi uint64
		if bitShift == 0 {
			hi = 0
		} else {
			j2 := (j + words - 1) % words
			hi = dp[j2]
		}
		if bitShift == 0 {
			res[i] = lo
		} else {
			res[i] = (lo << bitShift) | (hi >> (64 - bitShift))
		}
	}
	// Mask off unused tail bits beyond N
	unused := uint64(len(dp))*64 - N
	if unused > 0 {
		res[len(dp)-1] &= ^uint64(0) >> unused
	}
	return res
}

// ---- Dataset (from DEBUG block) ----

// flag{0x96ebec349e96}
// [48680176571161696314172514101, 75074172672694990229746755091, 56762341961855194664970618041, 48728941710705331441095482429, 76160192941339071317820684217, 54760513188051502531590346321, 61208889257403686102589831859, 46912877869797225624972114739, 58975289627165969912985054017, 59512834812284109101344061777, 42194432974389646711430644657, 47238596212109969505870582191, 44245731327413912750062491361, 57362651548161866515506749237, 56600647098257099639013982321, 41429338136581338024147348431, 49308978193880293941182931509, 73090991290440028917038720773, 60237133734685389132236376971, 71459755944226200248453690047, 71832814014007423345221300401, 73488321541928363381227370393, 69288983607493893894420042451, 59318963960196949572050557817, 42178968810443878661606657183, 46000580376169550491933876909, 64736142691287057276671380799, 43370324045827326360822580283, 75663480557994950971093754623, 57631916981584816941844134103, 60462088137797853756015537149, 64240958812795536807006687809, 39946861609584030942279578197, 50759849893143398138001800497, 56052403529480888682158278829, 42991818499658081981429587199, 63834899042757969616742864897, 50222772763217996228163625673, 72165771160676503934923266937, 74113456291472126279172949649, 69342276153283673931472202087, 53595451155478433035560046661, 52905655479031164979033513123, 61124804224983347547181542307, 40715422031007339222636998117, 70122862794306375681983634703, 75091252116516226639573537741, 48390216443481529216730548933]

var aDec = []string{
	"48680176571161696314172514101",
	"75074172672694990229746755091",
	"56762341961855194664970618041",
	"48728941710705331441095482429",
	"76160192941339071317820684217",
	"54760513188051502531590346321",
	"61208889257403686102589831859",
	"46912877869797225624972114739",
	"58975289627165969912985054017",
	"59512834812284109101344061777",
	"42194432974389646711430644657",
	"47238596212109969505870582191",
	"44245731327413912750062491361",
	"57362651548161866515506749237",
	"56600647098257099639013982321",
	"41429338136581338024147348431",
	"49308978193880293941182931509",
	"73090991290440028917038720773",
	"60237133734685389132236376971",
	"71459755944226200248453690047",
	"71832814014007423345221300401",
	"73488321541928363381227370393",
	"69288983607493893894420042451",
	"59318963960196949572050557817",
	"42178968810443878661606657183",
	"46000580376169550491933876909",
	"64736142691287057276671380799",
	"43370324045827326360822580283",
	"75663480557994950971093754623",
	"57631916981584816941844134103",
	"60462088137797853756015537149",
	"64240958812795536807006687809",
	"39946861609584030942279578197",
	"50759849893143398138001800497",
	"56052403529480888682158278829",
	"42991818499658081981429587199",
	"63834899042757969616742864897",
	"50222772763217996228163625673",
	"72165771160676503934923266937",
	"74113456291472126279172949649",
	"69342276153283673931472202087",
	"53595451155478433035560046661",
	"52905655479031164979033513123",
	"61124804224983347547181542307",
	"40715422031007339222636998117",
	"70122862794306375681983634703",
	"75091252116516226639573537741",
	"48390216443481529216730548933",
}

const bagDec = "1511337494195129828889342583399"

func loadDataset(n int) ([]Uint128, Uint128, error) {
	if n < 1 || n > len(aDec) {
		return nil, Uint128{}, fmt.Errorf("n must be in 1..%d", len(aDec))
	}
	a := make([]Uint128, n)
	for i := 0; i < n; i++ {
		u, err := fromDecString(aDec[i])
		if err != nil {
			return nil, Uint128{}, err
		}
		a[i] = u
	}
	bag, err := fromDecString(bagDec)
	if err != nil {
		return nil, Uint128{}, err
	}
	return a, bag, nil
}

func formatMaskLSB(mask uint64, n int) string {
	// Return bitstring with LSB first (bit 0 first) for teaching consistency
	bs := make([]byte, n)
	for i := 0; i < n; i++ {
		if ((mask >> uint(i)) & 1) == 1 {
			bs[i] = '1'
		} else {
			bs[i] = '0'
		}
	}
	return string(bs)
}

func verifyMaskSum(a []Uint128, bag Uint128, mask uint64) bool {
	// recompute sum using big.Int for safety
	sum := new(big.Int)
	for i := 0; i < len(a); i++ {
		if ((mask >> uint(i)) & 1) == 1 {
			// add a[i]
			val := new(big.Int).SetUint64(a[i].Lo)
			if a[i].Hi != 0 {
				hi := new(big.Int).Lsh(new(big.Int).SetUint64(a[i].Hi), 64)
				val.Add(val, hi)
			}
			sum.Add(sum, val)
		}
	}
	bagBig := new(big.Int).SetUint64(bag.Lo)
	if bag.Hi != 0 {
		hi := new(big.Int).Lsh(new(big.Int).SetUint64(bag.Hi), 64)
		bagBig.Add(bagBig, hi)
	}
	return sum.Cmp(bagBig) == 0
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	mode := flag.String("mode", "mitm-demo", "mode: mitm | mitm-demo | dp-mod")
	nFlag := flag.Int("n", 32, "use first n items (mitm/mitm-demo)")
	modBits := flag.Uint("modbits", 20, "DP modulo uses M=2^modbits (dp-mod)")
	flag.Parse()

	fmt.Printf("Mode=%s\n", *mode)
	switch *mode {
	case "mitm", "mitm-demo":
		if *mode == "mitm-demo" && *nFlag > 36 {
			fmt.Println("[hint] mitm-demo is intended for n<=36 to run quickly. For full n=48, use -mode mitm and ensure you have enough RAM.")
		}
		a, bag, err := loadDataset(*nFlag)
		if err != nil {
			fmt.Println("loadDataset error:", err)
			os.Exit(1)
		}
		fmt.Printf("n=%d; building left sums (this may take time & memory) ...\n", len(a))
		start := time.Now()
		ok, mask := mitmSolve(a, bag)
		elapsed := time.Since(start)
		if !ok {
			fmt.Printf("[result] no solution found in %v\n", elapsed)
			return
		}
		fmt.Printf("[result] found in %v\n", elapsed)
		fmt.Printf("mask (lsb->msb): %s\n", formatMaskLSB(mask, len(a)))
		if len(a) <= 64 {
			fmt.Printf("p (uint64)     : %d\n", mask)
			fmt.Printf("p (hex)        : 0x%x\n", mask)
		} else {
			fmt.Println("p exceeds 64 bits; construct big.Int if needed.")
		}

	case "dp-mod":
		a, bag, err := loadDataset(48)
		if err != nil {
			fmt.Println("loadDataset error:", err)
			os.Exit(1)
		}
		start := time.Now()
		ok := dpResidueModPow2(a, bag, *modBits)
		elapsed := time.Since(start)
		fmt.Printf("[dp-mod] mod 2^%d reachable? %v (time %v)\n", *modBits, ok, elapsed)
		fmt.Println("Note: modulo-DP is a feasibility filter, not an exact solver for large instances.")

	default:
		fmt.Println("unknown mode; use -mode mitm | mitm-demo | dp-mod")
	}
}
