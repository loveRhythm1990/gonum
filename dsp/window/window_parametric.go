// Copyright ©2020 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package window

import "math"

// GaussianComplex can modify a sequence by the Gaussian window and return the result.
// See https://en.wikipedia.org/wiki/Window_function#Gaussian_window
// and https://www.recordingblogs.com/wiki/gaussian-window for details.
//
// The Gaussian window is an adjustable window.
//
// The sequence weights are
//  w[k] = exp(-0.5 * ((k-M)/(σ*M))² ), M = (N-1)/2,
// for k=0,1,...,N-1 where N is the length of the window.
//
// The properties of the window depend on the value of σ (sigma).
// It can be used as high or low resolution window, depending of the σ value.
//
// Spectral leakage parameters are summarized in the table:
//         |  σ=0.3  |  σ=0.5 |  σ=1.2 |
//  -------|---------------------------|
//  ΔF_0   |   8     |   3.4  |   2.2  |
//  ΔF_0.5 |   1.82  |   1.2  |   0.94 |
//  K      |   4     |   1.7  |   1.1  |
//  ɣ_max  | -65     | -31.5  | -15.5  |
//  β      |  -8.52  |  -4.48 |  -0.96 |
type Gaussian struct {
	Sigma float64
}

// Transform applies the Gaussian transformation to seq in place, using the value
// of the receiver as the sigma parameter, and returning the result.
func (g Gaussian) Transform(seq []float64) []float64 {
	a := float64(len(seq)-1) / 2
	for i := range seq {
		x := -0.5 * math.Pow((float64(i)-a)/(g.Sigma*a), 2)
		seq[i] *= math.Exp(x)
	}
	return seq
}

// TransformComplex applies the Gaussian transformation to seq in place, using the value
// of the receiver as the sigma parameter, and returning the result.
func (g Gaussian) TransformComplex(seq []complex128) []complex128 {
	a := float64(len(seq)-1) / 2
	for i, v := range seq {
		x := -0.5 * math.Pow((float64(i)-a)/(g.Sigma*a), 2)
		w := math.Exp(x)
		seq[i] = complex(w*real(v), w*imag(v))
	}
	return seq
}

// Tukey can modify a sequence using the Tukey window and return the result.
// See https://en.wikipedia.org/wiki/Window_function#Tukey_window
// and https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2017/174042.pdf page 88
//
// The Tukey window is an adjustible window. It can be thought of as something
// between a rectangular and a Hann window, with a flat center and tapered edges.
//
// The properties of the window depend on the value of α (alpha). It controls
// the fraction of the window which contains a cosine taper. α = 0.5 gives a
// window whose central 50% is flat and outer quartiles are tapered. α = 1 is
// equivalent to a Hann window. α = 0 is equivalent to a rectangular window.
// 0 <= α <= 1; if α is outside the bounds, it is treated as 0 or 1.
//
// Spectral leakage parameters are summarized in the table:
//         |  α=0.25 |  α=0.5 | α=0.75 |
//  -------|---------------------------|
//  ΔF_0   |   1.1   |   1.22 |   1.36 |
//  ΔF_0.5 |   1.01  |   1.15 |   2.24 |
//  K      |   1.13  |   1.3  |   2.5  |
//  ɣ_max  | -14     | -15    | -19    |
//  β      |  -1.11  |  -2.5  |  -4.01 |
//
// whose values are from A.D. Poularikas,
// "The Handbook of Formulas and Tables for Signal Processing" table 7.1
type Tukey struct {
	Alpha float64
}

// Transform applies the Tukey transformation to seq in place, using the value
// of the receiver as the Alpha parameter, and returning the result
func (t Tukey) Transform(seq []float64) []float64 {
	if t.Alpha <= 0 {
		return Rectangular(seq)
	} else if t.Alpha >= 1 {
		return Hann(seq)
	}

	alphaL := t.Alpha * float64(len(seq)-1)
	width := int(0.5*alphaL) + 1
	for i := range seq[:width] {
		w := 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/alphaL))
		seq[i] *= w
		seq[len(seq)-1-i] *= w
	}
	return seq
}

// TransformComplex applies the Tukey transformation to seq in place, using the value
// of the receiver as the Alpha parameter, and returning the result
func (t Tukey) TransformComplex(seq []complex128) []complex128 {
	if t.Alpha <= 0 {
		return RectangularComplex(seq)
	} else if t.Alpha >= 1 {
		return HannComplex(seq)
	}

	alphaL := t.Alpha * float64(len(seq)-1)
	width := int(0.5*alphaL) + 1
	for i, v := range seq[:width] {
		w := 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/alphaL))
		v = complex(w*real(v), w*imag(v))
		seq[i] = v
		seq[len(seq)-1-i] = v
	}
	return seq
}

// Values is an arbitrary real window function.
type Values []float64

// NewValues returns a Values of length n with weights corresponding to the
// provided window function.
func NewValues(window func([]float64) []float64, n int) Values {
	v := make(Values, n)
	for i := range v {
		v[i] = 1
	}
	return window(v)
}

// Transform applies the weights in the receiver to seq in place, returning the
// result. If v is nil, Transform is a no-op, otherwise the length of v must
// match the length of seq.
func (v Values) Transform(seq []float64) []float64 {
	if v == nil {
		return seq
	}
	if len(v) != len(seq) {
		panic("window: length mismatch")
	}
	for i, w := range v {
		seq[i] *= w
	}
	return seq
}

// Transform applies the weights in the receiver to seq in place, returning the
// result. If v is nil, Transform is a no-op, otherwise the length of v must
// match the length of seq.
func (v Values) TransformComplex(seq []complex128) []complex128 {
	if v == nil {
		return seq
	}
	if len(v) != len(seq) {
		panic("window: length mismatch")
	}
	for i, w := range v {
		sv := seq[i]
		seq[i] = complex(w*real(sv), w*imag(sv))
	}
	return seq
}
