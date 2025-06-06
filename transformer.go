package transformer

import (
    "fmt"
    "math"
    "gorgonia.org/gorgonia"
    "gorgonia.org/tensor"
)

// PositionalEncoding builds sinusoidal positional encodings.
func PositionalEncoding(maxLen, dModel int) (*tensor.Dense, error) {
    pe := tensor.New(tensor.WithShape(maxLen, dModel), tensor.WithBacking(make([]float32, maxLen*dModel)))
    for pos := 0; pos < maxLen; pos++ {
        for i := 0; i < dModel; i++ {
            angle := float32(pos) / float32(math.Pow(10000, float64(2*(i/2))/float64(dModel)))
            if i%2 == 0 {
                pe.SetAt(float32(math.Sin(float64(angle))), pos, i)
            } else {
                pe.SetAt(float32(math.Cos(float64(angle))), pos, i)
            }
        }
    }
    return pe, nil
}

// LayerNorm applies layer normalization over the last dimension of x.
func LayerNorm(x *gorgonia.Node, epsilon float64) (*gorgonia.Node, error) {
    // Compute mean over last axis
    mean, err := gorgonia.Mean(x, -1)
    if err != nil {
        return nil, err
    }
    // Broadcast mean to x's shape
    meanB, err := gorgonia.Broadcast(mean, x.Shape(), nil, []byte{byte(len(x.Shape()) - 1)})
    if err != nil {
        return nil, err
    }
    // Center x
    centered, err := gorgonia.Sub(x, meanB)
    if err != nil {
        return nil, err
    }
    // Compute variance
    sq, err := gorgonia.Square(centered)
    if err != nil {
        return nil, err
    }
    variance, err := gorgonia.Mean(sq, -1)
    if err != nil {
        return nil, err
    }
    // Broadcast variance
    varB, err := gorgonia.Broadcast(variance, x.Shape(), nil, []byte{byte(len(x.Shape()) - 1)})
    if err != nil {
        return nil, err
    }
    // Normalize
    std, err := gorgonia.Sqrt(varB)
    if err != nil {
        return nil, err
    }
    denom, err := gorgonia.AddScalar(std, epsilon, true)
    if err != nil {
        return nil, err
    }
    normed, err := gorgonia.HadamardDiv(centered, denom)
    if err != nil {
        return nil, err
    }
    return normed, nil
}

// MultiHeadAttention represents the multi-head attention mechanism.
type MultiHeadAttention struct {
    heads   int
    dModel  int
    dHead   int
    wQ, wK, wV *gorgonia.Node // Linear weight matrices
    wO         *gorgonia.Node // Output linear weight
}

// NewMultiHeadAttention constructs a new MultiHeadAttention layer.
func NewMultiHeadAttention(g *gorgonia.ExprGraph, heads, dModel int) *MultiHeadAttention {
    dHead := dModel / heads
    return &MultiHeadAttention{
        heads:  heads,
        dModel: dModel,
        dHead:  dHead,
        wQ:     gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(dModel, dModel), gorgonia.WithName("WQ")),
        wK:     gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(dModel, dModel), gorgonia.WithName("WK")),
        wV:     gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(dModel, dModel), gorgonia.WithName("WV")),
        wO:     gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(dModel, dModel), gorgonia.WithName("WO")),
    }
}

// Forward applies multi-head attention to inputs Q, K, V.
func (mha *MultiHeadAttention) Forward(q, k, v *gorgonia.Node) (*gorgonia.Node, error) {
    Q, _ := gorgonia.Mul(q, mha.wQ)
    K_, _ := gorgonia.Mul(k, mha.wK)
    V_, _ := gorgonia.Mul(v, mha.wV)

    scores, _ := gorgonia.Mul(Q, gorgonia.Must(gorgonia.Transpose(K_)))
    scale := 1.0 / float32(math.Sqrt(float64(mha.dHead)))
    scoresScaled, _ := gorgonia.Mul(scores, gorgonia.NewConstant(scale))

    attnWeights, _ := gorgonia.SoftMax(scoresScaled)
    context, _ := gorgonia.Mul(attnWeights, V_)

    out, err := gorgonia.Mul(context, mha.wO)
    if err != nil {
        return nil, err
    }
    return out, nil
}

// FeedForward is the position-wise feed-forward network.
type FeedForward struct {
    w1, w2 *gorgonia.Node
    b1, b2 *gorgonia.Node
}

// NewFeedForward constructs a two-layer feed-forward network.
func NewFeedForward(g *gorgonia.ExprGraph, dModel, dFF int) *FeedForward {
    return &FeedForward{
        w1: gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(dModel, dFF), gorgonia.WithName("W1")),
        b1: gorgonia.NewVector(g, tensor.Float32, gorgonia.WithShape(dFF), gorgonia.WithName("B1")),
        w2: gorgonia.NewMatrix(g, tensor.Float32, gorgonia.WithShape(dFF, dModel), gorgonia.WithName("W2")),
        b2: gorgonia.NewVector(g, tensor.Float32, gorgonia.WithShape(dModel), gorgonia.WithName("B2")),
    }
}

// Forward applies the feed-forward network.
func (ff *FeedForward) Forward(x *gorgonia.Node) (*gorgonia.Node, error) {
    l1, err := gorgonia.Add(gorgonia.Must(gorgonia.Mul(x, ff.w1)), ff.b1)
    if err != nil {
        return nil, err
    }
    relu := gorgonia.Must(gorgonia.Rectify(l1))
    l2, err := gorgonia.Add(gorgonia.Must(gorgonia.Mul(relu, ff.w2)), ff.b2)
    if err != nil {
        return nil, err
    }
    return l2, nil
}

// TransformerBlock encapsulates one encoder block.
type TransformerBlock struct {
    mha *MultiHeadAttention
    ff  *FeedForward
}

// NewTransformerBlock builds one encoder layer.
func NewTransformerBlock(g *gorgonia.ExprGraph, heads, dModel, dFF int) *TransformerBlock {
    return &TransformerBlock{
        mha: NewMultiHeadAttention(g, heads, dModel),
        ff:  NewFeedForward(g, dModel, dFF),
    }
}

// Forward runs the block: attention, add & norm, FF, add & norm.
func (tb *TransformerBlock) Forward(x *gorgonia.Node) (*gorgonia.Node, error) {
    attnOut, err := tb.mha.Forward(x, x, x)
    if err != nil {
        return nil, err
    }
    res1, _ := gorgonia.Add(x, attnOut)
    normed1, err := LayerNorm(res1, 1e-6)
    if err != nil {
        return nil, err
    }

    ffOut, err := tb.ff.Forward(normed1)
    if err != nil {
        return nil, err
    }
    res2, _ := gorgonia.Add(normed1, ffOut)
    normed2, err := LayerNorm(res2, 1e-6)
    if err != nil {
        return nil, err
    }

    return normed2, nil
}

// Example usage:
func Example() {
    g := gorgonia.NewGraph()
    batchSize, seqLen, dModel := 2, 16, 64
    input := gorgonia.NewTensor(g, tensor.Float32, 3, gorgonia.WithShape(batchSize, seqLen, dModel), gorgonia.WithName("input"))

    pe, _ := PositionalEncoding(seqLen, dModel)
    fmt.Println("Positional Encoding shape:", pe.Shape())

    block := NewTransformerBlock(g, 8, dModel, 256)
    out, err := block.Forward(input)
    if err != nil {
        panic(err)
    }
    fmt.Println("Output node:", out)
}
