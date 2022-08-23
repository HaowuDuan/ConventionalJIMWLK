cd(@__DIR__)
using LinearAlgebra
#generators for fundamental representation
t=[[ 0  1/2 0
    1/2  0  0
     0   0  0  ],

    [ 0  0-im/2 0
      0+im/2  0  0
      0   0  0  ],

      [ 1/2   0  0
        0   -1/2 0
        0     0  0],

        [ 0   0  1/2
          0   0  0
          1/2   0  0 ],

          [ 0   0  0-im/2
            0   0  0
            0+im/2   0  0 ],

          [ 0   0  0
            0   0  1/2
            0   1/2  0],

            [ 0   0     0
              0   0  0-im/2
              0   0+im/2  0],

              [ 1/(sqrt(12))   0  0
                0   1/(sqrt(12))  0
                0     0  -2/(sqrt(12)) ]]



# Adjoint SU(3) generators
T=Array{ComplexF64}(undef, 8, 8, 8)

for a in 1:8
  for b in 1:8
    for c in 1:8
      T[c, a, b]=-2tr(t[c]*(t[a]*t[b]-t[b]*t[a]))
    end
  end
end
