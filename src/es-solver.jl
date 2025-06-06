module ES_solver

import PlasmaModelingToolkit.Constants: c
import PlasmaModelingToolkit.Constants: ε_0
import PlasmaModelingToolkit.Materials: Medium
import JLD2

struct Nodes{Z1, Z2, R1, R2}
    mask :: Matrix{Bool}
end

struct Edges{A, B, C, U, V} end


const NZ = 61
const NR = 29
const DZ = 0.0005
const DR = 0.0005
const div_E = zeros(Float64, NZ, NR)
const ρ = zeros(Float64, NZ, NR)
const φ = zeros(Float64, NZ, NR)
const E_z = zeros(Float64, NZ - 1, NR)
const E_r = zeros(Float64, NZ, NR - 1)
const V = zeros(Float64, NZ, NR)
const R = zeros(Float64, NZ, NR)
const PHI = zeros(Float64, NZ, NR)
const RHS = zeros(Float64, NZ, NR)
const RES = zeros(Float64, NZ, NR)
const A_0 = JLD2.load("data/es-data.jld2", "A_0")
const A_1 = JLD2.load("data/es-data.jld2", "A_1")
const A_2 = JLD2.load("data/es-data.jld2", "A_2")
const A_3 = JLD2.load("data/es-data.jld2", "A_3")
const A_4 = JLD2.load("data/es-data.jld2", "A_4")
const var"##nod#492" = JLD2.load("data/es-data.jld2", "##nod#492")
const var"##mat#493" = JLD2.load("data/es-data.jld2", "##mat#493")
const var"##nod#494" = JLD2.load("data/es-data.jld2", "##nod#494")
const var"##mat#495" = JLD2.load("data/es-data.jld2", "##mat#495")
const var"##edg#496" = JLD2.load("data/es-data.jld2", "##edg#496")
const var"##edg#497" = JLD2.load("data/es-data.jld2", "##edg#497")



# James R. Nagel, nageljr@ieee.org
# "Solving the Generalized Poisson Equation Using the Finite-Difference Method (FDM)"
# 2012
function update_potential!(::Medium{EPS, MU, SIG},
  nodes::Nodes{Z1, Z2, R1, R2}) where {EPS, MU, SIG, Z1, Z2, R1, R2}
  mask = nodes.mask
  for j=R1:R2, i=Z1:Z2
    if mask[i,j] continue end
    RES[i,j]  = RHS[i,j]
    RES[i,j] += A_1[i,j] * PHI[i+1,j]
    RES[i,j] += A_2[i,j] * PHI[i,j-1]
    RES[i,j] += A_3[i,j] * PHI[i-1,j]
    RES[i,j] += A_4[i,j] * PHI[i,j+1]
    RES[i,j] /= A_0[i,j]
    RES[i,j] -= PHI[i,j]
  end
  return nothing
end

function update_electric!(::Medium{EPS, MU, SIG},
  nodes::Nodes{Z1, Z2, R1, R2}) where {EPS, MU, SIG, Z1, Z2, R1, R2}
  mask = nodes.mask
  
  for j=R1:R2, i=Z1:Z2-1
    E_z[i,j] = (φ[i,j] - φ[i+1,j]) / DZ
  end

  for j=R1:R2-1, i=Z1:Z2
    E_r[i,j] = (φ[i,j] - φ[i,j+1]) / DR
  end
  return nothing
end

function update_potential_nbc!(::Edges{Z1, Z2, R0, 0,+1}) where {Z1, Z2, R0}
    z1, z2 = (Z1 > 1 ? Z1 : 2), (Z2 < NZ ? Z2 : NZ - 1)

    if Z1 == 1
        update_potential_nbc!(Val{(Z1,R0)}())
    end

    if Z2 == NZ
        update_potential_nbc!(Val{(Z2,R0)}())
    end
    
    for j=R0, i=z1:z2
        RES[i,j]  = RHS[i,j]
        RES[i,j] += A_1[i,j] * PHI[i+1,j]
        RES[i,j] += A_2[i,j] * PHI[i,j+1]
        RES[i,j] += A_3[i,j] * PHI[i-1,j]
        RES[i,j] += A_4[i,j] * PHI[i,j+1]
        RES[i,j] /= A_0[i,j]
        RES[i,j] -= PHI[i,j]
    end
end

function update_potential_nbc!(::Edges{Z1, Z2, R0, 0,-1}) where {Z1, Z2, R0}
    z1, z2 = (Z1 > 1 ? Z1 : 2), (Z2 < NZ ? Z2 : NZ - 1)

    if Z1 == 1
        update_potential_nbc!(Val{(Z1,R0)}())
    end

    if Z2 == NZ
        update_potential_nbc!(Val{(Z2,R0)}())
    end

    for j=R0, i=z1:z2
        RES[i,j]  = RHS[i,j]
        RES[i,j] += A_1[i,j] * PHI[i+1,j]
        RES[i,j] += A_2[i,j] * PHI[i,j-1]
        RES[i,j] += A_3[i,j] * PHI[i-1,j]
        RES[i,j] += A_4[i,j] * PHI[i,j-1]
        RES[i,j] /= A_0[i,j]
        RES[i,j] -= PHI[i,j]
    end
end

function update_potential_nbc!(::Edges{R1, R2, Z0,+1, 0}) where {R1, R2, Z0}
    r1, r2 = (R1 > 1 ? R1 : 2), (R2 < NR ? R2 : NR - 1)

    if R1 == 1
        update_potential_nbc!(Val{(Z0,R1)}())
    end
    
    if R2 == NR
        update_potential_nbc!(Val{(Z0,R2)}())
    end


    for j=r1:r2, i=Z0
        RES[i,j]  = RHS[i,j]
        RES[i,j] += A_1[i,j] * PHI[i+1,j]
        RES[i,j] += A_2[i,j] * PHI[i,j-1]
        RES[i,j] += A_3[i,j] * PHI[i+1,j]
        RES[i,j] += A_4[i,j] * PHI[i,j+1]
        RES[i,j] /= A_0[i,j]
        RES[i,j] -= PHI[i,j]
    end
end

function update_potential_nbc!(::Edges{R1, R2, Z0,-1, 0}) where {R1, R2, Z0}
    r1, r2 = (R1 > 1 ? R1 : 2), (R2 < NR ? R2 : NR - 1)

    if R1 == 1
        update_potential_nbc!(Val{(Z0,R1)}())
    end

    if R2 == NR
        update_potential_nbc!(Val{(Z0,R2)}())
    end

    for j=r1:r2, i=Z0
        RES[i,j]  = RHS[i,j]
        RES[i,j] += A_1[i,j] * PHI[i-1,j]
        RES[i,j] += A_2[i,j] * PHI[i,j-1]
        RES[i,j] += A_3[i,j] * PHI[i-1,j]
        RES[i,j] += A_4[i,j] * PHI[i,j+1]
        RES[i,j] /= A_0[i,j]
        RES[i,j] -= PHI[i,j]
    end
end

function update_potential_nbc!(::Val{IJ}) where {IJ}
    i, j = IJ
    RES[i,j]  = RHS[i,j]
    RES[i,j] += (i < NZ) ? A_1[i,j] * PHI[i+1,j] : A_1[i,j] * PHI[i-1,j]
    RES[i,j] += (j > 1 ) ? A_2[i,j] * PHI[i,j-1] : A_2[i,j] * PHI[i,j+1]
    RES[i,j] += (i > 1 ) ? A_3[i,j] * PHI[i-1,j] : A_3[i,j] * PHI[i+1,j]
    RES[i,j] += (j < NR) ? A_4[i,j] * PHI[i,j+1] : A_4[i,j] * PHI[i,j-1]
    RES[i,j] /= A_0[i,j]
    RES[i,j] -= PHI[i,j]
    nothing
end

function initialize!()
    fill!(PHI, 0.0)
    fill!(RHS, 0.0)
    fill!(RES, 0.0)
    
    for jj = 1:NR
        radius = 0.0 + (jj - 1) * DR
        area_r = (2π) * min(radius + 0.5DR, 0.014) * DZ
        area_z = π * min(radius + 0.5DR, 0.014) ^ 2 - π * max(radius - 0.5DZ, 0.0) ^ 2
        volume = DZ * area_z
        
        for ii = 1:NZ
            V[ii, jj] = volume
            R[ii, jj] = radius
        end
        
    end

    for ii = 1:1, jj = 2:NR
        V[ii, jj] /= 2.0
    end

    for ii = NZ, jj = 2:NR
        V[ii, jj] /= 2.0
    end
    
    return nothing
end

function solve!()
    PHI .= 0.0
    RHS .= ρ
    
    for _ = 1:1000
        update_potential!(var"##mat#493", var"##nod#492")
        update_potential!(var"##mat#495", var"##nod#494")
        PHI .+= RES
        update_potential_nbc!(var"##edg#497")
    end
    
    φ .= PHI
    update_electric!(var"##mat#493", var"##nod#492")
    update_electric!(var"##mat#495", var"##nod#494")

    return nothing
end
end
