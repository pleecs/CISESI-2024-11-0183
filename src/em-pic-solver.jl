module EM_PIC_solver

mutable struct State
    time::Float64
end
mutable struct Species
    N_e::Int64
    var"##ps#498"::Float64
    Species() = new() 
end

using SpecialFunctions
using OffsetArrays

struct Nodes{Z1, Z2, R1, R2}
    mask :: Matrix{Bool}
end

struct Edges{A, B, C, U, V} end

gzeros(T, n, m) = OffsetArray(zeros(T, n + 2, m + 2), -1, -1)
    
import PlasmaModelingToolkit.Constants: c
import PlasmaModelingToolkit.Constants: ε_0
import PlasmaModelingToolkit.Constants: μ_0
import PlasmaModelingToolkit.Constants: η_0
import PlasmaModelingToolkit.Constants: χ_01
import JLD2

using StaticArrays
import PlasmaModelingToolkit.Constants: c
import PlasmaModelingToolkit.Constants: kB

struct KineticSpecies{s, m_s, q_s, wg}
    maxcount :: Int64
end

function R01()
    return rand()
end
function RMB(T, m)
    return randn() * sqrt((kB * T) / m)
end

const NZ = 61
const NR = 29
const DZ = 0.0005
const DR = 0.0005
const DT = 5.0e-13
const H_θ = zeros(Float64, NZ - 1, NR - 1)
const E_z = zeros(Float64, NZ - 1, NR)
const E_r = zeros(Float64, NZ, NR - 1)
const V = zeros(Float64, NZ, NR)
const R = zeros(Float64, NZ, NR)
const div_E = zeros(Float64, NZ, NR)
const J_z = gzeros(Float64, NZ - 1, NR)
const J_r = gzeros(Float64, NZ, NR - 1)
const S_z = gzeros(Float64, NZ - 1, NR)
const S_r = gzeros(Float64, NZ, NR - 1)
const state = State(0.0)

const b_z = zeros(Float64, NZ, NR)
const b_r = zeros(Float64, NZ, NR)
const b_θ = zeros(Float64, NZ, NR)
const e_z = zeros(Float64, NZ, NR)
const e_r = zeros(Float64, NZ, NR)
const n_e = zeros(Float64, NZ, NR)
const x_e = zeros(SVector{2, Float64}, 25000)
const u_e = zeros(SVector{3, Float64}, 25000)
const species = Species()
const c² = c ^ 2

const var"##bcs#538" = JLD2.load("data/em-pic-data.jld2", "##bcs#538")
const var"##edg#545" = JLD2.load("data/em-pic-data.jld2", "##edg#545")
const var"##edg#505" = JLD2.load("data/em-pic-data.jld2", "##edg#505")
const var"##edg#507" = JLD2.load("data/em-pic-data.jld2", "##edg#507")
const var"##bcs#544" = JLD2.load("data/em-pic-data.jld2", "##bcs#544")
const var"##mat#500" = JLD2.load("data/em-pic-data.jld2", "##mat#500")
const var"##edg#547" = JLD2.load("data/em-pic-data.jld2", "##edg#547")
const var"##edg#509" = JLD2.load("data/em-pic-data.jld2", "##edg#509")
const var"##edg#511" = JLD2.load("data/em-pic-data.jld2", "##edg#511")
const var"##edg#513" = JLD2.load("data/em-pic-data.jld2", "##edg#513")
const var"##edg#515" = JLD2.load("data/em-pic-data.jld2", "##edg#515")
const var"##edg#517" = JLD2.load("data/em-pic-data.jld2", "##edg#517")
const var"##edg#519" = JLD2.load("data/em-pic-data.jld2", "##edg#519")
const var"##edg#521" = JLD2.load("data/em-pic-data.jld2", "##edg#521")
const var"##int#546" = JLD2.load("data/em-pic-data.jld2", "##int#546")
const var"##edg#523" = JLD2.load("data/em-pic-data.jld2", "##edg#523")
const var"##bcs#514" = JLD2.load("data/em-pic-data.jld2", "##bcs#514")
const var"##edg#525" = JLD2.load("data/em-pic-data.jld2", "##edg#525")
const var"##edg#527" = JLD2.load("data/em-pic-data.jld2", "##edg#527")
const var"##edg#529" = JLD2.load("data/em-pic-data.jld2", "##edg#529")
const var"##edg#531" = JLD2.load("data/em-pic-data.jld2", "##edg#531")
const var"##edg#533" = JLD2.load("data/em-pic-data.jld2", "##edg#533")
const var"##edg#535" = JLD2.load("data/em-pic-data.jld2", "##edg#535")
const var"##nod#503" = JLD2.load("data/em-pic-data.jld2", "##nod#503")
const var"##edg#537" = JLD2.load("data/em-pic-data.jld2", "##edg#537")
const var"##edg#539" = JLD2.load("data/em-pic-data.jld2", "##edg#539")
const var"##edg#541" = JLD2.load("data/em-pic-data.jld2", "##edg#541")
const var"##nod#501" = JLD2.load("data/em-pic-data.jld2", "##nod#501")
const var"##mat#502" = JLD2.load("data/em-pic-data.jld2", "##mat#502")
const var"##edg#543" = JLD2.load("data/em-pic-data.jld2", "##edg#543")

const var"##ps#498" = JLD2.load("data/em-pic-data.jld2", "##ps#498")
const var"##ks#499" = JLD2.load("data/em-pic-data.jld2", "##ks#499")

function compute_divergence!(mask)
    for j=2:NR-1, i=2:NZ-1 if mask[i,j] continue end
        div_E[i,j] +=(E_z[i,j] - E_z[i-1,j]) / DZ
    end
    
    for j=2:NR-1, i=2:NZ-1 if mask[i,j] continue end
        div_E[i,j] +=(E_r[i,j] - E_r[i,j-1]) / DR
        div_E[i,j] +=(E_r[i,j] + E_r[i,j-1]) /2R[i,j]
    end
    
    return nothing
end

function compute_divergence_src!(::Edges{Z1, Z2, R0, 0, _}) where {Z1, Z2, R0, _}
    for j=R0, i=Z1:Z2
        div_E[i,j] = 0.0
    end
end

function compute_divergence_src!(::Edges{R1, R2, Z0, _, 0}) where {R1, R2, Z0, _}
    for j=R1:R2, i=Z0
        div_E[i,j] = 0.0
    end
end

function compute_divergence_pmc!(::Edges{Z1, Z2, R0, 0,+1}) where {Z1, Z2, R0}
    z1, z2 = (Z1 > 1 ? Z1 : 2), (Z2 < NZ ? Z2 : NZ - 1)

    if Z1 == 1
        compute_divergence_pmc!(Val{(Z1,R0)}())
    end

    if Z2 == NZ
        compute_divergence_pmc!(Val{(Z2,R0)}())
    end
    
    for j=R0, i=z1:z2
        div_E[i,j] +=(E_z[i,j] - E_z[i-1,j]) / DZ
    
        div_E[i,j] +=(E_r[i,j] - 0.0) / 0.5DR
    end
end

function compute_divergence_pmc!(::Edges{Z1, Z2, R0, 0,-1}) where {Z1, Z2, R0}
    z1, z2 = (Z1 > 1 ? Z1 : 2), (Z2 < NZ ? Z2 : NZ - 1)

    if Z1 == 1
        compute_divergence_pmc!(Val{(Z1,R0)}())
    end

    if Z2 == NZ
        compute_divergence_pmc!(Val{(Z2,R0)}())
    end
    
    for j=R0, i=z1:z2
        div_E[i,j] +=(E_z[i,j] - E_z[i-1,j]) / DZ
    
        div_E[i,j] +=(0.0 - E_r[i,j-1]) / 0.5DR
    end
end

function compute_divergence_pmc!(::Edges{R1, R2, Z0, +1, 0}) where {R1, R2, Z0}
    r1, r2 = (R1 > 1 ? R1 : 2), (R2 < NR ? R2 : NR - 1)

    if R1 == 1
        compute_divergence_pmc!(Val{(Z0,R1)}())
    end

    if R2 == NR
        compute_divergence_pmc!(Val{(Z0,R2)}())
    end

    for j=r1:r2, i=Z0
        div_E[i,j] +=(E_z[i,j] - 0.0) / 0.5DZ
    
        div_E[i,j] +=(E_r[i,j] - E_r[i,j-1]) / DR
        div_E[i,j] +=(E_r[i,j] + E_r[i,j-1]) /2R[i,j]
    end
end

function compute_divergence_pmc!(::Edges{R1, R2, Z0, -1, 0}) where {R1, R2, Z0}
    r1, r2 = (R1 > 1 ? R1 : 2), (R2 < NR ? R2 : NR - 1)

    if R1 == 1
        compute_divergence_pmc!(Val{(Z0,R1)}())
    end

    if R2 == NR
        compute_divergence_pmc!(Val{(Z0,R2)}())
    end

    for j=r1:r2, i=Z0
        div_E[i,j] +=(0.0 - E_z[i-1,j]) / 0.5DZ

        div_E[i,j] +=(E_r[i,j] - E_r[i,j-1]) / DR
        div_E[i,j] +=(E_r[i,j] + E_r[i,j-1]) /2R[i,j]
    end
end

function compute_divergence_pmc!(::Val{IJ}) where {IJ}
    i, j = IJ

    if i == 1  div_E[i,j] += (E_z[i,j]  -  0.0) / 0.5DZ end
    if i == NZ div_E[i,j] += (0.0 - E_z[i-1,j]) / 0.5DZ end
    if j == 1  div_E[i,j] += (E_r[i,j]  -  0.0) / 0.5DR end
    if j == NR div_E[i,j] += (0.0 - E_r[i,j-1]) / 0.5DR end
end

import PlasmaModelingToolkit.Materials: Medium

function update_magnetic!() 
    @inbounds for j in 1:NR-1, i in 1:NZ-1
        H_θ[i,j] += (DT / μ_0) * (E_z[i,j+1] - E_z[i,j]) / DR +
                   -(DT / μ_0) * (E_r[i+1,j] - E_r[i,j]) / DZ
    end
end

function update_electric!(::Medium{EPS, MU, SIG}, nodes::Nodes{Z1, Z2, R1, R2}) where
    {EPS, MU, SIG, Z1, Z2, R1, R2}
    mask = nodes.mask
    LOSS = SIG * DT / 2ε_0
    for j=R1:R2, i=Z1:Z2-1
        if mask[i,j] && mask[i+1,j] continue end
        
        E_z[i,j] = (1 - LOSS) / (1 + LOSS) *  E_z[i,j] +
                   (DT / EPS) / (1 + LOSS) * (H_θ[i,j] + H_θ[i,j-1]) / 2R[i,j] +
                   (DT / EPS) / (1 + LOSS) * (H_θ[i,j] - H_θ[i,j-1]) / DR 
    end

    for j=R1:R2-1, i=Z1:Z2
        if mask[i,j] && mask[i,j+1] continue end

        E_r[i,j] = (1 - LOSS) / (1 + LOSS) *  E_r[i,j] -
                   (DT / EPS) / (1 + LOSS) * (H_θ[i,j] - H_θ[i-1,j]) / DZ
    end
end

"""
    apply_displacement_current!(medium, nodes)

Updates electric field using displacement currents as sources,
but it works only on **unmasked** nodes. It won't update
electric field on boundaries or dielectric interfaces.
"""
function apply_displacement_current!(::Medium{EPS, MU, SIG},
    nodes::Nodes{Z1, Z2, R1, R2}) where {EPS, MU, SIG, Z1, Z2, R1, R2}
    mask = nodes.mask
    LOSS = SIG * DT / 2EPS
    @inbounds for j=R1:R2, i=Z1:Z2-1
        if mask[i,j] && mask[i+1,j] continue end
        E_z[i,j] -= (DT / EPS) / (1 + LOSS) * (44/120 * J_z[i,j] + 38/120 * J_z[i+1,j] + 38/120 * J_z[i-1,j])
    end
    @inbounds for j=R1:R2-1, i=Z1:Z2
        if mask[i,j] && mask[i,j+1] continue end
        E_r[i,j] -= (DT / EPS) / (1 + LOSS) * (44/120 * J_r[i,j] + 38/120 * J_r[i,j+1] + 38/120 * J_r[i,j-1])
    end
end

function apply_electric_source!(::Edges{Z1, Z2, R0, 0, _}, I_s) where {Z1, Z2, R0, _}
  @inbounds for i=Z1:(Z2-1)
    J_z[i, R0] = I_s / S_z[i, R0]
  end
end

function update_electric_src!(::Edges{R1, R2, Z0, _, 0}, ε, η) where {R1, R2, Z0, _}
    EPS = ε
    LOSS = (1 / η / DZ) * (DT / EPS)
    @inbounds for j in R1:R2-1, i=Z0
        E_r[i,j] -= (DT / EPS) / (1 + LOSS) * J_r[i,j]
    end
end

function update_electric_src!(::Edges{Z1, Z2, R0, 0, _}, ε, η) where {Z1, Z2, R0, _}
    EPS = ε
    LOSS = (1 / η / DZ) * (DT / EPS)
    @inbounds for i in Z1:Z2-1, j=R0
        E_z[i,j] -= (DT / EPS) / (1 + LOSS) * J_z[i,j]
    end
end

import PlasmaModelingToolkit.InterfaceConditions: DielectricInterface

function update_electric_pec!(::Edges{R1, R2, Z0, _, 0}) where {R1, R2, Z0, _}
  @inbounds for j=R1:R2-1, i=Z0
    E_r[i,j] = 0.0
  end
end

function update_electric_pec!(::Edges{Z1, Z2, R0, 0, _}) where {Z1, Z2, R0, _}
  @inbounds for i=Z1:Z2-1, j=R0
    E_z[i,j] = 0.0
  end
end

function update_electric_pmc!(::Edges{Z1, Z2, R0, 0, -1}) where {Z1, Z2, R0}
  EPS = ε_0
  LOSS = 0.0
  @inbounds for i=Z1:Z2-1, j=R0
    E_z[i,j] = (1 - LOSS) / (1 + LOSS) *  E_z[i,j] +
               (DT / EPS) / (1 + LOSS) * (-2H_θ[i,j-1]) / DR 
  end
end

function update_electric_pmc!(::Edges{Z1, Z2, R0, 0, +1}) where {Z1, Z2, R0}
  EPS = ε_0
  LOSS = 0.0
  @inbounds for i=Z1:Z2-1, j=R0
    E_z[i,j] = (1 - LOSS) / (1 + LOSS) *  E_z[i,j] +
               (DT / EPS) / (1 + LOSS) * (+2H_θ[i,j]) / DR 
  end
end

function update_electric_pmc!(::Edges{R1, R2, Z0, -1, 0}) where {R1, R2, Z0}
  EPS = ε_0
  LOSS = 0.0
  for j=R1:R2-1, i=Z0
    E_r[i,j] = (1 - LOSS) / (1 + LOSS) *  E_r[i,j] -
               (DT / EPS) / (1 + LOSS) * (-2H_θ[i-1,j]) / DZ
  end
end

function update_electric_pmc!(::Edges{R1, R2, Z0, +1, 0}) where {R1, R2, Z0}
  EPS = ε_0
  LOSS = 0.0
  for j=R1:R2-1, i=Z0
    E_r[i,j] = (1 - LOSS) / (1 + LOSS) *  E_r[i,j] -
               (DT / EPS) / (1 + LOSS) * (+2H_θ[i,j]) / DZ
  end
end

function update_electric!(::DielectricInterface{EPS1, EPS2, SIG},
                          ::Edges{R1, R2, Z0, _, 0}) where {EPS1, EPS2, SIG, R1, R2, Z0, _}
    d = 0.0
    EPS = (0.5 + d) * EPS1 + (0.5 - d) * EPS2
    LOSS = SIG * DT / 2EPS
    for j=R1:R2-1, i=Z0
        E_r[i,j] = (1 - LOSS) / (1 + LOSS) *  E_r[i,j] -
                   (DT / EPS) / (1 + LOSS) * (H_θ[i,j] - H_θ[i-1,j]) / DZ
    end
end

function update_electric!(::DielectricInterface{EPS1, EPS2, SIG},
                          ::Edges{Z1, Z2, R0, 0, _}) where {EPS1, EPS2, SIG, Z1, Z2, R0, _}
    d = 0.0
    EPS = (0.5 + d) * EPS1 + (0.5 - d) * EPS2
    LOSS = SIG * DT / 2EPS
    for i=Z1:Z2-1, j=R0
        E_z[i,j] = (1 - LOSS) / (1 + LOSS) *  E_z[i,j] +
                   (DT / EPS) / (1 + LOSS) * (H_θ[i,j] + H_θ[i,j-1]) / 2R[i,j] +
                   (DT / EPS) / (1 + LOSS) * (H_θ[i,j] - H_θ[i,j-1]) / DR
    end
end


function calculate_electric_energy(::Medium{EPS, MU, SIG}, nodes::Nodes) where {EPS, MU, SIG}
    U_E = 0.0
    mask = nodes.mask
    for j in 1:NR-1, i in 1:NZ-1
        if (mask[i,j] == mask[i+1,j] == false)
            E_1 = E_r[i,j] + E_r[i+1,j]
            U_E += 0.25E_1^2 * V[i,j]
        end

        if (mask[i,j] == mask[i,j+1] == false)
            E_2 = E_z[i,j] + E_z[i,j+1]
            U_E += 0.25E_2^2 * V[i,j]
        end
    end
    return 0.5EPS * U_E
end

function calculate_magnetic_energy(::Medium{EPS, MU, SIG}, nodes::Nodes) where {EPS, MU, SIG}
    U_H = 0.0
    mask = nodes.mask
    for j in 1:NR-1, i in 1:NZ-1
        if (mask[i,j] == mask[i+1,j+1] == false) &&
           (mask[i+1,j] == mask[i,j+1] == false)
            H_0 = H_θ[i,j]
            U_H += H_0^2 * V[i,j]
        end
    end
    return 0.5MU * U_H
end

@generated function remove_particle!(ks::KineticSpecies{s, m_s, q_s, wg},
    p) where {s, m_s, q_s, wg}
    N_s = Symbol(String(:N_) * String(s))
    u_s = Symbol(String(:u_) * String(s))
    x_s = Symbol(String(:x_) * String(s))
quote
    $(x_s)[p] = $(x_s)[species.$(N_s)]
    $(u_s)[p] = $(u_s)[species.$(N_s)]
    
    species.$(N_s) -= 1
    return nothing
end end

using LinearAlgebra

@inline function lorentz_factor(u)
    u² = dot(u, u)
    return sqrt(1 + u² / c²)
end

# Relativistic Boris pusher (Ripperda2018)
@generated function update_momentum!(ks::KineticSpecies{sym, m_s, q_s, wg},
    p, E, B) where {sym, m_s, q_s, wg}
    u_s = Symbol(String(:u_) * String(sym))
quote
    Q = .5DT * $(q_s) / $(m_s)
    
    # first half electric acceleration
    u  = $(u_s)[p] + Q * E # u⁻
    u² = dot(u, u)
    γ  = sqrt(1. + u²/c²)
    t  = Q / γ * B
    t² = dot(t, t)
    s  = 2.0t / (1. + t²)
    
    # rotation step
    u⁺ = u + (u + (u × t)) × s
    
    # second half electric field acceleration
    $(u_s)[p] = u⁺ + Q * E
    return nothing
end end

@generated function integrate_velocity!(ks::KineticSpecies{s, m_s, q_s, wg},
    p) where {s, m_s, q_s, wg}
    u_s = Symbol(String(:u_) * String(s))
    x_s = Symbol(String(:x_) * String(s))
quote
    dx = DT * $(u_s)[p][1]
    dy = DT * $(u_s)[p][2]
    $(x_s)[p] += @SVector [dx, dy]
    return nothing
end end

@generated function cartesian_to_cylindrical!(ks::KineticSpecies{s, m_s, q_s, wg},
    p) where {s, m_s, q_s, wg}
    u_s = Symbol(String(:u_) * String(s))
    x_s = Symbol(String(:x_) * String(s))
quote
    vZ = $(u_s)[p][1]
    vR = $(u_s)[p][2]
    vθ = $(u_s)[p][3]
    
    Z  = $(x_s)[p][1]
    R  = $(x_s)[p][2]
    Y  = vθ * DT
    
    r  = √(R^2 + Y^2)

    cosθ = (r ≈ 0.0) ? 1.0 : R / r
    sinθ = (r ≈ 0.0) ? 0.0 : Y / r
    
    vr = cosθ * vR - sinθ * vθ
    vy = sinθ * vR + cosθ * vθ
    
    $(x_s)[p] = @SVector [ Z,  r]
    $(u_s)[p] = @SVector [vZ, vr, vy]
    
    return nothing
end end

@generated function current_deposition!(ks::KineticSpecies{s, m_s, q_s, wg},
       x_1, y_1, x_2, y_2, v_x, v_y) where {s, m_s, q_s, wg}
quote
    i_1 = floor(Int64, x_1 / DZ)
    j_1 = floor(Int64, y_1 / DR)
    i_2 = floor(Int64, x_2 / DZ)
    j_2 = floor(Int64, y_2 / DR)
    
    x_r  = min(min(i_1 * DZ, i_2 * DZ) + DZ,
           max(max(i_1 * DZ, i_2 * DZ), .5(x_1 + x_2)))
    F_x1 = $(wg) * $(q_s) * (x_r - x_1) / DT / DZ
    F_x2 = $(wg) * $(q_s) * (v_x) / DZ - F_x1
    W_x1 = (x_1 + x_r) / 2.0DZ - i_1
    W_x2 = (x_r + x_2) / 2.0DZ - i_2
        
    y_r  = min(min(j_1 * DR, j_2 * DR) + DR,
           max(max(j_1 * DR, j_2 * DR), .5(y_1 + y_2)))
    F_y1 = $(wg) * $(q_s) * (y_r - y_1) / DT / DR
    F_y2 = $(wg) * $(q_s) * (v_y) / DR - F_y1
    W_y1 = (y_1 + y_r) / 2.0DR - j_1
    W_y2 = (y_r + y_2) / 2.0DR - j_2
    
    J_z[i_1+1,j_1+1] += F_x1 / S_z[i_1+1,j_1+1] * (1. - W_y1) # J_z[i_1 + ½, j_1]
    J_z[i_1+1,j_1+2] += F_x1 / S_z[i_1+1,j_1+2] *      (W_y1) # J_z[i_1 + ½, j_1 + 1]
        
    J_z[i_2+1,j_2+1] += F_x2 / S_z[i_2+1,j_2+1] * (1. - W_y2) # J_z[i_2 + ½, j_2]
    J_z[i_2+1,j_2+2] += F_x2 / S_z[i_2+1,j_2+2] *      (W_y2) # J_z[i_2 + ½, j_2 + 1]
        
    J_r[i_1+1,j_1+1] += F_y1 / S_r[i_1+1,j_1+1] * (1. - W_x1) # J_r[i_1,     j_1 + ½]
    J_r[i_1+2,j_1+1] += F_y1 / S_r[i_1+2,j_1+1] *      (W_x1) # J_r[i_1 + 1, j_1 + ½]
        
    J_r[i_2+1,j_2+1] += F_y2 / S_r[i_2+1,j_2+1] * (1. - W_x2) # J_r[i_2,     j_2 + ½]
    J_r[i_2+2,j_2+1] += F_y2 / S_r[i_2+2,j_2+1] *      (W_x2) # J_r[i_2 + 1, j_2 + ½]
    return nothing
end end

@generated function cloud_in_cell(ks::KineticSpecies{s, m_s, q_s, wg},
    p) where {s, m_s, q_s, wg}
    x_s = Symbol(String(:x_) * String(s))
quote
    h_z = $(x_s)[p][1] / DZ
    h_r = $(x_s)[p][2] / DR
    i = floor(Int64, h_z)
    j = floor(Int64, h_r)
    W_z = h_z - i
    W_r = h_r - j
    
    return i, j, W_z, W_r
end end

@generated function push_particles!(ks::KineticSpecies{s, m_s, q_s, wg}
    ) where {s, m_s, q_s, wg}
    N_s = Symbol(String(:N_) * String(s))
    u_s = Symbol(String(:u_) * String(s))
    x_s = Symbol(String(:x_) * String(s))
quote
    for p in 1:species.$(N_s)
        i, j, W_z, W_r = cloud_in_cell(ks, p)
        
        E_p = @SVector [
              e_z[i+1,j+1] * (1. - W_z) * (1. - W_r) +
              e_z[i+2,j+1] * (W_z)      * (1. - W_r) +
              e_z[i+1,j+2] * (1. - W_z) *      (W_r) +
              e_z[i+2,j+2] * (W_z)      *      (W_r),
              e_r[i+1,j+1] * (1. - W_z) * (1. - W_r) +
              e_r[i+2,j+1] * (W_z)      * (1. - W_r) +
              e_r[i+1,j+2] * (1. - W_z) *      (W_r) +
              e_r[i+2,j+2] * (W_z)      *      (W_r),
              0.0]
        B_p = @SVector [b_z[i+1,j+1], b_r[i+1,j+1], b_θ[i+1,j+1]]
        
        x_1 = $(x_s)[p][1]
        y_1 = $(x_s)[p][2]
        
        update_momentum!(ks, p, E_p, B_p)
        integrate_velocity!(ks, p)
        cartesian_to_cylindrical!(ks, p)
        
        x_2 = $(x_s)[p][1]
        y_2 = $(x_s)[p][2]

        if abs(x_2 - x_1) > DZ @error "Particle is too fast!" end
        if abs(y_2 - y_1) > DR @error "Particle is too fast!" end

        γ   = lorentz_factor($(u_s)[p])
        v_x = $(u_s)[p][1] / γ
        v_y = $(u_s)[p][2] / γ

        current_deposition!(ks, x_1, y_1, x_2, y_2, v_x, v_y)
    end
end end

@generated function calculate_kinetic_energy(::KineticSpecies{s, m_s, q_s, wg}) where {s, m_s, q_s, wg}
    N_s = Symbol(String(:N_) * String(s))
    u_s = Symbol(String(:u_) * String(s))
quote
    MASS = $(m_s)
    U_K = 0.0
    for p in 1:species.$(N_s)
        u_p = $(u_s)[p]
        U_K += dot(u_p, u_p) * $(wg)
    end
    return 0.5MASS * U_K
end end

function interpolate_electric!()
    for j = 1:NR, i = 2:NZ - 1
        e_z[i, j] = E_z[i, j] / 2 + E_z[i - 1, j] / 2
    end
    for j = 2:NR - 1, i = 1:NZ
        e_r[i, j] = E_r[i, j] / 2 + E_r[i, j - 1] / 2
    end
end

function interpolate_magnetic!()
    return nothing
end

function compute_divergence!()    
    fill!(div_E, 0.0)
    compute_divergence!((var"##nod#501").mask)
    compute_divergence!((var"##nod#503").mask)
    compute_divergence_src!(var"##edg#515")
    compute_divergence_pmc!(var"##edg#539")
end

function calculate_electric_energy()
    U_E = 0.0    
    U_E += calculate_electric_energy(var"##mat#500", var"##nod#501")
    U_E += calculate_electric_energy(var"##mat#502", var"##nod#503")
    return U_E
end
function calculate_magnetic_energy()
    U_H = 0.0    
    U_H += calculate_magnetic_energy(var"##mat#500", var"##nod#501")
    U_H += calculate_magnetic_energy(var"##mat#502", var"##nod#503")
    return U_H
end


function calculate_kinetic_energy()
    U_K = 0.0
    U_K += calculate_kinetic_energy(var"##ks#499")
    return U_K
end


function apply_electric_sources!()
    apply_electric_source!(var"##edg#515", 0.5 * (15.5 * sin((2π) * 5.087e9 * state.time) * exp((-0.5 * (state.time - 5.0e-9) ^ 2) / 1.0e-9 ^ 2)))
end


function apply_particle_sources!()
    
    n_s = (
        if state.time <= 0.0
            0.0
        elseif state.time <= 7.5e-9
            ((((state.time - 0.0) ^ 2 / 1.5e-8 ^ 2) / 1.0) * 2.0) * 3.0 + 0.0
        elseif state.time <= 7.5e-9
            (0.5 + ((state.time - 7.5e-9) / 1.5e-8) * 2.0) * 3.0 + 0.0
        elseif state.time <= 1.5e-8
            (1.0 - (((state.time - 1.5e-8) ^ 2 / 1.5e-8 ^ 2) / 1.0) * 2.0) * 3.0 + 0.0
        elseif state.time > 1.5e-8
            3.0
        end / 1.60217662e-13) * DT + species.var"##ps#498"
    N_s = floor(Int64, n_s)
    for i = species.N_e + 1:species.N_e + N_s
        x_e[i] = @SVector([rand() * 0.002 + 0.0024000000000000002, sqrt(rand() * 4.0e-6 + 0.0)])
        u_e[i] = @SVector([randn() * sqrt((kB * 300.0) / 9.10938356e-31) + 1.20918436036692e8, randn() * sqrt((kB * 300.0) / 9.10938356e-31) + 0.0, randn() * sqrt((kB * 300.0) / 9.10938356e-31) + 0.0])
    end
    species.N_e += N_s
    species.var"##ps#498" = n_s - N_s  
end


function apply_particle_loaders!()  
end

function initialize!()
    state.time = 0.0
    
    fill!(E_z, 0.0)
    fill!(E_r, 0.0)
    fill!(H_θ, 0.0)
    for jj = 1:NR
        radius = 0.0 + (jj - 1) * DR
        area_r = (2π) * min(radius + 0.5DR, 0.014) * DZ
        area_z = π * min(radius + 0.5DR, 0.014) ^ 2 - π * max(radius - 0.5DZ, 0.0) ^ 2
        volume = DZ * area_z
        for ii = 1:NZ
            V[ii, jj] = volume
            R[ii, jj] = radius
        end
        for ii = 0:NZ
            S_z[ii, jj] = area_z
        end
        for ii = 0:NZ + 1
            S_r[ii, jj] = area_r
        end
    end
    for ii = 0:NZ + 1, jj = 0
        S_r[ii, jj] = 0.0
    end
    for ii = 0:NZ, jj = 0
        S_z[ii, jj] = 0.0
    end
    for ii = 0:NZ, jj = NR + 1
        S_z[ii, jj] = π * ((0.014 + 0.5DR) ^ 2 - 0.014 ^ 2)
    end
    for ii = 1, jj = 2:NR
        V[ii, jj] /= 2.0
    end
    for ii = NZ, jj = 2:NR
        V[ii, jj] /= 2.0
    end
    for ii = 1, jj = 2:NR
        S_r[ii, jj] /= 2.0
    end
    for ii = NZ, jj = 2:NR
        S_r[ii, jj] /= 2.0
    end

    species.N_e = 0
    species.var"##ps#498" = 0.0
    apply_particle_loaders!()
    
end
function timestep!()
    fill!(J_z, 0.0)
    fill!(J_r, 0.0)
    
    apply_particle_sources!()
    push_particles!(var"##ks#499")
    
    p = 1
    while p <= species.N_e
        reflected = false
        absorbed = false
        wrapped = false
        if (x_e[p])[1] < 0.0
            absorbed = true
        end
        if (x_e[p])[1] > 0.03
            absorbed = true
        end
        if (x_e[p])[2] < 0.0
            absorbed = true
        end
        if (x_e[p])[2] > 0.014
            absorbed = true
        end
        if absorbed
            remove_particle!(var"##ks#499", p)
            continue
        end
        p += 1
    end
    
    update_magnetic!()
    apply_electric_sources!()
    update_electric!(var"##mat#500", var"##nod#501")
    apply_displacement_current!(var"##mat#500", var"##nod#501")
    update_electric!(var"##mat#502", var"##nod#503")
    apply_displacement_current!(var"##mat#502", var"##nod#503")
    update_electric!(var"##int#546", var"##edg#547")
    update_electric_pec!(var"##edg#505")
    update_electric_pec!(var"##edg#507")
    update_electric_pec!(var"##edg#509")
    update_electric_pec!(var"##edg#511")
    update_electric_pec!(var"##edg#513")
    update_electric_src!(var"##edg#515", 8.85418781e-12, Inf)
    update_electric_pec!(var"##edg#517")
    update_electric_pec!(var"##edg#519")
    update_electric_pec!(var"##edg#521")
    update_electric_pec!(var"##edg#523")
    update_electric_pec!(var"##edg#525")
    update_electric_pec!(var"##edg#527")
    update_electric_pec!(var"##edg#529")
    update_electric_pec!(var"##edg#531")
    update_electric_pec!(var"##edg#533")
    update_electric_pec!(var"##edg#535")
    update_electric_pec!(var"##edg#537")
    update_electric_pmc!(var"##edg#539")
    update_electric_pec!(var"##edg#541")
    update_electric_pec!(var"##edg#543")
    update_electric_pec!(var"##edg#545")
    
    interpolate_electric!()
    interpolate_magnetic!()
    state.time += DT
end
end