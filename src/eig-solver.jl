module EIG_solver

mutable struct State
    time::Float64
end


using SpecialFunctions
using OffsetArrays

gzeros(T, n, m) = OffsetArray(zeros(T, n + 2, m + 2), -1, -1)

import PlasmaModelingToolkit.Constants: c
import PlasmaModelingToolkit.Constants: ε_0
import PlasmaModelingToolkit.Constants: μ_0
import PlasmaModelingToolkit.Constants: η_0
import PlasmaModelingToolkit.Constants: χ_01
import JLD2

import PlasmaModelingToolkit.InterfaceConditions: DielectricInterface
import PlasmaModelingToolkit.Materials: Medium

struct Nodes{Z1, Z2, R1, R2}
    mask :: Matrix{Bool}
end

struct Edges{A, B, C, U, V} end

const NZ = 61
const NR = 29
const DZ = 0.0005
const DR = 0.0005
const DT = min(DZ, DR) / (√2 * c)
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

const var"##bcs#475" = JLD2.load("data/eig-data.jld2", "##bcs#475")
const var"##edg#442" = JLD2.load("data/eig-data.jld2", "##edg#442")
const var"##edg#444" = JLD2.load("data/eig-data.jld2", "##edg#444")
const var"##bcs#481" = JLD2.load("data/eig-data.jld2", "##bcs#481")
const var"##mat#437" = JLD2.load("data/eig-data.jld2", "##mat#437")
const var"##edg#484" = JLD2.load("data/eig-data.jld2", "##edg#484")
const var"##edg#446" = JLD2.load("data/eig-data.jld2", "##edg#446")
const var"##nod#440" = JLD2.load("data/eig-data.jld2", "##nod#440")
const var"##edg#448" = JLD2.load("data/eig-data.jld2", "##edg#448")
const var"##edg#450" = JLD2.load("data/eig-data.jld2", "##edg#450")
const var"##edg#452" = JLD2.load("data/eig-data.jld2", "##edg#452")
const var"##edg#454" = JLD2.load("data/eig-data.jld2", "##edg#454")
const var"##edg#456" = JLD2.load("data/eig-data.jld2", "##edg#456")
const var"##edg#458" = JLD2.load("data/eig-data.jld2", "##edg#458")
const var"##nod#438" = JLD2.load("data/eig-data.jld2", "##nod#438")
const var"##int#483" = JLD2.load("data/eig-data.jld2", "##int#483")
const var"##edg#460" = JLD2.load("data/eig-data.jld2", "##edg#460")
const var"##bcs#451" = JLD2.load("data/eig-data.jld2", "##bcs#451")
const var"##edg#462" = JLD2.load("data/eig-data.jld2", "##edg#462")
const var"##edg#464" = JLD2.load("data/eig-data.jld2", "##edg#464")
const var"##edg#466" = JLD2.load("data/eig-data.jld2", "##edg#466")
const var"##edg#468" = JLD2.load("data/eig-data.jld2", "##edg#468")
const var"##edg#470" = JLD2.load("data/eig-data.jld2", "##edg#470")
const var"##edg#472" = JLD2.load("data/eig-data.jld2", "##edg#472")
const var"##edg#474" = JLD2.load("data/eig-data.jld2", "##edg#474")
const var"##edg#476" = JLD2.load("data/eig-data.jld2", "##edg#476")
const var"##edg#478" = JLD2.load("data/eig-data.jld2", "##edg#478")
const var"##mat#439" = JLD2.load("data/eig-data.jld2", "##mat#439")
const var"##edg#480" = JLD2.load("data/eig-data.jld2", "##edg#480")
const var"##edg#482" = JLD2.load("data/eig-data.jld2", "##edg#482")


function apply_electric_source!(::Edges{Z1, Z2, R0, 0, _}, I_s) where {Z1, Z2, R0, _}
  @inbounds for i=Z1:(Z2-1)
    J_z[i, R0] = I_s / S_z[i, R0]
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

    
function calculate_electric_energy()        
    U_E = 0.0   
    U_E += calculate_electric_energy(var"##mat#437", var"##nod#438")
    U_E += calculate_electric_energy(var"##mat#439", var"##nod#440")

    return U_E
end
    
function calculate_magnetic_energy()        
    U_H = 0.0
    U_H += calculate_magnetic_energy(var"##mat#437", var"##nod#438")
    U_H += calculate_magnetic_energy(var"##mat#439", var"##nod#440")
        
    return U_H
end

function initialize!()
    state.time = 0.0
    
    fill!(E_z, 0.0)
    fill!(E_r, 0.0)
    fill!(H_θ, 0.0)
    H_θ .= randn(NZ - 1, NR - 1)
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
    end
end

function timestep!()
    update_magnetic!()
    update_electric!(var"##mat#437", var"##nod#438")
    apply_displacement_current!(var"##mat#437", var"##nod#438")
    update_electric!(var"##mat#439", var"##nod#440")
    apply_displacement_current!(var"##mat#439", var"##nod#440")
    update_electric!(var"##int#483", var"##edg#484")
    update_electric_pec!(var"##edg#442")
    update_electric_pec!(var"##edg#444")
    update_electric_pec!(var"##edg#446")
    update_electric_pec!(var"##edg#448")
    update_electric_pec!(var"##edg#450")
    update_electric_src!(var"##edg#452", 8.85418781e-12, Inf)
    update_electric_pec!(var"##edg#454")
    update_electric_pec!(var"##edg#456")
    update_electric_pec!(var"##edg#458")
    update_electric_pec!(var"##edg#460")
    update_electric_pec!(var"##edg#462")
    update_electric_pec!(var"##edg#464")
    update_electric_pec!(var"##edg#466")
    update_electric_pec!(var"##edg#468")
    update_electric_pec!(var"##edg#470")
    update_electric_pec!(var"##edg#472")
    update_electric_pec!(var"##edg#474")
    update_electric_pmc!(var"##edg#476")
    update_electric_pec!(var"##edg#478")
    update_electric_pec!(var"##edg#480")
    update_electric_pec!(var"##edg#482")
    
    state.time += DT
end
end