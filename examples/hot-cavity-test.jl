import PlasmaModelingToolkit.Units: GHz
using ProgressMeter
using CiSE_klystron_2025: EM_PIC_solver, ES_solver

EM_PIC_solver.initialize!()

RF_DRIVE_FREQ = 5.087GHz
TEND = 200.0 / RF_DRIVE_FREQ

B_SOLENOID = 1.3
EM_PIC_solver.b_z .= -B_SOLENOID;

nt = ceil(Int64, TEND / EM_PIC_solver.DT)
probe = zeros(nt)
parts = zeros(nt)
energ = zeros(nt)
diver = zeros(nt)
i, j = floor(Int64, EM_PIC_solver.NZ / 2), floor(Int64, EM_PIC_solver.NR / 3)

@showprogress for it in 1:nt
    EM_PIC_solver.timestep!()

    EM_PIC_solver.compute_divergence!()
    diver[it] = maximum(abs, EM_PIC_solver.div_E)
    if EM_PIC_solver.state.time > 10e-9 && maximum(abs, EM_PIC_solver.div_E) > 1e7
        copy!(ES_solver.ρ, EM_PIC_solver.div_E)
        ES_solver.solve!()
        copy!(EM_PIC_solver.E_z, EM_PIC_solver.E_z - ES_solver.E_z)
        copy!(EM_PIC_solver.E_r, EM_PIC_solver.E_r - ES_solver.E_r)
    end

    energ[it] = EM_PIC_solver.calculate_electric_energy() + EM_PIC_solver.calculate_magnetic_energy()
    probe[it] = EM_PIC_solver.H_θ[i,j]
    parts[it] = EM_PIC_solver.species.N_e
end

using LsqFit
times = range(0.0, EM_PIC_solver.state.time, length=nt)
energ = enere + enerh
mask = times .>= 10e-9
p0 = [10.0, 20.0]
model(t, p) = p[1] * exp.(-2π * RF_DRIVE_FREQ * t / p[2])
fit = curve_fit(model, times[mask], 1e7energ[mask], p0)
_, Q_hot = fit.param
@show Q_hot