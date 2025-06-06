import PlasmaModelingToolkit.Units: GHz
using ProgressMeter
using CiSE_klystron_2025: EM_solver, EM_PIC_solver, ES_solver


EM_solver.initialize!()

RF_DRIVE_FREQ = 5.087GHz
TEND = 200.0 / RF_DRIVE_FREQ


nt = ceil(Int64, TEND / EM_solver.DT)
probe = zeros(nt)
enere = zeros(nt)
enerh = zeros(nt)
i, j = floor(Int64, EM_solver.NZ / 2), floor(Int64, EM_solver.NR / 3)

@showprogress for it in 1:nt
    EM_solver.timestep!()
    probe[it] = EM_solver.E_z[i,j]
    enere[it] = EM_solver.calculate_electric_energy()
    enerh[it] = EM_solver.calculate_magnetic_energy()
end

using LsqFit
times = range(0.0, EM_solver.state.time, length=nt)
energ = enere + enerh
mask = times .>= 10e-9
p0 = [10.0, 20.0]
model(t, p) = p[1] * exp.(-2π * RF_DRIVE_FREQ * t / p[2])
fit = curve_fit(model, times[mask], 1e7energ[mask], p0)
_, Q_cold = fit.param
@show Q_cold