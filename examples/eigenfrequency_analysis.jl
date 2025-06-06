import PlasmaModelingToolkit.Units: GHz
using ProgressMeter
using CiSE_klystron_2025: EIG_solver

EIG_solver.initialize!()

RF_DRIVE_FREQ = 5.087GHz
TEND = 200.0 / RF_DRIVE_FREQ

dt = EIG_solver.DT
nt = ceil(Int64, TEND / dt)
probe = zeros(nt)
enere = zeros(nt)
enerh = zeros(nt)
i, j = floor(Int64, EIG_solver.NZ / 2), floor(Int64, EIG_solver.NR / 3)

@showprogress for it in 1:nt
    EIG_solver.timestep!()
    probe[it] = EIG_solver.E_z[i,j]
end

using FFTW, Statistics
signal = probe
samples = length(signal)
spectrum = fft(signal .- mean(signal))
frequency = fftfreq(samples, 1.0 / dt)
magnitude = abs.(spectrum) ./ maximum(abs, spectrum)
magnitude[frequency .> +20.0GHz] .= 0.0
magnitude[frequency .< -20.0GHz] .= 0.0

f_c = frequency[argmax(magnitude)]
@show f_c