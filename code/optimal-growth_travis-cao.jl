# Computational Bootcamp
# Problem set 1, Question 7
# Travis Cao

# Console clearing in Juno
clearconsole()

# Change working directory to current file folder
cd(dirname(@__FILE__))

# Use necessary packages
using Parameters, Plots

# Struct variables and results
@with_kw struct Primitives
    β::Float64 = .99 # discount rate
    θ::Float64 = .36 # capital share
    δ::Float64 = .025 # capital depreciation rate
    k_grid::Array{Float64,1} = collect(range(.01, length = 1000, stop = 45.0)) # capital grid
    nk::Int64 = length(k_grid) # number of capital elements
    markov::Array{Float64,2} = [0.977 0.023; 0.074 0.926] # markov transition process
    z_grid::Array{Float64,1} = [1.25, 0.2] # productivity state grid
    nz::Int64 = length(z_grid) # number of productivity states
end

@with_kw mutable struct Results
    val_func::Array{Float64,2} = zeros(Primitives().nk, Primitives().nz)
    pol_func::Array{Float64,2} = zeros(Primitives().nk, Primitives().nz)
end

# Bellman operator
function Bellman(prim::Primitives, res::Results)
    @unpack β, θ, δ, k_grid, nk, markov, z_grid, nz = prim
    v_next = zeros(nk, nz)

    for i_k = 1:nk, i_z = 1:nz # loop over state space
        max_util = -Inf # some max_util initialization
        k, z = k_grid[i_k], z_grid[i_z] # convert state indices to state values
        budget = z*k^θ + (1-δ)*k #budget in the current state

        # find k_prime such that max util
        for i_kp = 1:nk
            kp = k_grid[i_kp]
            c = budget - kp # consumption
            if c > 0 # check to make sure that consumption is positive
                val = log(c) + β * sum(res.val_func[i_kp, :].*markov[i_z, :])
                if val > max_util # check for new max value
                    max_util = val
                    res.pol_func[i_k, i_z] = kp # update policy function
                end
            end
        end

        v_next[i_k, i_z] = max_util # update next guess of value function
    end

    return v_next
end

function solve_stochastic_model()
    prim = Primitives()
    res = Results()
    error, n, tol = Inf, 0, 1e-3

    while error > tol
        n += 1
        v_next = Bellman(prim, res)
        error = maximum(abs.(v_next .- res.val_func))
        res.val_func = v_next
    end

    println("Value function converged in ", n, " iterations.")
    return prim, res
end

@time primitives, results = solve_stochastic_model()
val_func = results.val_func
pol_func = results.pol_func

#### Value function converged in 669 iterations.
#### 135.648450 seconds (2.14 G allocations: 191.074 GiB, 8.54% gc time)
