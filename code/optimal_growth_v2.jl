# Import required packages
using Parameters, Plots, LinearAlgebra

# Create primitives
@with_kw struct Primitives
    # Define global constants
    β::Float64 = 0.99
    θ::Float64 = 0.36
    δ::Float64 = 0.025
    k_grid::Array{Float64,1} = collect(range(1.0, length = 50, stop = 45.0))
    nk::Int64 = length(k_grid)
    markov::Array{Float64,2} = [0.977 0.023; 0.074 0.926]
    z_grid::Array{Float64,1} = [1.25, 0.2]
    nz::Int64 = length(z_grid)
end

# Structure for results
mutable struct Results
    # Value and policy functions
    value_func::Array{Float64,2}
    policy_func::Array{Float64,2}
end

# Functionality to solve model
function Solve_model()
    # Initialize primitives, results
    prim = Primitives()
    value_func = zeros(prim.nk, prim.nz)
    policy_func = zeros(prim.nk, prim.nz)
    res = Results(value_func, policy_func)

    # Initialize error, counter
    error = 100 
    n = 0

    # Loop until convergence
    while error > eps()
        # Increment counter
        n += 1

        # Call Bellman operator and update
        v_next = Bellman(prim, res)
        error = maximum(abs.(v_next .- res.value_func))
        res.value_func = v_next
    end

    # Print progress
    println("Value function converged in $n iterations!")

    # Return values
    prim, res
end

#Bellman operator
function Bellman(prim::Primitives, res::Results)
    # Upack primitive structure, instantiate value function
    @unpack β, δ, θ, nz, nk, z_grid, k_grid, markov = prim
    v_next = zeros(nk, nz)

    # Iterate over state space
    for (i_k, k) in enumerate(k_grid)
        for (i_z, z) in enumerate(z_grid)
            # Candidate maximum value, budget constraint
            max_util = -1e10
            budget = z * k^θ + (1 - δ) * k 

            # Iterate over next period capital choice
            for (i_kp, kp) in enumerate(k_grid)
                # Find consummption
                c = budget - kp

                # Check positivity
                if c>0
                    # Compute value
                    val = log(c) + β * (res.value_func[i_kp,:] ⋅ markov[i_z, :])

                    # Check maximum
                    if val > max_util
                        # Update values
                        max_util = val
                        res.policy_func[i_k, i_z] = kp
                    end
                end
            end

            # Update next iteration
            v_next[i_k, i_z] = max_util 
        end
    end
    v_next
end

# Check functionality
@elapsed prim, res = Solve_model()

# Plot functions
p1 = plot(prim.k_grid, res.value_func, title="Value Functions", label = ["High Productivity" "Low Productivity"])
p2 = plot(prim.k_grid, res.policy_func, title="Policy Functions", label = ["High Productivity" "Low Productivity"])
Plots.plot(p1, p2, layout = (2,1), legend=:none)