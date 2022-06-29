# Import packages
using Parameters, Plots, LinearAlgebra

# Create primatives
@with_kw struct Primitives
    # Define global constants
    β::Float64 = 0.99 
    θ::Float64 = 0.36
    δ::Float64 = 0.025
    M::Array{Float64, 2} = [0.977 0.023; 0.074 0.926]
    k_grid::Array{Float64,1} = collect(range(1.0, length = 50, stop = 45.0))
    z_grid::Array{Float64,1} = [1.25, 0.2]
    nk::Int64 = length(k_grid)
    nz::Int64 = length(z_grid)
end

# Structure for results
mutable struct Results
    # Value and policy functions
    value_func::Array{Float64, 2}
    policy_func::Array{Float64, 2}
end

# Functionality to solve model
function Solve_model()
    # Initialize primitives
    prim = Primitives()
    value_func, policy_func = zeros(prim.nk, prim.nz), zeros(prim.nk, prim.nz)
    res = Results(value_func, policy_func)

    # Instantiate error and iteration; loop until convergence
    error, n = 100, 0
    while error > eps()
        # Increment counter
        n += 1

        # Call Bellman operator and update
        v_next = Bellman(prim, res)
        error = maximum(abs.(v_next .- res.value_func))
        res.value_func = v_next

        # Print error every so often
        if mod(n, 5000) == 0 || error < eps()
            println(" ")
            println("*************************************************")
            println("AT ITERATION = ", n)
            println("MAX DIFFERENCE = ", error)
            println("*************************************************")
        end
    end

    # Return values
    prim, res
end

# Bellman operator
function Bellman(prim::Primitives, res::Results)
    # Unpack primitive structure, instantiate value function
    @unpack β, θ, δ, M, k_grid, z_grid, nk, nz = prim
    v_next = zeros(nk, nz)

    # Iterate over state space and productivity
    for (i_k, k) in enumerate(k_grid)
        for (i_z, z) in enumerate(z_grid)
            # Candidate maximum value, budget constraint
            max_util = -1e10
            budget = k^θ + (1 - δ) * k

            # Iterate over next period capital choice
            for (i_kp, kp) in enumerate(k_grid)
                # Find consumption
                c = budget - kp

                # Check positivity
                if c > 0
                    # Compute value
                    val = log(c) + β * (res.value_func[i_kp, :] ⋅ M[i_z, :])

                    # Check maximum
                    if val > max_util
                        # Update values
                        max_util = val
                        res.policy_func[i_kp, i_z] = kp
                    end
                end
            end

            # Update next iteration
            v_next[i_k, i_z] = max_util
        end
    end

    # Return values
    v_next
end

# Check Functionality
@elapsed prim, res = Solve_model()
p1 = plot(prim.k_grid, res.value_func)
p2 = plot(prim.k_grid, res.policy_func)

# Plots
Plots.plot(p1, p2, layout = (2,1), legend=:none)