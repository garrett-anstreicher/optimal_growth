using Parameters, Plots #read in necessary packages

#global variables instead of structs
@with_kw struct Primitives
    β::Float64 = 0.99 #discount rate.
    θ::Float64 = 0.36 #capital share
    δ::Float64 = 0.025 #capital depreciation
    Π::Array{Float64, 2} = [0.977 0.023; 0.074 0.926] # Markov transition matrix
    k_grid::Array{Float64,1} = collect(range(1.0, length = 1800, stop = 45.0)) #capital grid
    Z_grid::Array{Float64,1} = [1.25; 0.2] # Productivity states
    nk::Int64 = length(k_grid) #number of capital elements
    nZ::Int64 = length(Z_grid)
end

#initialize value function and policy functions, again as globals.
mutable struct Results
    val_func::Array{Float64,2}
    pol_func::Array{Float64,2}
end

#Bellman operator. Note the lack of type declarations in the function -- another exaple of sub-optimal coding
function Bellman(prim::Primatives, res::results)
    @unpack β, θ, δ, Π, k_grid, nk, Z_grid, nZ = prim #unpack primitive structure

    v_next = zeros(nk, nZ) # preallocate next value function

    for j_Z = 1:nZ # loop over productivity states
        Z = Z_grid[j_Z]
        for i_k = 1:nk #loop over capital state space
            max_util = -1e10 #something crappy
            k = k_grid[i_k]#convert state indices to state values
            # Need to add state-specific productivity here
            budget = Z*k^θ + (1-δ)*k #budget given current state. Doesn't this look nice?

            for i_kp = 1:nk #loop over choice of k'
                kp = k_grid[i_kp]
                c = budget - kp #consumption

                if c>0 #check to make sure that consumption is positive
                    # current state indexes the ROW of the transition matrix
                    # ⋅ is short for dot() - dot product to compute expectation
                    val = log(c) + β * (res.val_func[i_kp, 1:nZ] ⋅ Π[j_Z, 1:nZ])
                    if val>candidate_max #check for new max value
                        max_util = val
                        res.pol_func[i_kp] = kp #update policy function
                    end
                end
            end
            v_next[i_k, j_Z] = max_util #update next guess of value function
        end
    end
    v_next # return updated value function
end

#more bad globals
error = 100
n = 0
tol = 1e-4
while error>tol
    global n, val_func, error, pol_func #declare that we're using the global definitions of these variables in this loop
    n+=1
    v_next, pol_func = Bellman(val_func, pol_func)
    error = maximum(abs.(val_func - v_next)) #reset error term
    val_func = v_next #update value function held in results vector
    println(n, "  ",  error)
end
println("Value function converged in ", n, " iterations.")

#############
