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

function Solve_model()
    # initialize primitives and results
    prim = Primitives()
    val_func, pol_func = zeros(prim.nk, prim.nZ), zeros(prim.nk, prim.nZ)
    res = Results(val_func, pol_func)

    # initialize error and loop counter
    error, n = 100, 0

    # iterate value function until convergence
    while error > eps()
        n += 1

        # update value function guess
        v_next = Bellman(prim, res)

        # compute error
        error = maximum(abs.(v_next .- res.val_func))

        # update results with new guess at value function
        res.val_func = v_next

        if mod(n, 5000) == 0 || error < eps()
            println(" ")
            println("*************************************************")
            println("AT ITERATION = ", n)
            println("MAX DIFFERENCE = ", error)
            println("*************************************************")
        end
    end
    prim, res
end

#Bellman operator. Note the lack of type declarations in the function -- another exaple of sub-optimal coding
function Bellman(prim::Primatives, res::results)
    @unpack β, θ, δ, Π, k_grid, nk, Z_grid, nZ = prim #unpack primitive structure

    v_next = zeros(nk, nZ) # preallocate next value function

    for j_Z = 1:nZ # loop over productivity states
        Z = Z_grid[j_Z]
        for i_k = 1:nk #loop over capital state space
            max_util = -1e10 #something crappy
            k = k_grid[i_k] #convert state indices to state values
            budget = Z*k^θ + (1-δ)*k #budget given current state

            for i_kp = 1:nk #loop over choice of k'
                kp = k_grid[i_kp]
                c = budget - kp #consumption

                if c>0 #check to make sure that consumption is positive
                    # current state indexes the ROW of the transition matrix
                    # ⋅ is short for dot() - dot product to compute expectation
                    val = log(c) + β * (res.val_func[i_kp, 1:nZ] ⋅ Π[j_Z, 1:nZ])
                    if val>candidate_max #check for new max value
                        max_util = val
                        res.pol_func[i_kp,j_Z] = kp #update policy function
                    end
                end
            end
            v_next[i_k, j_Z] = max_util #update next guess of value function
        end
    end
    v_next # return updated value function
end


#############
