### Question 7
# based on codes by Garrett Anstreicher
using Parameters, Plots #read in necessary packages

#changed from global variables to structs
@with_kw struct Primitives
    β::Float64 = 0.99 #discount rate.
    θ::Float64 = 0.36 #capital share
    δ::Float64 = 0.025 #capital depreciation
    z_g::Float64 = 1.25
    z_l::Float64 = 0.2
    z = [z_g; z_l]
    π_gg::Float64 = 0.977
    π_lg::Float64 = 0.023
    π_gl::Float64 = 0.074
    π_ll::Float64 = 0.926
    Π = [π_gg π_lg; π_gl π_ll]
    k_grid::Array{Float64,1} = collect(range(1.0, length = 100, stop = 75.0)) #capital grid
    nk::Int64 = length(k_grid) #number of capital elements
    nz::Int64 = 2
end

#initialize value function and policy functions, changed from globals to mutable struct, also type declarations added.
mutable struct Results
    val_func::Array{Float64,2} # the first column is for high type, and second for low type
    pol_func::Array{Float64,2}
end

#Bellman operator.
function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, θ, z_g, z_l, z, nk, nz, π_gg, π_lg, π_gl, π_ll, Π, k_grid = prim #unpack allows us to not need to use res. or prim.
    v_next = zeros(nk, nz)
    pol_func = zeros(nk, nz)
    for i_z = 1:nz
        i_zp::Int64 = abs(i_z-2)+1
        for i_k = 1:nk #loop over state space
            candidate_max = -1e10 #something crappy
            k = k_grid[i_k]#convert state indices to state values
            budget = z[i_z]*k^θ + (1-δ)*k #budget given current state. Doesn't this look nice?

            for i_kp = 1:nk #loop over choice of k_prime
                kp = k_grid[i_kp]
                c = budget - kp #consumption
                if c>0 #check to make sure that consumption is positive
                    val = log(c) + β * Π[i_z, i_z] * res.val_func[i_kp, i_z] + + β * Π[i_z, i_zp] * res.val_func[i_kp, i_zp]
                    if val>candidate_max #check for new max value
                        candidate_max = val
                        pol_func[i_k, i_z] = kp #update policy function
                    end
                end
            end
            v_next[i_k, i_z] = candidate_max #update next guess of value function
        end
    end
    return v_next, pol_func
end


function Solve_model()
    prim = Primitives()
    val_func, pol_func = zeros(prim.nk, prim.nz), zeros(prim.nk, prim.nz)  #initial guess
    res = Results(val_func, pol_func)   # results

    error, n = 100, 0 # initial error and number of iteration
    tol = 1e-4
    while error > tol
        n+=1
        v_next, pol_func = Bellman(prim, res)
        error = maximum(abs.(v_next .- res.val_func))
        res.val_func = v_next
        println("Current error: ", error)
    end
    println("Value function converged in ", n, " iterations")
    return prim, res
end

#############
@elapsed prim,res = Solve_model()

Plots.plot(prim.k_grid, res.val_func[:,1], title = "High type value function")
Plots.plot(prim.k_grid, res.val_func[:,2], title = "Low type value function")
