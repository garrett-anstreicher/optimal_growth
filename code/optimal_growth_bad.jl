
using Parameters
#struct to hold model primitives
@with_kw struct Primitives
    β::Float64 = 0.99 #Discount factor
    θ::Float64 = 0.36 #Production
    δ::Float64 = 0.025 #Depreciation
    trans::Array{Float64,2} = [0.977 0.023; 0.074 0.926] #Transition matrix: [GG GB; BG BB]
    Z_g = 1.25 #Good productivity
    Z_b = 0.20 #Bad productivity
    k_grid::Array{Float64,1}= collect(range(0.01, length = 1000, stop = 45.0)) #Capital Grid
    nk::Int64 = length(k_grid) #Number of capital grid states
end

mutable struct Results
    val_func::Array{Float64,2} #value function nk by 2 for the two states
    pol_func::Array{Float64,2} #policy function nk by 2 for the two states
end


function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, θ, nk, trans, Z_g, Z_b, k_grid = prim #Unpack the primitive structure
    @unpack val_func, pol_func = res #Unpack the value and policy functions

    v_next = zeros(nk, 2) #Preallocate next guess of value function

    for i_k = 1:nk #loop over state space
        candidate_maxG = -1e10 #something crappy for good state
        candidate_maxB = -1e10 #something crappy for bad state
        k = k_grid[i_k] #convert state indices to state values
        budgetG = Z_g*k^θ + (1-δ)*k #budget given good productivity and current capital
        budgetB = Z_b*k^θ + (1-δ)*k #budget given bad productivity and current capital

        for i_kp = 1:nk #loop over choice of k_prime
            kp = k_grid[i_kp] #value of k'
            cG = budgetG - kp #consumption with good productivity
            cB = budgetB - kp #consumption with bad productivity
            if cG>0 #check to make sure that consumption is positive
                valG = log(cG) + β * (trans[1, 1] * val_func[i_kp, 1] + trans[1, 2] * val_func[i_kp, 2]) #compute utility
                if valG>candidate_maxG #check for new max value
                    candidate_maxG = valG #reset max value
                    pol_func[i_k, 1] = kp #update policy function
                end
            end
            if cB>0 #check to make sure that consumption is positive
                valB = log(cB) + β * (trans[2, 1] * val_func[i_kp, 1] + trans[2, 2] * val_func[i_kp, 2]) #Compute utility
                if valB>candidate_maxB #check for new max value
                    candidate_maxB = valB #reset max value
                    pol_func[i_k, 2] = kp #update policy function
                end
            end
        end
        v_next[i_k, 1] = candidate_maxG #update next guess of value function for good prod.
        v_next[i_k, 2] = candidate_maxB #update next guess of value function for bad prod.
    end
    v_next
end

function Solve_model()
    prim = Primitives() #Initialize primitives
    val_func, pol_func = zeros(prim.nk, 2), zeros(prim.nk, 2) #Initialize results
    res = Results(val_func, pol_func)

    error, n = 100, 0
    while error >1E-4 #loop until convergence
        n+=1
        v_next = Bellman(prim, res) #Next guess of value function
        error = maximum(abs.(v_next-res.val_func)) #check for convergence
        res.val_func = v_next #update
        println("Current error: ", error)
    end
    #println("Value function converged in ", n, " iterations.")
    prim, res
end

using Plots
@elapsed prim, res = Solve_model()
Plots.plot(prim.k_grid, res.val_func lab = ["V_G" "V_B]")
