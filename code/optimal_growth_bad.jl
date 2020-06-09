using Parameters, Plots #read in necessary packages

#Initialize structs
@with_kw mutable struct Primitives
    β = 0.99
    θ = 0.36
    δ = 0.025
    pi = [0.977 0.023; 0.074 0.926]
    z = [1.25 0.2]
    k_grid = collect(range(0.01,length=1000,stop=45))
    nk = length(k_grid)
end

mutable struct Results
    val_func
    pol_func
end

#Bellman operator. Note the lack of type declarations inthe function -- another exaple of sub-optimal coding
function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, θ, nk, k_grid, z, pi = prim
    @unpack val_func, pol_func = res
    v_next = zeros(nk,2)

    for i_k = 1:nk #loop over state space
        candidate_max = -1e10 #something crappy
        k = k_grid[i_k]#convert state indices to state values

        for i_pi = 1:2
            ζ = z[i_pi]
            budget = ζ*k^θ + (1-δ)*k #budget given current state. Doesn't this look nice?

            for i_kp = 1:nk #loop over choice of k_prime
                kp = k_grid[i_kp]
                c = budget - kp #consumption
                if c>0 #check to make sure that consumption is positive
                    val = log(c) + β * (pi[i_pi,1]*val_func[i_kp,1] + pi[i_pi,2]*val_func[i_kp,2])
                    if val>candidate_max #check for new max value
                        candidate_max = val
                        pol_func[i_k, i_pi] = kp #update policy function
                    end
                end
            end
            v_next[i_k, i_pi] = candidate_max #update next guess of value function
        end
    end
    v_next
end

function Solve_model()
    prim = Primitives()         # keeping all the default values
    val_func, pol_func = zeros(prim.nk,2), zeros(prim.nk,2)
    res = Results(val_func, pol_func)
    error = 100.0
    n = 0
    tol = 1e-4
    while error>tol
        n+=1
        v_next = Bellman(prim, res)
        error = maximum(abs.(res.val_func .- v_next)) #reset error term
        res.val_func = v_next #update value function held in results vector
        println(n, "  ",  error)
    end
    println("Value function converged in ", n, " iterations.")
end

@elapsed prim, res = Solve_model()


#############
