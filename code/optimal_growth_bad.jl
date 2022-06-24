###hello everybody!


using Parameters, Plots #read in necessary packages

#global variables instead of structs
β = 0.99 #discount rate.
θ = 0.36 #capital share
δ = 0.025 #capital depreciation
k_grid = collect(range(1.0, length = 1800, stop = 45.0)) #capital grid
nk = length(k_grid) #number of capital elements

#initialize value function and policy functions, again as globals.
val_func = zeros(nk)
pol_func = zeros(nk)

#Bellman operator. Note the lack of type declarations inthe function -- another exaple of sub-optimal coding
function Bellman(val_func, pol_func)
    v_next = zeros(nk)

    for i_k = 1:nk #loop over state space
        candidate_max = -1e10 #something crappy
        k = k_grid[i_k]#convert state indices to state values
        budget = k^θ + (1-δ)*k #budget given current state. Doesn't this look nice?

        for i_kp = 1:nk #loop over choice of k_prime
            kp = k_grid[i_kp]
            c = budget - kp #consumption
            if c>0 #check to make sure that consumption is positive
                val = log(c) + β * val_func[i_kp]
                if val>candidate_max #check for new max value
                    candidate_max = val
                    pol_func[i_k] = kp #update policy function
                end
            end
        end
        v_next[i_k] = candidate_max #update next guess of value function
    end
    v_next, pol_func
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
