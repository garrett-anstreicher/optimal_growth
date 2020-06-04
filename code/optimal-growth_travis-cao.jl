using Parameters, Plots #read in necessary packages

#global variables instead of structs
β = 0.99 #discount rate.
θ = 0.36 #capital share
δ = 0.025 #capital depreciation
k_grid = collect(range(1.0, length = 100, stop = 75.0)) #capital grid
nk = length(k_grid) #number of capital elements
markov = [0.977 0.023; 0.074 0.926] #markov transition process
z_grid = [1.25, 0.2] #productivity state grid
nz = length(z_grid) #number of productivity states

#initialize value function and policy functions, again as globals.
val_func = zeros(nk, nz)
pol_func = zeros(nk, nz)

#Bellman operator. Note the lack of type declarations inthe function -- another exaple of sub-optimal coding
function Bellman(val_func, pol_func)
    v_next = zeros(nk, nz)

    for i_k = 1:nk, i_z = 1:nz #loop over state space
        candidate_max = -1e10 #something crappy
        k, z = k_grid[i_k], z_grid[i_z] #convert state indices to state values
        budget = z*k^θ + (1-δ)*k #budget given current state. Doesn't this look nice?

        for i_kp = 1:nk #loop over choice of k_prime
            kp = k_grid[i_kp]
            c = budget - kp #consumption
            if c>0 #check to make sure that consumption is positive
                val = log(c) + β * sum(val_func[i_kp,:].*markov[i_z, :])
                if val>candidate_max #check for new max value
                    candidate_max = val
                    pol_func[i_k, i_z] = kp #update policy function
                end
            end
        end
        v_next[i_k, i_z] = candidate_max #update next guess of value function
    end
    v_next, pol_func
end

#more bad globals
error = 100
n = 0
tol = 1e-3
while error>tol
    global n, val_func, error, pol_func #declare that we're using the global definitions of these variables in this loop
    n+=1
    v_next, pol_func = Bellman(val_func, pol_func)
    error = maximum(abs.(val_func - v_next)) #reset error term
    val_func = v_next #update value function held in results vector
end
println("Value function converged in ", n, " iterations.")

#############
