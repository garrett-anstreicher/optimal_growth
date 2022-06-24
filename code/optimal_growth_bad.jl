using Parameters, Plots #read in necessary packages

#global variables instead of structs
β = 0.99 #discount rate.
θ = 0.36 #capital share
δ = 0.025 #capital depreciation
Pi = [0.977  0.023; 0.074  0.926]
prod_mat = [1.25, 0.2]
k_grid = collect(range(1.0, length = 1800, stop = 45.0)) #capital grid
nk = length(k_grid) #number of capital elements

#initialize value function and policy functions, again as globals.
val_func = zeros(nk,2)
pol_func = zeros(nk,2)

#Bellman operator. Note the lack of type declarations inthe function -- another exaple of sub-optimal coding
function Bellman(val_func, pol_func, prod_ind)
    v_next = zeros(nk,2)

    for i_k = 1:nk #loop over state space
        candidate_max = -1 #something crappy
        k = k_grid[i_k]#convert state indices to state values
        prod = prod_mat[prod_ind] #convert current productivity index to productivity value
        budget = prod*k^θ + (1-δ)*k #budget given current state. Doesn't this look nice?

        for i_kp = 1:nk #loop over choice of k_prime
            kp = k_grid[i_kp]
            c = budget - kp #consumption
            if c>0 #check to make sure that consumption is positive
                val = log(c)
                for i in 1:length(prod_mat)
                    val += β*Pi[prod_ind,i]*val_func[i_kp, i]
                end
                if val>candidate_max #check for new max value
                    candidate_max = val
                    pol_func[i_k,prod_ind] = kp #update policy function
                end
            end
        end
        v_next[i_k,prod_ind] = candidate_max #update next guess of value function
    end
    return v_next[:,prod_ind], pol_func
end

#more bad globals
error = 100
n = 0
tol = 1e-4
while error>tol
    global n, val_func, error, pol_func #declare that we're using the global definitions of these variables in this loop
    n+=1
    v_next=zeros(nk,2)
    for i in 1:length(prod_mat)
        v_next[:,i], pol_func = Bellman(val_func, pol_func, i)
    end
    error = maximum(abs.(val_func - v_next)) #reset error term
    val_func = v_next
    #update value function held in results vector
    println(n, "  ",  error)
end
println("Value function converged in ", n, " iterations.")

#############
