This Project explores one particular method for the optimization problem, the Newton's Method. We compare the pertubation of the vanilla
method vs the safeguarded method, and demonstrating that the safeguard we apply is indeed effective and saving the convergence of certain
initial conditions that would have diverged otherwise. The 2 safeguard methods are taking the modified LDL decomposition of the Hessian, 
which makes it into a positive definite version of itself hence guaranteeing descent, and line-search, which tries to reduce the step size 
if the function value were not decreasing as desired. 