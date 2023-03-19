module LowChowRankPolynomial

using HomotopyContinuation, DynamicPolynomials

#generates a random number between -n and n
function genrand(n)
    return rand()*n*2 - n
end

#creates a random polynomial with coeffcients between -n and n and with varsnum number of variables
function get_random_polynomial(varsnum,n)
    @polyvar x[1:varsnum]
    p = genrand(n)
    for i in 1:varsnum
        p += genrand(n) * x[i]^2
        p += genrand(n) * x[i]
        for j in i+1:varsnum
            p += genrand(n) * x[i] * x[j]
        end
    end
    
    return p,x
end

#takes the norm of a polynomial by squarring the coefficients
function get_norm(p)
    normop = norm(MultivariatePolynomials.coefficients(p))
    return normop
end



function approximate_chow_polynomial(P, x, epsilon; L=false)
    
    if L
        return lagrange(P, x, epsilon)
    end
    
    #setup the starting variables
    v = nvariables(P)
    R = Any[]
    
    #setup our variables to create the function to minimise and set a1=1
    @var a[1:v] b[1:v+1]
    
    #get the equations for our function to minimise
    alpha = x[1] + sum([a[i]*x[i+1] for i in 1:v-1]) + a[v]
    beta = sum([b[i]*x[i] for i in 1:v]) + b[v+1]
    ab = alpha*beta
    
    #find our a1,...,bn coefficients
    cab = coefficients(ab)
    #Get our coeffs from originalpolynomial
    while get_norm(P) > epsilon
        cP = MultivariatePolynomials.coefficients(P)
        c = length(cP)

        #setup the function to minimise
        f = sum([(cP[i] - (cab[i]))^2 for i in 1:c])

        #get the gradient
        J = differentiate(f, vcat(a,b)) # vcat(a,b) : concat of vars a and b

        #set a system
        system = System(J; variables = vcat(a,b))

        #solve the system and get the real solutions
        result = solve(system; show_progress= false)
        real_sols = real_solutions(result; tol=1e-5)

        #test if there are results, if not return false
        if real_sols == []
            return false
        end

        #find the minimum point using the function that we want to minimize 
        _, minindex = findmin(map(s -> evaluate(f, a=>s[1:v], b=>s[v+1:(v*2)+1]), real_sols))
        minarg = real_sols[minindex]

        #setup the polynomial output
        alphaout = x[1] + sum([x[i+1]*minarg[i] for i in 1:v-1]) + minarg[v]
        betaout = sum([x[i-v]*minarg[i] for i in v+1:(v*2)]) + minarg[(v*2)+1]

        Q = alphaout * betaout
        P = P-Q
        push!(R,(alphaout, betaout)) 
    end    
    return R
    
end


function lagrange(P, x, epsilon)
    
    R = Any[]
    
    #Using HomotopyContinuation to find optimal solution
    @var c1 c2 c3 c4 c5 c6
    @var u[1:6] λ[1:1]
    f = c1*c4^2 - 4*c1*c5*c6 + c2^2*c5 - c2*c3*c4 + c3^2*c6

    J = differentiate([f], [c1,c2,c3,c4,c5,c6])
    C = System([[c1,c2,c3,c4,c5,c6] - u - J'*λ; f], variables = [c1;c2;c3;c4;c5;c6;λ], parameters = u)
    #We get the solutions
    
    while get_norm(P) > epsilon
        #Put in u0 the coefs we have for our point
        u₀ = MultivariatePolynomials.coefficients(P, [x[1]^2, x[1], x[1]*x[2], x[2], x[2]^2, 1])
        
        solution = HomotopyContinuation.solve(C; target_parameters = u₀, show_progress= false)
        real_sols = real_solutions(solution; tol=1e-5)
        #stops if it cant find real solutions anymore to avoid an error
        if real_sols == []
            return false, 0
        end

        #We take only the real solutions
        ed_points = map(p -> p[1:6], real_sols)


        #find optimal c1,...,c6 
        _, idx = findmin([norm(x - u₀) for x in ed_points])  
        c1d,c2d,c3d,c4d,c5d,c6d = ed_points[idx]

        #use our minimise function to find the a1...b3
        @var a1 a2 a3 b1 b2 b3
        minf = (c1d - a1*b1)^2 + (c2d - a1*b3 - a3*b1)^2 + (c3d - a1*b2 - a2*b1)^2 + (c4d - a2*b3 - a3*b2)^2 + (c5d - a2*b2)^2 + (c6d - a3*b3)^2

        #find the gradient
        J = differentiate(minf, [a1,a2,a3,b1,b2,b3])
        #push in J the linear function
        eq = genrand(5)*a1 + genrand(5)*a2 + genrand(5)*a3 + 1
        push!(J, eq)

        #set the system and find the results
        system = System(J; variables = [a1,a2,a3,b1,b2,b3])

        result = HomotopyContinuation.solve(system; show_progress= false)

        real_sols = real_solutions(result; tol=1e-5)


        function mf(v)
            a1, a2, a3, b1, b2, b3 = v
            return (c1d - a1*b1)^2 + (c2d - a1*b3 - a3*b1)^2 + (c3d - a1*b2 - a2*b1)^2 + (c4d - a2*b3 - a3*b2)^2 + (c5d - a2*b2)^2 + (c6d - a3*b3)^2
        end
        #get the minimum solution and output
        minval, minindex = findmin(map(s -> mf(s[1:6]), real_sols))

        a = real_sols[minindex][1:6]

        alphaout = (a[1] * x[1] + a[2] * x[2] + a[3])
        betaout = (a[4] * x[1] + a[5] * x[2] + a[6])
        
        Q = alphaout * betaout
        P = P-Q
        push!(R,(alphaout, betaout))
    end
    
    return R
    
end


end 
