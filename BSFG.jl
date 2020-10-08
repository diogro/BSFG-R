using Statistics 
using Random
using Distributions
using LinearAlgebra
using BlockDiagonals
using Kronecker
using ProgressMeter

cholcov = function(x::Array{Float64, 2})
    tol = 1e-6
    if(isposdef(x))
        out = Array(cholesky(x).U)
    else
        eigX = eigen(x)
        val = sqrt.(eigX.values[eigX.values .> tol])
        vec = eigX.vectors[:,eigX.values .> tol]
        out = Diagonal(val) * vec'
    end
    out
end

function makeSVDdict(A::Array{Float64, 2}, B::Array{Float64, 2})
    U,V,Q,C,S,R = svd(cholcov(A), cholcov(B))
    #svd_Design_Ainv.Q = inv(q)'
    #svd_Design_Ainv. 
    #svd_Design_Ainv.s2 = diag(S2'*S2)
    #Qt_Design = svd_Design_Ainv.Q'*Design;    
    H = R * Q'
    q = H'
    Q = ("Q", inv(q)')
    s1 = ("s1", Array(diag(C' * C)))
    s2 = ("s2", Array(diag(S' * S)))
    Dict([Q, s1, s2])
end

function cov2cor(x::Array{Float64, 2})
    sds = sqrt.(diag(x))
    x ./ (sds * sds')
end

# --- Initialize variables --- #
#residual parameters. This structure holds the priors hyperparamters for
#the gamma prior on the model residual variances. It also holds the current
#estimate of the residual precision

mutable struct Residuals
    as::Float64
    bs::Float64
    Y::Array{Float64,2}
    p::Int64
    ps::Array{Float64,1}
end

#Factors. This struct holds all information about the latent factors
#including the current number of factors; priors hyperparameters for the
#factor Loadings; as well as current values of the factor loadings; their
#precision; the factor scores; & the genetic heritability of each factor()

mutable struct LatentFactors
    r::Int64            
    n::Int64
    p::Int64
    k::Int64
    df::Int64
    ad1::Float64
    bd1::Float64
    ad2::Float64
    bd2::Float64
    h2_divisions::Int64        #discretizations of heritability
    sp::Int64
    nrun::Int64

    psijh::Array{Float64,2}    #individual loadings precisions
    delta::Array{Float64,1}    # components of tauh
    tauh::Array{Float64,1}     #extra shrinkage of each loading column
    Plam::Array{Float64,2}     #total precision of each loading
    Lambda::Array{Float64,2}   #factor loadings
    h2::Array{Float64,1}       #factor heritability
    
    num::Int64
    no_f::Array{Float64,1}
    nofout::Array{Float64,1}
    
    scores::Array{Float64, 2}
    
    function LatentFactors(r, n, p, k, df, ad1, bd1, ad2, bd2, h2_divisions, sp, nrun)
        psijh = rand(Gamma(df/2,2/df), p, k);
        delta = [rand(Gamma(ad1+10,1/bd1), 1); rand(Gamma(ad2,1/bd2), k-1)];
        tauh = cumprod(delta);
        Plam = broadcast(*, psijh, tauh')
        Lambda = randn(p,k) .* sqrt.(1. ./ Plam)
        h2 = rand(k)
        num = 0
        no_f=zeros(nrun)
        nofout = k*ones(nrun+1)
        scores = zeros(k, n)
        new(r, n, p, k, df, ad1, bd1, ad2, bd2, h2_divisions, sp, nrun, psijh, delta, tauh, Plam, Lambda, h2, num, no_f, nofout, scores)
    end
end

#genetic_effects. This structure holds information about latent genetic
#effects. U is latent genetic effects on factor traits. d is genetic
#effects on residuals of the factor traits. Plus prior hyperparameters for
#genetic effect precisions

mutable struct GeneticEffects
    n::Int64
    as::Float64
    bs::Float64
    ps::Array{Float64, 1}
    U::Array{Float64, 2}
    d::Array{Float64, 2}
    function GeneticEffects(n, as, bs, r, p, k, h2)
        ps = rand(Gamma(as,1/bs),p)
        U  = broadcast(*, randn(k,r), sqrt.(h2))
        d  = broadcast(*, randn(p,r), 1. ./ sqrt.(ps))
        new(n, as, bs, ps, U, d)
    end
end

#interaction_effects. Similar to genetic_effects structure except for
#additional random effects that do not contribute to variation in the
#factor traits

mutable struct InteractionEffects
    as::Float64
    bs::Float64
    ps::Array{Float64, 1}
    mean::Array{Float64, 2}
    n::Int64
    W::Array{Float64, 2}
    W_out::Array{Float64, 2}
    function InteractionEffects(as, bs, p, r2)
        ps = rand(Gamma(as, 1/bs), p)
        mean = zeros(p,r2)
        n = r2
        W = broadcast(*, randn(p,r2), 1. ./ sqrt.(ps))
        W_out = zeros(p,r2)
        new(as, bs, ps, mean, n, W, W_out)
    end
end

#fixed_effects hold B
mutable struct FixedEffects
    b::Int64                # number of fixed effects, including intercept
    cov::Array{Float64, 2}  #inverse covariance of fixed effects
    mean::Array{Float64, 2} #mean of fixed effects
    B::Array{Float64, 2}    #current estimate of fixed effects
    function FixedEffects(b, p)
        cov = zeros(b,b)
        mean = zeros(p,b)
        B = randn(p,b)
        new(b, cov, mean, B)
    end
end

#Posterior holds Posterior samples & Posterior means

mutable struct PosteriorSample
    Lambda::Array{Array{Float64,2},1}
    no_f::Array{Float64, 1}
    ps::Array{Float64, 2} 
    resid_ps::Array{Float64, 2} 
    B::Array{Float64, 2} 
    U::Array{Array{Float64,2},1} 
    d::Array{Float64, 2} 
    W::Array{Float64, 2} 
    delta::Array{Array{Float64,1},1}
    G_h2::Array{Array{Float64,1},1} 
    function PosteriorSample(n, p, k, sp, r, r2, b)
        Lambda = Array{Array{Float64,2},1}(undef,sp) 
        no_f = zeros(sp)
        ps = zeros(p, sp)
        resid_ps = zeros(p ,sp)
        B = zeros(p, b)
        U = Array{Array{Float64,2},1}(undef,sp) 
        d = zeros(p, r)
        W = zeros(p, r2)
        delta = Array{Array{Float64,1},1}(undef,sp) 
        G_h2 = Array{Array{Float64,1},1}(undef,sp) 
        new(Lambda, no_f, ps, resid_ps, B, U, d, W, delta, G_h2)
    end
end

struct InputData
    n::Int64
    p::Int64
    b::Int64
    r1::Int64
    r2::Int64
    Y_full::Array{Float64, 2}
    Y::Array{Float64, 2}
    Mean_Y::Array{Float64, 1}
    Var_Y::Array{Float64, 1}
    X::Array{Float64, 2}
    A::Array{Float64, 2}
    Z_1::Array{Float64, 2}
    Z_2::Array{Float64, 2}
    function InputData(Y, X, A, Z_1) 
        n = size(Y, 1)
        p = size(Y, 2)
        b = size(X, 1)
        r1 = size(Z_1, 1)
        r2 = 0
        Y_full = copy(Y);
        Mean_Y = zeros(p);
        Var_Y = zeros(p);
        for i in 1:p
            Mean_Y[i] = mean(Y[:, i]);
            Var_Y[i]  =  var(Y[:, i]);
        end
        Y = broadcast(-, Y, Mean_Y');                     
        Y = broadcast(*, Y, 1. ./ sqrt.(Var_Y)');  
        Z_2 = zeros(0,n);
        new(n, p, b, r1, r2, Y_full, Y, Mean_Y, Var_Y, X, A, Z_1, Z_2)
    end
    function InputData(Y, X, A, Z_1, Z_2) 
        n = size(Y, 1)
        p = size(Y, 2)
        b = size(X, 1)
        r1 = size(Z_1, 1)
        r2 = size(Z_2, 1)
        Y_full = copy(Y);
        Mean_Y = zeros(p);
        Var_Y = zeros(p);
        for i in 1:p
            Mean_Y[i] = mean(Y[:, i]);
            Var_Y[i]  =  var(Y[:, i]);
        end
        Y = broadcast(-, Y, Mean_Y');                     
        Y = broadcast(*, Y, 1. ./ sqrt.(Var_Y)');  
        new(n, p, b, r1, r2, Y_full, Y, Mean_Y, Var_Y, X, A, Z_1, Z_2)
    end
end


struct Priors
    draw_iter::Int64
    burn::Int64
    sp::Int64
    thin::Int64
    
    b0::Float64
    b1::Float64
    
    epsilon::Float64
    h2_divisions::Int64
    
    k_init::Int64
    
    as::Float64
    bs::Float64
    
    df::Float64
    
    ad1::Float64
    bd1::Float64
    
    ad2::Float64
    bd2::Float64
    
    k_min::Float64
    prop::Float64
    
    nrun::Int64
    function Priors(draw_iter, burn, sp, thin, b0, b1, epsilon, h2_divisions, k_init, 
        as, bs, df, ad1, bd1, ad2, bd2, k_min, prop)
        nrun = burn+sp*thin
        new(draw_iter, burn, sp, thin, b0, b1, epsilon, h2_divisions, k_init, 
            as, bs, df, ad1, bd1, ad2, bd2, k_min, prop, nrun)
    end
end

function sample_lambda!(Factors::LatentFactors, Ytil::Array{Float64, 2}, 
    resid::Residuals, genetic_effects::GeneticEffects, eig_ZAZ::SVD)
    #Sample factor loadings (Factors.Lambda) while marginalizing over residual
    #genetic effects: Y - Z_2W = FL' + E, vec(E)~N(0,kron(Psi_E,In) + kron(Psi_U, ZAZ^T))
    # note: conditioning on W, but marginalizing over U.
    #  sampling is done separately by trait because each column of Lambda is
    #  independent in the conditional posterior
    # note: eig_ZAZ has parameters that diagonalize aI + bZAZ for fast
    #  inversion: inv(aI + bZAZ) = 1/b*Ur*diag(1./(eta+a/b))*Ur'
    p=resid.p
    k=Factors.k

    Ur = eig_ZAZ.U;
    eta = eig_ZAZ.S;
    FtU = Factors.scores*Ur;
    UtY = Ur' * Ytil';

    Zlams = rand(Normal(0,1),k,p);
    for j = 1:p
       FUDi  = genetic_effects.ps[j] * broadcast(*, FtU, 1. ./ (eta' .+ genetic_effects.ps[j]/resid.ps[j]));
       means = FUDi * UtY[:,j];
       Qlam  = FUDi*FtU' + diagm(Factors.Plam[j,:]); 
       Llam  = cholesky(Hermitian(Qlam)).L
       vlam  = Llam  \ means; 
       mlam  = Llam' \ vlam; 
       ylam  = Llam' \ Zlams[:,j];
       Factors.Lambda[j,:] = (ylam + mlam);
    end
end

function sample_means(Ytil::Array{Float64, 2}, Qt_Design::Array{Float64, 2}, N::Int64, 
  resid::Residuals, random_precision::Array{Float64, 1}, svd_Design_Ainv::Dict)
    # when used to sample [B;D]:
    # Y - FL' - Z_2W = XB + ZD + E, vec(E)~N(0,kron(Psi_E,In)). 
    # Note: conditioning on F, L and W.
    #  The vector [b_j;d_j] is sampled simultaneously. Each trait is sampled separately because their
    #  conditional posteriors factor into independent MVNs.
    # Note:svd_Design_Ainv has parameters to diagonalize mixed model equations for fast inversion: 
    #  inv(a*blkdiag(fixed_effects.cov,Ainv) + b*[X; Z_1][X; Z_1]') = Q*diag(1./(a.*s1+b.*s2))*Q'
    # Qt_Design = Q'*Design, which doesn't change each iteration. Design = [X;Z_1]
    #
    # function also used to sample W:
    #  Y - FL' - XB - ZD = Z_2W + E, vec(E) ~ N(0,kron(Psi_E,In)). 
    #  Here, conditioning is on B and D.

    p=resid.p;

    Q = svd_Design_Ainv["Q"];
    s1 = svd_Design_Ainv["s1"];
    s2 = svd_Design_Ainv["s2"];

    means = broadcast(*, Qt_Design * Ytil', resid.ps');
    location_sample = zeros(N,p);
    Zlams = randn(N,p);
    for j = 1:p,
        d = s1 * random_precision[j] + s2*resid.ps[j];
        mlam = broadcast(*, means[:,j], 1. ./ d);
        location_sample[:,j] = Q * (mlam + broadcast(*, Zlams[:,j], 1. ./ sqrt.(d)));
    end
    location_sample=location_sample';
    location_sample
end

function sample_h2s_discrete!(Factors::LatentFactors, eig_ZAZ::SVD)
    # sample factor heritibilties from a discrete set on [0,1)
    # prior places 50% of the weight at h2=0
    # samples conditional on F, marginalizes over U.

    Ur = eig_ZAZ.U;
    eta = eig_ZAZ.S;

    r = Factors.r;
    k = Factors.k;
    s = Factors.h2_divisions;

    log_ps = zeros(k,s);
    std_scores_b = Factors.scores*Ur;
    for i=1:s
        h2 = (i-1)/s;
        std_scores = Factors.scores;
        if h2 > 0
            std_scores = 1/sqrt(h2) * broadcast(*, std_scores_b, 1. ./ sqrt.(eta .+ (1-h2)/h2)');
            det = sum(log.((eta .+ (1-h2)/h2) * h2) / 2.);
        else       
            det = 0;
        end
        log_ps[:,i] = sum(logpdf(Normal(0, 1), std_scores), dims = 2) .- det; #Prior on h2
        if i==1
            log_ps = log_ps .+ log(s-1);
        end
    end
    for j=1:k
        norm_factor = max.(log_ps[j,:]) .+ log(sum(exp.(log_ps[j,:] - max.(log_ps[j,:]))));
        ps_j = exp.(log_ps[j,:] - norm_factor);
        log_ps[j,:] = ps_j;
        Factors.h2[j] = sum(rand() .> cumsum(ps_j)) / s;
    end
end

function sample_Us!(Factors::LatentFactors, genetic_effects::GeneticEffects, 
                    svd_ZZ_Ainv::Dict, Z_1::Array{Float64, 2})
    #samples genetic effects (U) conditional on the factor scores F:
    # F_i = U_i + E_i, E_i~N(0,s2*(h2*ZAZ + (1-h2)*I)) for each latent trait i
    # U_i = zeros(r,1) if h2_i = 0
    # it is assumed that s2 = 1 because this scaling factor is absorbed in
    # Lambda
    # svd_ZZ_Ainv has parameters to diagonalize a*Z_1*Z_1' + b*I for fast
    # inversion:

    Q = svd_ZZ_Ainv["Q"];
    s1 = svd_ZZ_Ainv["s1"];
    s2 = svd_ZZ_Ainv["s2"];

    k = Factors.k;
    n = genetic_effects.n;
    tau_e = 1. ./ (1 .- Factors.h2);
    tau_u = 1. ./ Factors.h2;
    b = Q' * Z_1 * broadcast(*, Factors.scores, tau_e)';
    z = randn(n,k);
    for j=1:k
        if tau_e[j]==1
            genetic_effects.U[j,:] = zeros(1,n);
        elseif isinf(tau_e[j])
            genetic_effects.U[j,:] = Factors.scores[j,:];
        else
            d = s2 * tau_u[j] + s1 * tau_e[j];
            mlam = broadcast(*, b[:,j], 1. ./ d);
            genetic_effects.U[j,:] = (Q*(mlam + broadcast(*, z[:,j], 1. ./ sqrt.(d))))';
        end
    end
end

function sample_factors_scores!(Ytil::Array{Float64, 2}, Factors::LatentFactors, 
    resid::Residuals, genetic_effects::GeneticEffects, Z_1::Array{Float64, 2})
#Sample factor scores given factor loadings, U, factor heritabilities and
#phenotype residuals

    k = Factors.k;
    n = Factors.n;
    Lambda = Factors.Lambda;
    Lmsg = broadcast(*, Lambda, resid.ps);
    tau_e = reshape(1. ./ (1. .- Factors.h2), k);
    S = cholesky(Hermitian(Lambda' * Lmsg + diagm(tau_e))).L;
    Meta = S' \ (S \ (Lmsg' * Ytil + broadcast(*, genetic_effects.U * Z_1 , tau_e)));
    Factors.scores = Meta + S' \ randn(k,n);   
end


function sample_delta!( Factors::LatentFactors, Lambda2_resid )
    #sample delta and tauh parameters that control the magnitudes of higher
    #index factor loadings.

    ad1 = Factors.ad1;
    ad2 = Factors.ad2;
    bd1 = Factors.bd1;
    bd2 = Factors.bd2;
    k = Factors.k;
    psijh = Factors.psijh;

    mat = broadcast(*, psijh, Lambda2_resid);
    n_genes = size(mat,1);
    ad = ad1 + 0.5*n_genes*k; 
    bd = bd1 + 0.5*(1/Factors.delta[1]) * sum(Factors.tauh .* sum(mat, dims = 1)');
    Factors.delta[1] = rand(Gamma(ad,1. / bd));
    Factors.tauh = cumprod(Factors.delta);

    for h = 2:k
        ad = ad2 + 0.5*n_genes*(k-h+1); 
        bd = bd2 + 0.5*(1. / Factors.delta[h])*sum(Factors.tauh[h:end].*sum(mat[:,h:end], dims = 1)');
        Factors.delta[h] = rand(Gamma(ad,1/bd));
        Factors.tauh = cumprod(Factors.delta);
    end
end

function  update_k!(Factors::LatentFactors, genetic_effects::GeneticEffects, 
    b0::Float64, b1::Float64, i::Int64, epsilon::Float64, prop::Float64, Z_1::Array{Float64, 2} )
#adapt the number of factors by dropping factors with only small loadings
#if they exist, or adding new factors sampled from the prior if all factors
#appear important. The rate of adaptation decreases through the chain,
#controlled by b0 and b1

    df = Factors.df;
    ad2 = Factors.ad2;
    bd2 = Factors.bd2;
    p = Factors.p;
    k = Factors.k;
    r = Factors.r;
    gene_rows = 1:p;
    Lambda = Factors.Lambda;

    # probability of adapting
    prob = 1/exp(b0 + b1*i);                
    uu = rand();

    # proportion of elements in each column less than eps in magnitude
    lind = mean(abs.(Lambda[1:p,:]) .< epsilon, dims = 1);    
    vec = lind .>= prop; num = sum(vec);       # number of redundant columns

    Factors.num = num;
    Factors.no_f[i] = k-num;

    if uu < prob && i>200
        if  i > 20 && num == 0 && all(lind .< 0.995) && k < 2*p 
            #add a column
            k=k+1;
            Factors.k = k;
            Factors.psijh = [Factors.psijh rand(Gamma(df/2, 2/df), p, 1)]
            Factors.delta = [Factors.delta; rand(Gamma(ad2, 1/bd2))];
            Factors.tauh = cumprod(Factors.delta);
            Factors.Plam = broadcast(*, Factors.psijh, Factors.tauh');
            Factors.Lambda = [Factors.Lambda randn(p,1) .* sqrt.(1 ./ Factors.Plam[:, k])];
            Factors.h2 = [Factors.h2; rand()];
            genetic_effects.U = [genetic_effects.U; randn(1,genetic_effects.n)];
            Factors.scores = [Factors.scores; genetic_effects.U[k,:]' * Z_1 + randn(1,r) .* sqrt(1-Factors.h2[k])];
        elseif num > 0      # drop redundant columns
            nonred = Array(1:k)[vec[:] .== 0]; # non-redundant loadings columns
            k = max(k - num, 1);
            Factors.k = k;
            Factors.Lambda = Lambda[:,nonred];
            Factors.psijh = Factors.psijh[:,nonred];
            Factors.scores = Factors.scores[nonred,:];
            for red = setdiff(1:k-1,nonred)
                #combine deltas so that the shrinkage of kept columns doesnt
                #decrease after dropping redundant columns
                Factors.delta[red+1] = Factors.delta[red+1] * Factors.delta[red];
            end
            Factors.delta = Factors.delta[nonred];
            Factors.tauh = cumprod(Factors.delta);
            Factors.Plam = broadcast(*, Factors.psijh, Factors.tauh');
            Factors.h2 = Factors.h2[nonred];
            genetic_effects.U = genetic_effects.U[nonred,:];
        end
    end
    Factors.nofout[i+1]=k;
end

function save_posterior_samples!(sp_num::Int64, Pr::Priors, D::InputData, Posterior::PosteriorSample, 
                                 resid::Residuals, fixed_effects::FixedEffects, 
                                 genetic_effects::GeneticEffects, Factors::LatentFactors, 
                                 interaction_effects::InteractionEffects)
    #save posteriors. Re-scale samples back to original variances.
    sp = Pr.sp;
    VY = D.Var_Y';       
    Lambda = broadcast(*, Factors.Lambda, sqrt.(VY'));     #re-scale by Y variances
    G_h2 = Factors.h2;
    U = genetic_effects.U;     
    delta = Factors.delta;
    genetic_ps = genetic_effects.ps ./ VY';
    resid_ps = resid.ps ./ VY';

    # save factor samples
    Lambda = Lambda[:,1:Factors.k];

    Posterior.Lambda[sp_num] = copy(Lambda);
    Posterior.delta[sp_num] = copy(delta);
    Posterior.G_h2[sp_num] = copy(G_h2);
    Posterior.U[sp_num] = copy(U);

    Posterior.no_f[sp_num] = Factors.k;

    Posterior.ps[:,sp_num] = copy(genetic_ps);
    Posterior.resid_ps[:,sp_num] = copy(resid_ps);

    # save B,U,W
    Posterior.B = Posterior.B + (broadcast(*, fixed_effects.B, sqrt.(VY'))  ./ sp);
    Posterior.d = Posterior.d + (broadcast(*, genetic_effects.d, sqrt.(VY')) ./ sp);
    Posterior.W = Posterior.W + (broadcast(*, interaction_effects.W, sqrt.(VY')) ./ sp); 

end

function sampleBSF_G!(Posterior::PosteriorSample, genetic_effects::GeneticEffects, 
                      Factors::LatentFactors, resid::Residuals, 
                      interaction_effects::InteractionEffects, fixed_effects::FixedEffects,
                      D::InputData, Pr::Priors)
    #precalculate some matrices
    #invert the random effect covariance matrices
    Ainv = inv(D.A)
    A_2_inv = Matrix{Int}(I, D.n, D.n); #Z_2 random effects are assumed to have covariance proportional to the identity. Can be modified.

    #pre-calculate transformation parameters to diagonalize aI + bZAZ for fast
    #inversion: inv(aI + bZAZ) = 1/b*u*diag(1./(s+a/b))*u'
    #uses singular value decomposition of ZAZ for stability when ZAZ is low
    #rank()
    # eig_ZAZ.vectors = u; -> eig_ZAZ.U
    # eig_ZAZ.values = diag(s); -> eig_ZAZ.S

    ZAZ = D.Z_1' * D.A * D.Z_1;
    eig_ZAZ = svd(ZAZ);

    Design=[D.X; D.Z_1];
    Design2 = Design * Design';
    svd_Design_Ainv = makeSVDdict(Array(BlockDiagonal([fixed_effects.cov, Ainv])), Design2);
    Qt_Design = svd_Design_Ainv["Q"]' * Design;

    #fixed effects + random effects 1
    #diagonalize mixed model equations for fast inversion: 
    #inv(a*blkdiag(fixed_effects.cov,Ainv) + b*[X Z_1]"[X Z_1]'') = Q*diag(1./(a.*s1+b.*s2))*Q"
    #inv(Array(BlockDiagonal([fixed_effects.cov, Ainv])) + Design2) ≈ svd_Design_Ainv["Q"] * diagm(1. ./ (svd_Design_Ainv["s1"]+svd_Design_Ainv["s2"])) * svd_Design_Ainv["Q"]'


    #random effects 2
    #as above; but for random effects 2. Here; fixed effects will be conditioned on; not sampled simultaneously. Otherwise identical.
    if(D.r2 > 1)
        Design_Z2 = Z_2
        Design2_Z2 = Design_Z2*Design_Z2'
        svd_Z2_2_A2inv = makeSVDdict(A_2_inv, Design2_Z2)
        Qt_Z2 = svd_Z2_2_A2inv["Q"]'*Design_Z2
    end

    #genetic effect variances of factor traits
    # diagonalizing a*Z_1*Z_1' + b*Ainv for fast inversion
    #diagonalize mixed model equations for fast inversion: 
    # inv(a*Z_1*Z_1" + b*Ainv) = Q*diag(1./(a.*s1+b.*s2))*Q'
    #similar to fixed effects + random effects 1 above; but no fixed effects.
    ZZt = D.Z_1 * D.Z_1'
    svd_ZZ_Ainv = makeSVDdict(ZZt, Array(Ainv))
    # inv(Array(ZZt) + Array(Ainv)) ≈ svd_ZZ_Ainv["Q"] * diagm(1. ./ (svd_ZZ_Ainv["s1"]+svd_ZZ_Ainv["s2"])) * svd_ZZ_Ainv["Q"]'

    sp_num=0
    @showprogress 1 "Running Gibbs sampler..." for i = 1:Pr.nrun

        ##fill in missing phenotypes
        ##conditioning on everything else()
        #phenMissing = isnan(Y_full)
        #if sum(sum(phenMissing))>0
        #    meanTraits = fixed_effects.B*X +  genetic_effects.d*Z_1 ...
        #        + interaction_effects.W*Z_2 + Factors.Lambda*Factors.scores
        #    meanTraits = meanTraits';        
        #    resids = bsxfun[@times,randn(size(Y_full)),1./sqrt(resid.ps')]
        #    Y[phenMissing] = meanTraits[phenMissing] + resids[phenMissing]
        #end

        #sample Lambda
        #conditioning on W; X; F; marginalizing over D
        Ytil = D.Y' - fixed_effects.B * D.X - interaction_effects.W * D.Z_2
        sample_lambda!(Factors, Ytil, resid, genetic_effects, eig_ZAZ)

        #sample fixed effects + random effects 1 [[B D]']
        #conditioning on W; F; L
        Ytil = D.Y' - interaction_effects.W * D.Z_2 - Factors.Lambda*Factors.scores
        N = genetic_effects.n + fixed_effects.b
        location_sample = sample_means(Ytil, Qt_Design, N, resid, genetic_effects.ps, svd_Design_Ainv)
        fixed_effects.B = location_sample[:,1:fixed_effects.b]
        genetic_effects.d = location_sample[:, fixed_effects.b+1 : fixed_effects.b+genetic_effects.n ]

        #sample random effects 2
        #conditioning on B; D; F; L
        N = interaction_effects.n
        if N>0
            Ytil = D.Y'-fixed_effects.B * D.X - genetic_effects.d * D.Z_1 - Factors.Lambda * Factors.scores
            location_sample = sample_means(Ytil, Qt_Z2, N, resid, interaction_effects.ps, svd_Z2_2_A2inv)
            interaction_effects.W = location_sample
        end

        #sample factor h2
        #conditioning on F; marginalizing over U
        sample_h2s_discrete!(Factors, eig_ZAZ)

        #sample genetic effects [U]
        #conditioning on F; Factor h2
        sample_Us!(Factors, genetic_effects, svd_ZZ_Ainv, D.Z_1)

        #sample F
        #conditioning on U; Lambda; B; D; W; factor h2s
        Ytil = D.Y' - fixed_effects.B * D.X - genetic_effects.d * D.Z_1 - interaction_effects.W * D.Z_2
        sample_factors_scores!(Ytil, Factors, resid, genetic_effects, D.Z_1)


        # -- Update ps -- #
        Lambda2 = Factors.Lambda .^ 2 
        as_p = Factors.df/2. + 0.5
        bs_p = 2. ./ (Factors.df .+ broadcast(*, Lambda2, Factors.tauh'))
        for i = 1:D.p, j = 1:Factors.k
            Factors.psijh[i, j] = rand(Gamma(as_p, bs_p[i, j]))
        end

        #continue from previous Y residual above
        Ytil = Ytil - Factors.Lambda * Factors.scores
        inv_bs = 1. ./ (resid.bs .+ 0.5 * sum(Ytil .^ 2, dims=2)) #model residual precision
        for i = 1:D.p
            resid.ps[i] = rand(Gamma(resid.as + 0.5 * D.n, inv_bs[i]))
        end

        #random effect 1 [D] residual precision
        inv_b_g = 1. ./ (genetic_effects.bs .+ 0.5 * sum(genetic_effects.d .^ 2, dims=2))
        for i = 1:D.p
         genetic_effects.ps[i] = rand(Gamma(genetic_effects.as + 0.5 * genetic_effects.n, inv_b_g[i]))
        end

        #n = interaction_effects.n
        #interaction_effects.ps=gamrnd[interaction_effects.as + 0.5*n,1./(interaction_effects.bs+0.5*sum(interaction_effects.W.^2,dims=2))]; #random effect 2 [W] residual precision

        #------Update delta & tauh------#
        sample_delta!(Factors, Lambda2)

        #---update precision parameters----#
        Factors.Plam = broadcast(*, Factors.psijh, Factors.tauh')

        # ----- adapt number of factors to samples ----#
        update_k!( Factors, genetic_effects, Pr.b0, Pr.b1 ,i , Pr.epsilon, Pr.prop, D.Z_1 )

        # -- save sampled values [after thinning] -- #
        if i%Pr.thin==0 && i > Pr.burn
            sp_num = Int64((i-Pr.burn)/Pr.thin)
            save_posterior_samples!(sp_num, Pr, D, Posterior, resid, fixed_effects,
                genetic_effects, Factors, interaction_effects)
        end
    end
end

function PosteriorMeans(Posterior::PosteriorSample, D::InputData, Pr::Priors)
    kmax = convert(Int64, maximum(Posterior.no_f));

    Lambda_est = zeros(D.p, kmax);
    G_s = zeros(D.p, D.p, Pr.sp);
    P_s = zeros(D.p, D.p, Pr.sp);
    E_s = zeros(D.p, D.p, Pr.sp);
    G_est = zeros(D.p, D.p);
    P_est = zeros(D.p, D.p);
    E_est = zeros(D.p, D.p);

    factor_h2s_est = zeros(kmax);
    for j=1:Pr.sp
        Lj  = Posterior.Lambda[j];
        h2j = Posterior.G_h2[j];

        Pj = Lj * Lj'                   + diagm(1. ./ (Posterior.ps[:,j])) + diagm(1. ./(Posterior.resid_ps[:,j]));
        Gj = Lj * diagm(     h2j) * Lj' + diagm(1. ./ (Posterior.ps[:,j]));
        Ej = Lj * diagm(1 .- h2j) * Lj'                                    + diagm(1. ./(Posterior.resid_ps[:,j]));


        P_s[:, :, j] = copy(Pj);
        G_s[:, :, j] = copy(Gj);
        E_s[:, :, j] = copy(Ej);

        P_est = P_est + Pj./Pr.sp;
        G_est = G_est + Gj./Pr.sp;
        E_est = E_est + Ej./Pr.sp;

        k_local = size(h2j, 1)
        Lambda_est[:, 1:k_local]  = Lambda_est[:, 1:k_local] + Lj./Pr.sp;
        factor_h2s_est[1:k_local] = factor_h2s_est[1:k_local] + h2j./Pr.sp;

    end

    posterior_mean = Dict("G" => G_est,
      "P" => P_est,
      "E" => E_est,
      "Gs" => G_s,
      "Ps" => P_s,
      "Es" => E_s,
      "Lambda" => Lambda_est,
      "F_h2s" => factor_h2s_est)
    posterior_mean
end