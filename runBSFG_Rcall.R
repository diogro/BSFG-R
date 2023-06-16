
library(ratones)
library(stanAnimal)

list_pkgs <- c("pedigreemm", "mvtnorm", "pedtools", "JuliaCall", "nadiv", "MCMCglmm")
new_pkgs <- list_pkgs[!(list_pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs) > 0){ install.packages(new_pkgs) }
library(nadiv)
library(pedtools)
library(mvtnorm)
library(JuliaCall)
library(MCMCglmm)


r_ped <- randomPed(250, 50)
plot(r_ped)
ped = data.frame(id = r_ped[["_comp1"]]$id, 
                 sire = r_ped[["_comp1"]]$fidx, 
                 dam = r_ped[["_comp1"]]$midx)
str(r_ped)
ped = data.frame(id = r_ped$ID, 
                 sire = r_ped$FIDX, 
                 dam = r_ped$MIDX)
ped[ped==0] = NA
ped = prepPed(ped)
A = as.matrix(nadiv::makeA(ped))

corrG <- matrix(c(1, 0.7, 0.0,
                  0.7, 1, 0.5,
                  0.0, 0.5,   1), 3, 3, byrow = T)
corrE <- matrix(c(  1, 0.0, 0.0,
                    0.0,   1, 0.0,
                    0.0, 0.0,  1), 3, 3, byrow = T)
varG = 1:3
varE = 1.5*varG
G = sqrt(varG) %*% t(sqrt(varG)) * corrG
E = sqrt(varE) %*% t(sqrt(varE)) * corrE
varG / (varG + varE)
a = rbv(ped, G)
cov(a)
cor(a)

beta = matrix(c(1, 2, 3, 
                0.1, 0.2, 0.5,
                0.05, 0.1, 0.3), 3, 3, byrow = TRUE)
colnames(beta) = c("x", "y", "z")
rownames(beta) = c("Intercept", "sex", "Z")

sex = sample(c(0, 1), nrow(a), replace = T)
Z = rnorm(nrow(a))
Intercept = rep(1, nrow(a))
X = cbind(Intercept, sex, Z)
rownames(X) = rownames(a)
e = rmvnorm(nrow(a), sigma = E)

Y = X %*% beta + a + e

colnames(Y) = c("x", "y", "z")


julia_setup()
julia_library("BayesianSparseFactorGmatrix")
bsf_out = julia_call("runBSFGModel", Y, t(X), A, diag(nrow(Y)),
                     burn=as.integer(1000)[1], sp=as.integer(1000)[1], thin=as.integer(10)[1])
G_julia = bsf_out[[1]]$G
E_julia = bsf_out[[1]]$E
diag(G_julia) / (diag(G_julia + E_julia))
cov(a)
cor(a)
cov2cor(G_julia)
cov2cor(E_julia)

par(mfrow = c(1, 2))
library(corrplot)
corrplot(cor(a), method = "ellipse")
corrplot(cov2cor(G_julia), method = "ellipse")
