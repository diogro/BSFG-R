
library(ratones)
library(stanAnimal)

if(!require(nadiv)){install.packages("nadiv"); library(nadiv)}
if(!require(pedtools)){install.packages("pedtools"); library(pedtools)}
if(!require(JuliaCall)){install.packages("JuliaCall"); library(JuliaCall)}
if(!require(tidyverse)){install.packages("tidyverse"); library(tidyverse)}
if(!require(MCMCglmm)){install.packages("MCMCglmm"); library(MCMCglmm)}
if(!require(mvtnorm)){install.packages("mvtnorm"); library(mvtnorm)}
library(rhdf5)

library(AtchleyMice)
mice_ped = as.data.frame(mice_pedigree)

ped = prepPed(mice_ped)
A = as.matrix(nadiv::makeA(ped))
colnames(A)
F6_ID = mice_info$F6$ID
A = A[F6_ID, F6_ID]

ped = randomPed(50, 5, seed = 1)
ped = prepPed(as.data.frame(ped[1:3]))
A = as.matrix(nadiv::makeA(ped))
A_chol = chol(A)

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

p = ncol(G)
r = nrow(A)
a = t(A_chol) %*% matrix(rnorm(p*r),r,p) %*% chol(G)

cov(a)
cor(a)

beta = matrix(c(1, 2, 3, 
                0.1, 0.2, 0.5,
                0.05, 0.1, 0.3), 3, 3, byrow = TRUE)
colnames(beta) = c("x", "y", "z")
rownames(beta) = c("Intercept", "sex", "Z")

sex = as.numeric(factor(mice_info$F6$Sex)[1:nrow(a)])-1

Z = rnorm(nrow(a))
Intercept = rep(1, nrow(a))
X = cbind(Intercept, sex, Z)
rownames(X) = rownames(a)
e = rmvnorm(nrow(a), sigma = E)

Y = X %*% beta + a + e

colnames(Y) = c("x", "y", "z")

out_folder = "BSFG_run"
out_file = file.path(out_folder, "/setup.h5")
if(file.exists(out_file))
  file.remove(out_file)
h5createFile(out_file)
h5createGroup(out_file, "Input")
h5write(Y, out_file, name="Input/Y")
h5write(t(X), out_file, name="Input/X")
h5write(as.matrix(A), out_file, name="Input/A")
h5write(diag(nrow(A)), out_file, name="Input/Z_1")
H5close()

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
