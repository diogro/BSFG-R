
library(AtchleyMice)
library(BSFGR)

if(!require(nadiv)){install.packages("nadiv"); library(nadiv)}
if(!require(pedtools)){install.packages("pedtools"); library(pedtools)}
if(!require(JuliaCall)){install.packages("JuliaCall"); library(JuliaCall)}
if(!require(tidyverse)){install.packages("tidyverse"); library(tidyverse)}

data("mice_pedigree")
fullA = makeA(mice_pedigree)

skull_data = mice_crania$ed %>%
  filter(Gen == "F6")
skull_formula = paste0("cbind(",
                       paste(names(dplyr::select(skull_data, IS_PM:BA_OPI)),
                             collapse = ", "), ") ~ Sex + Litter_size_birth + Birth_litter_size_weaning + Foster_litter_size_weaning")
modelInput = genAnimalModelInput(skull_formula, skull_data, fullA, out_type = "julia", out_folder = "BSFG_julia_mice")
attach(modelInput)
julia_setup()
julia_library("BayesianSparseFactorGmatrix")
bsf_out = julia_call("runBSFGModel", Y, t(X), as.matrix(A), diag(nrow(Y)),
                     burn=as.integer(100)[1], sp=as.integer(100)[1], thin=as.integer(1)[1])
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
