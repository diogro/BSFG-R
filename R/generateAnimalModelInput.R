#' Generate input for BSFG model in Matlab and Julia
#'
#' @param formula formula specifing fixed effects and response variables
#' @param data data.frame with data, default is ratonesdf
#' @param A relatinship matrix from pedigree
#' @param out_folder folder to create the Matlab input files
#' @importFrom R.matlab writeMat
#' @importFrom stats lm as.formula model.matrix
#' @return
#' list with animal model input objects
#' @export
#' @examples
#' library(ratones)
#' data(ratonesdf)
#' data(ratones_ped)
#' formula = paste0("cbind(",
#'           paste(names(dplyr::select(ratonesdf, IS_PM:BA_OPI)),
#'                 collapse = ", "), ") ~ SEX + AGE")
#' animal_model_data = generateAnimalModelInput(formula, ratonesdf,
#'                                              ratones_ped$A, out_foldel = "./")
genAnimalModelInput <- function(formula, data, A,
                                     IDcol = "ID", out_folder = NULL){

  model = lm(as.formula(formula), data = data)
  Y = model$model[[1]]
  X = model.matrix(model)
  pos = sapply(data[[IDcol]], function(x) which(x == rownames(A)))
  A = as.matrix(A[pos, pos])
  Z = diag(nrow(A))
  if(!is.null(out_folder)){
    if(!dir.exists(out_folder))
      dir.create(file.path(out_folder), showWarnings = FALSE)
    writeMat(paste0(out_folder, "/setup.mat"), A = A, X = t(X), Y = Y, Z_1 = Z)
  }
  return(list(K = ncol(Y),
              J = ncol(X),
              N = nrow(Y),
              A = A,
              X = X,
              Y = Y,
              Z = Z))
}
