IAT <- function (x) 
{
  print("test")
  if (missing(x)) 
    stop("The x argument is required.")
  if (!is.vector(x)) 
    x <- as.vector(x)
  print(dim(x))
  dt <- x
  n <- length(x)
  mu <- mean(dt)
  s2 <- var(dt)
  maxlag <- max(3, floor(n/2))
  #print(n)
  print("mu")
  print(mu)
  print("s2")
  print(s2)
  #print(maxlag)
  Ga <- rep(0, 2)
  Ga[1] <- s2
  lg <- 1
  Ga[1] <- Ga[1] + sum((dt[1:(n - lg)] - mu) * (dt[(lg + 1):n] - 
                                                  mu))/n
  print("Ga")
  print(Ga)
  print(dt[(lg + 1):n] - mu)
  
  m <- 1
  lg <- 2 * m
  print(length((dt[1:(n - lg)] - mu)))
  
  Ga[2] <- sum((dt[1:(n - lg)] - mu) * (dt[(lg + 1):n] - mu))/n
  print(Ga[2])
  lg <- 2 * m + 1
  Ga[2] <- Ga[2] + sum((dt[1:(n - lg)] - mu) * (dt[(lg + 1):n] - 
                                                  mu))/n
  IAT <- Ga[1]/s2
  print("test")
  print(IAT)
  while ((Ga[2] > 0) & (Ga[2] < Ga[1])) {
    m <- m + 1
    if (2 * m + 1 > maxlag) {
      cat("Not enough data, maxlag=", maxlag, "\n")
      break
    }
    Ga[1] <- Ga[2]
    lg <- 2 * m
    Ga[2] <- sum((dt[1:(n - lg)] - mu) * (dt[(lg + 1):n] - 
                                            mu))/n
    lg <- 2 * m + 1
    Ga[2] <- Ga[2] + sum((dt[1:(n - lg)] - mu) * (dt[(lg + 
                                                        1):n] - mu))/n
    IAT <- IAT + Ga[1]/s2
    print(IAT)
  }
  IAT <- -1 + 2 * IAT
  return(IAT)
}

setwd('/Users/davidhuijser/Documents/emcee/Autocorrelation')
means = read.table('/Users/davidhuijser/Documents/emcee/Autocorrelation/Converted_Gaussian_data_means_n=100_j=3NEWEST.txt', header=F)
#vars = read.table('/Users/davidhuijser/Documents/emcee/Autocorrelation/Converted_Gaussian_data_means_n=100_j=3NEWEST.txt', header=F)
vorm = dim(means)
Z_means = array( as.matrix(means),dim=c(vorm[1]/4,4, vorm[2]))


print("Print means X")
print(Z_means[1:9,1,1])
print("Print means Y")
print(Z_means[1,1:4,1])
print("Print means Z")
print(Z_means[1,1,1:9])


# test 

IAT(Z_means[,1,1])
#IAT(Z_means[,2,])
#IAT(Z_means[,3,])
#IAT(Z_means[,4,])