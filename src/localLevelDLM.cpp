//#define ARMA_NO_DEBUG
#include <RcppDist.h>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

using namespace arma;


void forward(const mat& y, const vec& m0, const mat& C0,
              const cube& FF, const mat& G,
              const mat& V, const mat& W,
              mat& a, mat& m, mat& f, mat& error, 
              cube& C, cube& Q, cube& R)
{
  int T = y.n_cols;
  a.col(0) = G * m0.col(0);
  R.slice(0) = G * C0 * G.t() + W;
  f.col(0) = FF.slice(0) * a.col(0);
  Q.slice(0) = FF.slice(0) * R.slice(0) * FF.slice(0).t() + V;
  error.col(0) = y.col(0) - f.col(0);
  m.col(0) = a.col(0) + R.slice(0) * FF.slice(0).t() * inv(Q.slice(0)) * error.col(0);
  C.slice(0) = R.slice(0) - R.slice(0) * FF.slice(0).t() * inv(Q.slice(0)) * FF.slice(0) * R.slice(0);
  for(int t=1; t<T; t++){
    // # predict
    a.col(t) = G * m.col(t-1);
    R.slice(t) = G * C.slice(t-1) * G.t() + W;
    // # // marginalize
    f.col(t) = FF.slice(t) * a.col(t);
    Q.slice(t) = FF.slice(t) * R.slice(t) * FF.slice(t).t() + V;
    // # // filter
    error.col(t) = y.col(t) - f.col(t);
    m.col(t) = a.col(t) + R.slice(t) * FF.slice(t).t() * inv(Q.slice(t)) * error.col(t);
    C.slice(t) = R.slice(t) - R.slice(t) * FF.slice(t).t() * inv(Q.slice(t)) * FF.slice(t) * R.slice(t);
  }
}

void backward(mat& theta, mat& h, mat& H, const mat& a, const mat& m, const cube& C, 
               const cube& R, const mat& G)
{
  int T = a.n_cols;
  int p = a.n_rows;
  int p2 = G.n_cols;
  theta.col(T) = rmvnorm(1, m.col(T), C.slice(T)).t();
  // expressions from Bayesian Filtering and Smoothing book
  for(int t=T-1; t>=0; t--){
    h = m.col(t) + C.slice(t)*G.t()*inv(R.slice(t))*(theta.col(t+1)-a.col(t));
    H = C.slice(t) - C.slice(t)*G.t()*inv(R.slice(t))*G*C.slice(t);
    theta.col(t) = rmvnorm(1, h, H).t();
  }
}

void emission(const mat& y, const mat& theta,
              const cube& FF, const mat&G, 
              const vec& v0, const mat& S0v, const mat& S0w, mat& V, mat& W)
{
  int T = y.n_cols, p = y.n_rows, p2 = G.n_cols, k, t;
  mat theta_ = theta;
  theta_.shed_col(0);
  mat SSy(p,p,fill::zeros), SSs(p2,p2,fill::zeros);
  for(t=0; t<T; t++){
    k = t+1;
    SSy += (y.col(t) - FF.slice(t)*theta_.col(t)) * (y.col(t) - FF.slice(t)*theta_.col(t)).t()/2;
    SSs += (G*theta.col(k) - G*theta.col(k-1)) * (G*theta.col(k) - G*theta.col(k-1)).t()/2;
  }
  V = inv(rwish(v0(0)+T/2.0, inv(S0v + SSy)));
  W = inv(rwish(v0(1)+T/2.0, inv(S0w + SSs)));
}


// [[Rcpp::export]]
Rcpp::List dlm3(const mat& y, const vec& v0, const int& niter,
                mat& C0, vec& m0, const mat& S0v, const mat& S0w,
                const cube& FF, const mat& G)
{
  int T = y.n_cols;
  int p = y.n_rows;
  int p2 = G.n_cols;
  
  // initialize and allocate variables necessary for forward filtering and 
  // backward sampling
  // for code reability, order according to Kalman filter, then backward sampele, etc
  
  cube mc_V(p,p,niter,fill::ones), mc_W(p2,p2,niter,fill::ones), 
  mc_theta0(p2,1,niter,fill::ones), mc_theta(p2,T+1,niter,fill::zeros);
  mat V(p,p,fill::eye), W(p2,p2,fill::eye);
  
  mat a(p2, T, fill::zeros), f(p, T, fill::zeros), m(p2, T, fill::zeros), error(p, T, fill::zeros),
  H(p2,p2,fill::eye), theta(p2,T+1,fill::zeros), h(p2, 1, fill::ones),
  SSy(p,p,fill::zeros), SSs(p2,p2,fill::zeros);
  
  cube R(p2, p2, T, fill::ones), Q(p, p, T, fill::ones), C(p2, p2, T, fill::zeros);
  for(int i=0; i<niter; i++) {
    // forward filter
    forward(y, m0, C0, FF, G, V, W, a, m, f, error, C, Q, R);
    // // backward sample
    backward(theta, h, H, a, join_rows(m0,m), join_slices(C0,C), R, G);
    // // sample variance parameters
    emission(y, theta, FF, G, v0, S0v, S0w, V, W);
    
    mc_V.slice(i) = V;
    mc_W.slice(i) = W;
    mc_theta.slice(i) = theta;
    mc_theta0.slice(i) = theta.col(0);
  }
  Rcpp::List out(4);
  out["mc_V"] = mc_V;
  out["mc_W"] = mc_W;
  out["mc_theta0"] = mc_theta0;
  out["mc_theta"] = mc_theta;
  return(out);
}

/*** R
simdata = function(N, m0, C0, V, W, X){
  mu = matrix(0,nrow=length(m0),ncol=N)
  y = rep(0,N)
  mu[,1] = m0
  y[1] = sum(mu[,1]) + mvtnorm::rmvnorm(1,rep(0,1),matrix(V))
  for(t in 2:N){
    e = mvtnorm::rmvnorm(1,rep(0,1),matrix(V))
    g = t(mvtnorm::rmvnorm(1,matrix(0,nrow=length(m0)),W))
    mu[,t] = mu[,t-1] + g
    y[t] = X[t,]%*%mu[,t] + e
  }
  return(list("y"=y,"mu"=mu))
}
N=1000
T = N
m0 = matrix(c(1,0),nrow=2,ncol=1)
X = matrix(0, nrow=T, ncol=2)
X[,1] = 1
X[,2] = 0
# X[,3] = 0
V=matrix(.4);W=diag(.8,2)
sim=simdata(N, m0, C0=diag(1,2), V=V, W=W, X)
y1=sim$y
mu1 = sim$mu
matplot(X*t(mu1),type="l")
y=y1#matrix(cbind(y1,y2),ncol=2)
plot(y1)
matplot(X*t(mu1),type="l",lwd=6,col="blue",add=0)
p = nrow(m0)#ncol(y)
T = length(y)#row(y)
FF = array(matrix(c(1),nrow = 2),dim =c(1,1,T))
for(i in 1:T){
  FF[,,i] = 1#matrix(nrow=1,c(X[i,1],X[i,2]))
}
p=1
start.time <- Sys.time()
gibbs = dlm3(t(y),v0 = rep(1,2),
             FF = FF, G=diag(p),
             niter = 5000,
             C0 = 1*diag(p), m0 = matrix(0,nrow=p),
             S0v = diag(1), S0w = diag(p))

end.time <- Sys.time()
(end.time-start.time)

matplot(X*t(mu1),type="l")
matplot(t(apply(gibbs$mc_theta,1:2,quantile,.025)),type="l",add=T,col="darkblue")
matplot(t(apply(gibbs$mc_theta,1:2,quantile,.975)),type="l",add=T,col="darkblue")

(apply(gibbs$mc_V,1:2,quantile))
apply(gibbs$mc_W,1:2,quantile)
#
#
# apply(gibbs$mc_V,1:2,quantile,c(.025,.5,.975))
# apply(gibbs$mc_W,1:2,quantile,c(.025,.5,.975))
#
# hist(gibbs$mc_theta0[1,1,],breaks=100)
# hist(gibbs$mc_theta0[2,1,], breaks=100)

# hist(gibbs$mc_V[2,2,],breaks=100)
# hist(gibbs$mc_W[2,2,],breaks=100)
#
# plot(gibbs$mc_V[2,2,],type="l")
# plot(gibbs$mc_W[1,1,],type="l")
#
# sim=simdata(N, 0, 1, .2, .5)
# y1=sim$y
#
# sim=simdata(N, 0, 1, .2, .5)
# y2=sim$y
#
# y=cbind(y1,y2)

*/


