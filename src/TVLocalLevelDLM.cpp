#include <config.h>
#include <RcppDist.h>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

using namespace arma;


void forward(const mat& y, const vec& m0, const mat& C0,
             const cube& FF, const mat& G,
             const cube& V, const cube& W,
             mat& a, mat& m, mat& f, mat& error,
             cube& C, cube& Q, cube& R)
{
  int T = y.n_cols;
  a.col(0) = G * m0.col(0);
  R.slice(0) = G * C0 * G.t() + W.slice(0);
  f.col(0) = FF.slice(0) * a.col(0);
  Q.slice(0) = FF.slice(0) * R.slice(0) * FF.slice(0).t() + V.slice(0);
  error.col(0) = y.col(0) - f.col(0);
  m.col(0) = a.col(0) + R.slice(0) * FF.slice(0).t() * inv(Q.slice(0)) * error.col(0);
  C.slice(0) = R.slice(0) - R.slice(0) * FF.slice(0).t() * inv(Q.slice(0)) * FF.slice(0) * R.slice(0);
  for(int t=1; t<T; t++){
    // # predict
    a.col(t) = G * m.col(t-1);
    R.slice(t) = G * C.slice(t-1) * G.t() + W.slice(t);
    // # // marginalize
    f.col(t) = FF.slice(t) * a.col(t);
    Q.slice(t) = FF.slice(t) * R.slice(t) * FF.slice(t).t() + V.slice(t);
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
              const vec& v0, const mat& S0v, const mat& S0w, cube& V, cube& W)
{
  int T = y.n_cols, p = y.n_rows, p2 = G.n_cols, k, t;
  mat theta_ = theta;
  theta_.shed_col(0);
  mat SSy(p,p,fill::zeros), SSs(p2,p2,fill::zeros);
  for(t=0; t<T; t++){
    k = t+1;
    SSy = (y.col(t) - FF.slice(t)*theta_.col(t)) * (y.col(t) - FF.slice(t)*theta_.col(t)).t()/2;
    SSs = (G*theta.col(k) - G*theta.col(k-1)) * (G*theta.col(k) - G*theta.col(k-1)).t()/2;
    V.slice(t) = inv(rwish(v0(0)+t/2.0, inv(S0v + SSy)));
    W.slice(t) = inv(rwish(v0(1)+t/2.0, inv(S0w + SSs)));
  }
}


// [[Rcpp::export]]
Rcpp::List dlm4(const mat& y, const vec& v0, const int& niter,
                mat& C0, vec& m0, const mat& S0v, const mat& S0w,
                const cube& FF, const mat& G)
{
  int T = y.n_cols;
  int p = y.n_rows;
  int p2 = G.n_cols;
  
  // initialize and allocate variables necessary for forward filtering and
  // backward sampling
  // for code reability, order according to Kalman filter, then backward sampele, etc
  
  cube mc_theta0(p2,1,niter,fill::ones);
  cube V(p,p,T,fill::eye), W(p2,p2,T,fill::eye);
  for(int t=0;t<T;t++){
    V.slice(t) = ones(p,p);
    W.slice(t) = ones(p2,p2);
  }
  field<cube> mc_V(niter);
  field<cube> mc_W(niter);
  for(int i=0; i<niter;i++){
    mc_V(i).ones(1,1,T);
    mc_W(i).ones(p2,p2,T);
  }
  
  
  mat a(p2, T, fill::zeros), f(p, T, fill::zeros), m(p2, T, fill::zeros), error(p, T, fill::zeros),
  H(p2,p2,fill::eye), theta(p2,T+1,fill::zeros), h(p2, 1, fill::ones),
  SSy(p,p,fill::zeros), SSs(p2,p2,fill::zeros);
  
  cube R(p2, p2, T, fill::ones), Q(p2, p2, T, fill::ones), C(p, p, T, fill::zeros);
  for(int i=1; i<niter; i++) {
    // forward filter
    emission(y, theta, FF, G, v0, S0v, S0w, V, W);
    forward(y, m0, C0, FF, G, V, W, a, m, f, error, C, Q, R);
    // // backward sample
    //backward(theta, h, H, a, join_rows(m0,m), join_slices(C0,C), R, G);
    // // sample variance parameters

    mc_V(i-1) = V;
    mc_W(i-1) = W;
    mc_theta0.slice(i-1) = theta.col(0);
  }
  Rcpp::List out(4);
  out["m"] = m;
  out["mc_V"] = mc_V;
  out["mc_W"] = mc_W;
  out["mc_theta0"] = mc_theta0;
  out["theta"] = theta;
  return(out);
}

/*** R
simdata = function(N, m0, C0, V, W){
  mu = rep(0,N)
  y = rep(0,N)
  mu[1] = m0
  y[1] = mu[1] + rnorm(1,0,C0)
  for(t in 2:(N+1)){
    e = rnorm(1, 0, sqrt(V[t]))
    g = rnorm(1, 0, sqrt(W[t]))
    mu[t] = mu[t-1] + g
    y[t] = mu[t] + e
  }
  return(list("y"=y,"mu"=mu))
}
N=1000
T = N
sim=simdata(N, 50, 1, 1, 1)
y1=sim$y
mu1 = sim$mu

sim=simdata(N, 50, 1, .2, .05)
y2=sim$y
mu2 = sim$mu

y=cbind(y1,y2)
mu=cbind(mu1,mu2)

p = ncol(y)
matplot(y,type="l")
T = nrow(y)
FF = array(diag(p), dim=c(p,p,T))
start.time <- Sys.time()
gibbs = dlm4(t(y),v0 = rep(1,2),
                       FF = FF, G=diag(p),
                       niter = 100,
                       C0 = 10*diag(p), m0 = 0,
                       S0v = diag(p), S0w = diag(p))
end.time <- Sys.time()
(end.time-start.time)
cbind(mu1,mu2)[1,]
# matplot(cbind(mu1,mu2),type="l")
# matplot(t(gibbs$m),add=T,type='o')
dim(gibbs$mc_V)
apply(gibbs$mc_V[[1]],1:2,mean)
apply(gibbs$mc_W[[1]],1:2,mean)


apply(gibbs$mc_V,1:2,quantile,c(.025,.5,.975))
apply(gibbs$mc_W,1:2,quantile,c(.025,.5,.975))

hist(gibbs$mc_theta0[1,1,],breaks=100)
hist(gibbs$mc_theta0[2,1,], breaks=100)

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


