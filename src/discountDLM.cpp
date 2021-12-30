// #define ARMA_NO_DEBUG
#include <RcppDist.h>
#include <iostream>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

using namespace arma;

void forward(const mat& y, const vec& m0, const mat& C0,
             const cube& FF, const mat& G,
             const mat& V,
             mat& a, mat& m, mat& f,
             cube& C, cube& Q, cube& R, vec& alpha, vec& beta, vec& s,
             const vec& alpha0, const vec& beta0, 
             const double delta_w, const double delta_v){
  uword T = y.n_cols;
  a.col(0) = G * m0.col(0);
  R.slice(0) = G * C0 * G.t() + (1-delta_w)/delta_w*G.t()*C0*G;
  f.col(0) = FF.slice(0) * a.col(0);
  Q.slice(0) = FF.slice(0) * R.slice(0) * FF.slice(0).t() + alpha0(0)/beta0(0)*V;
  mat Qinv = inv(Q.slice(0));
  m.col(0) = a.col(0) + R.slice(0) * FF.slice(0).t() * Qinv * (y.col(0) - f.col(0));
  C.slice(0) = R.slice(0) - R.slice(0) * FF.slice(0).t() * Qinv * FF.slice(0) * R.slice(0);
  // modify alpha and beta for discount factor
  alpha(0) = as_scalar(delta_v*alpha0) + y.n_rows;
  beta(0) = as_scalar(delta_v*beta0) + dot(y.col(0) - f.col(0), Qinv * (y.col(0) - f.col(0)));
  for(uword t=1; t<T; t++){
    s(t-1) = beta(t-1) / alpha(t-1);
    // # predict
    a.col(t) = G * m.col(t-1);
    R.slice(t) = G * C.slice(t-1) * G.t() / delta_w;
    // # // marginalize
    f.col(t) = FF.slice(t) * a.col(t);
    Q.slice(t) = FF.slice(t) * R.slice(t) * FF.slice(t).t() + s(t-1)*V;
    Qinv = inv(Q.slice(t));
    // discount factors
    alpha(t) = delta_v*alpha(t-1) + y.n_rows;
    beta(t) = delta_v*beta(t-1) + s(t-1) * dot(y.col(t) - f.col(t), Qinv * (y.col(t) - f.col(t)));
    s(t) = beta(t) / alpha(t);
    // # // filter
    m.col(t) = a.col(t) + R.slice(t) * FF.slice(t).t() * Qinv * (y.col(t) - f.col(t));
    C.slice(t) = (R.slice(t) - R.slice(t) * FF.slice(t).t() * Qinv * FF.slice(t) * R.slice(t)) * s(t)/s(t-1);
  }
}

void backward(mat& theta, const mat& a, const mat& m, const cube& C, 
              const cube& R, const mat& G, const vec& n, const vec& d,
              const double delta_v, const double delta_w, vec& v, vec& s){
  uword T = a.n_cols;
  theta.col(T) = rmvnorm(1, m.col(T), C.slice(T)).t();
  // i believe this is fixed, referencing the MATLAB code
  double vinv;
  vinv = Rcpp::rgamma(1, n(T)/2.0, 2.0/d(T))(0);
  v(T) = 1.0/vinv;
  for(uword t=T-1; t>=0; t--){
    vinv = delta_v*vinv+Rcpp::rgamma(1, ((1-delta_v)*n(t))/2.0, 1.0/(n(t)*s(t)))(0);
    v(t) = 1/vinv;
    theta.col(t) = rmvnorm(1,
              (1-delta_w)*m.col(t) + delta_w*theta.col(t+1),
              (1-delta_w)*C.slice(t)).t();
  }
}


// [[Rcpp::export]]
Rcpp::List dlm_df(const mat& y, const int& niter, const int& burnin, mat& C0, mat& m0,
                          const cube& FF, const mat& G, const vec alpha0, const vec beta0, 
                          const double delta_v, const double delta_w){
  int T = y.n_cols;
  int p = y.n_rows;
  int p2 = G.n_cols;
  
  cube theta_draw(p2, T+1, niter-burnin,fill::zeros);

  mat v_draw(T+1, niter-burnin, fill::zeros);
  
  
  mat a(p2, T, fill::zeros), m(p2, T, fill::zeros), f(p, T, fill::zeros), error(p, T, fill::zeros),
  H(p2,p2,fill::eye), theta(p2,T+1,fill::zeros), h(p2, 1, fill::ones);
  
  cube R(p2, p2, T, fill::ones), Q(p, p, T, fill::ones), A(p2, p, T, fill::zeros), C(p2, p2, T, fill::zeros);
  
  vec alpha(T, fill::ones), beta(T, fill::ones), s(T, fill::ones), v(T+1, fill::ones);
  
  for(int i=0; i<niter; i++) {
    // forward filter
    forward(y, m0, C0, FF, G, a, m, f, C, Q, R,
            alpha, beta, s, alpha0, beta0, delta_w, delta_v);
    // // backward sample
    backward(theta, a, join_rows(m0,m),join_slices(C0,C), R, G,
             join_cols(alpha0, alpha), join_cols(beta0, beta),
             delta_v, delta_w, v, s);
    
    // // save mcmc sweeps
    if(i>=burnin){
      mc_v.col(i-burnin) = v;
      mc_beta.col(i-burnin) = gpbeta;
      mc_theta.slice(i-burnin) = theta;
      mc_K.slice(i-burnin) = K;
      mc_W(i-burnin) = C*(1-delta_w)/delta_w;
    }
  }
  Rcpp::List out;
  out["phi_draws"] = phi_draws;
  out["theta_draws"] = theta_draws;
  return(out);
}

// /*** R
// simdata = function(N, m0, C0, V, W, X){
//   mu = matrix(0,nrow=length(m0),ncol=N)
//   y = rep(0,N)
//   mu[,1] = m0
//   y[1] = sum(mu[,1]) + mvtnorm::rmvnorm(1,rep(0,1),matrix(V))
//   for(t in 2:N){
//     e = mvtnorm::rmvnorm(1,rep(0,1),matrix(V))
//     g = t(mvtnorm::rmvnorm(1,matrix(0,nrow=length(m0)),W))
//     mu[,t] = mu[,t-1] + g
//     y[t] = X[t,]%*%mu[,t] + e
//   }
//   return(list("y"=y,"mu"=mu))
// }
// 
// N=500
// T = N
// m0 = matrix(c(5,0),nrow=2,ncol=1)
// X = matrix(0, nrow=T, ncol=2)
// X[,1] = 1
// X[,2] = rnorm(T,5,1)
// # X[,3] = 0
// V=matrix(.05);W=diag(.1,2)
// sim=simdata(N, m0, C0=diag(1,2), V=V, W=W, X)
// y1=sim$y
// mu1 = sim$mu
// matplot(t(mu1),type="l")
// y=y1#matrix(cbind(y1,y2),ncol=2)
// plot(y1)
// matplot(X*t(mu1),type="l",lwd=6,col="blue",add=0)
// p = nrow(m0)#ncol(y)
// T = length(y)#row(y)
// FF = array(matrix(c(1,1,1),nrow = 1),dim =c(1,3,T))
// FF = array(matrix(c(1),nrow = 2),dim =c(1,2,T))
// for(i in 1:T){
//   FF[,,i] = matrix(nrow=1,c(X[i,1],X[i,2]))
// }
// p=nrow(m0)
// start.time <- Sys.time()
// fit = dlm2(y = t(y),nsamples = 5000, burnin=5000/2, FF = FF, G=diag(p), C0 = 1*diag(p), m0 = matrix(c(5,0)),
//                       alpha0 = 1, beta0 = 1, delta_v=1, delta_w=.95)
// end.time <- Sys.time()
// (end.time-start.time)
// matplot(t(apply(fit$theta_draws,1:2,median)),type="l",lwd=4)
// 
// matplot(X*t(mu1),type="l",lwd=6,col="blue",add=0)
// matplot(X*t(apply(fit$theta_draws[,-1,],1:2,median)),type="l",lwd=4,add=T)
// 
// 
// matplot(fit$phi_draws[,,250],type="l",add=0)
// for(i in 250:500){
//   matplot(fit$phi_draws[,,i],type="l",add=T)
// }
// # temp = fit$theta_draws[1,,]
// # matplot(y,type="l")
// # apply(temp[,sample(1:ncol(temp), 100)],2,points,col = rgb(red = 0, green = 0, blue = 0, alpha = 0.05))
// # matplot(y,type="l")
// 
// 
// # hist(gibbs$mc_V[2,2,],breaks=100)
// # hist(gibbs$mc_W[2,2,],breaks=100)
// #
// # plot(gibbs$mc_V[2,2,],type="l")
// # plot(gibbs$mc_W[1,1,],type="l")
// #
// # sim=simdata(N, 0, 1, .2, .5)
// # y1=sim$y
// #
// # sim=simdata(N, 0, 1, .2, .5)
// # y2=sim$y
// #
// # y=cbind(y1,y2)
// 
// */




