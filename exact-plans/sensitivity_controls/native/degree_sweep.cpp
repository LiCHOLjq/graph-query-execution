// The unchanged NATIVE strong execution kernels and flat-edge oracle are retained.
#define main frozen_strong_main
#include "strong_base.cpp"
#undef main

Graph generate_degree(int n,int d,int degree,uint64_t seed){
 Graph g(n,d,2);mt19937_64 rng(seed*104729+937);normal_distribution<double>normal(0,1);
 for(double&z:g.x)z=normal(rng);
 for(int r=0;r<2;++r)for(int u=0;u<n-1;++u)for(int j=0;j<degree;++j)g.add(u,rng()%n,r);
 return g;
}
int main(int argc,char**argv){try{
 if(argc>1&&string(argv[1])=="test")return frozen_strong_main(argc,argv);
 ofstream out(argc>1?argv[1]:"degree_sweep_raw.csv");header(out);int qid=0;
 vector<string>methods={"ForwardTopK","ReverseRecursiveTopK","LazyShared"};
 for(int n:{10000,100000})for(int degree:{4,6,8,10,12,16,24,32})for(int seed=0;seed<10;++seed){
  Graph g=generate_degree(n,64,degree,seed);
  for(int h:{2,4,6})for(int k:{1,10,100}){
   Query q{(seed*7919+37)%(n-1),k,{},vector<double>(g.d)};
   for(int i=0;i<h;++i)q.w.push_back(i%2);
   mt19937 gen(seed*1697+h);normal_distribution<double>norm(0,1);for(double&v:q.q)v=norm(gen);
   bench(g,q,"degree_sweep","degree_"+to_string(degree),seed,qid++,out,3,256,methods);
  }
  cerr<<"done n="<<n<<" degree="<<degree<<" seed="<<seed<<" queries="<<qid<<'\n';
 }
 cerr<<"ALL_EXACT queries="<<qid<<" timed="<<qid*methods.size()*3<<'\n';
 }catch(exception&e){cerr<<e.what()<<'\n';return 1;}return 0;}
