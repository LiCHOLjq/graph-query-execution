#define main included_benchmark_main
#ifdef STRONG_TEST
#include "strong_baseline.cpp"
#else
#include "landscape.cpp"
#endif
#undef main
int main(){
#ifdef STRONG_TEST
 vector<string> methods={"ForwardTopK","ReverseRecursiveTopK","LazyShared"};
#else
 vector<string> methods={"ForwardGF","ForwardResume","ReverseRecursive","ReverseResume","LazyShared","LazySeparate"};
#endif
 size_t checks=0;
 for(int seed=0;seed<10;++seed){auto g=generate(35,4,"dense",seed);fill(g.x.begin(),g.x.end(),0);
  for(int h=1;h<=6;++h)for(int k:{1,10,100}){Query q{0,k,{},vector<double>(4,0)};for(int i=0;i<h;++i)q.w.push_back(i%2);
   auto truth=oracle_set(g,q);sort(truth.begin(),truth.end());truth.resize(min(truth.size(),(size_t)k));
   for(int quantum:{1,2,7,256})for(auto&m:methods){Counters c;if(execute(g,q,m,c,quantum)!=truth)throw runtime_error("dense exact tie mismatch");++checks;}
  }
 }
 cout<<"PASS: "<<checks<<" dense all-zero vector tests; expected answer explicitly ascending eligible IDs; quanta1,2,7,256\n";
}
