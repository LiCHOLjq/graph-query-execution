// MATCHED changes only the sweep and adds untimed, explicitly separated diagnostics.
// All timed algorithms, counters, scalar distances and oracle remain unchanged.
#define main frozen_strong_main
#include "strong_base.cpp"
#undef main
#include <cstring>
#include <map>

Graph generate_degree(int n,int d,int degree,uint64_t seed){
 Graph g(n,d,2);mt19937_64 rng(seed*104729+937);normal_distribution<double>normal(0,1);
 for(double&z:g.x)z=normal(rng);
 for(int r=0;r<2;++r)for(int u=0;u<n-1;++u)for(int j=0;j<degree;++j)g.add(u,rng()%n,r);
 return g;
}
uint64_t word_hash(uint64_t h,uint64_t word){for(int i=0;i<8;++i){h^=(word>>(8*i))&255;h*=1099511628211ull;}return h;}
uint64_t ids_hash(const vector<int>&a){uint64_t h=14695981039346656037ull;for(int v:a)h=word_hash(h,v);return h;}
uint64_t doubles_hash(const vector<double>&a){uint64_t h=14695981039346656037ull;for(double v:a){uint64_t bits;memcpy(&bits,&v,8);h=word_hash(h,bits);}return h;}
uint64_t graph_hash(const Graph&g){uint64_t h=14695981039346656037ull;for(const auto&e:g.edges){h=word_hash(h,e.u);h=word_hash(h,e.v);h=word_hash(h,e.r);}return h;}

void diagnostics(const Graph&g,Query q,int seed,ostream&ds,ostream&trace){
 const auto es=oracle_set(g,q);vector<unsigned char>is(g.n);for(int v:es)is[v]=1;
 vector<pair<double,int>> order;for(int v=0;v<g.n;++v)if(v!=q.a)order.emplace_back(dist(g,v,q),v);sort(order.begin(),order.end());
 Counters c;Reverse r(g,q,c);vector<int>found;int first=0,rank=0;
 uint64_t first_edges=0,first_states=0;
 for(const auto&dv:order){
  ++rank;auto pe=c.edges,ps=c.states;bool hit=r.iterative(dv.second,256);
  if(hit!=(bool)is[dv.second])throw runtime_error("diagnostic membership mismatch");
  if(rank==1){first_edges=c.edges;first_states=c.states;}
  if(hit){found.push_back(dv.second);if(!first)first=rank;}
  trace<<seed<<','<<rank<<','<<dv.second<<','<<hit<<','<<c.edges-pe<<','<<c.states-ps<<','<<c.edges<<','<<c.states<<','<<found.size()<<'\n';
  int k=found.size();
  if(hit&&(k==1||k==2||k==3||k==5||k==10||k==100)){
   Query tq=q;tq.k=k;Counters oc;auto truth=rank_set(g,tq,es,oc);
   if(found!=truth)throw runtime_error("diagnostic ordered answer mismatch");
   ds<<seed<<','<<g.n<<','<<g.d<<','<<g.edges.size()<<",8,6,"<<k<<','<<q.a<<','<<es.size()<<','<<setprecision(17)<<double(es.size())/(g.n-1)<<','<<first<<','<<rank<<','<<first_edges<<','<<first_states<<','<<c.edges<<','<<c.states<<','<<ids_hash(es)<<','<<ids_hash(truth)<<','<<graph_hash(g)<<','<<doubles_hash(g.x)<<','<<doubles_hash(q.q)<<'\n';
  }
  if(k==100)break;
 }
 if(found.size()!=100)throw runtime_error("unexpected underfilled diagnostic");
}
int main(int argc,char**argv){try{
 if(argc>1&&string(argv[1])=="test"){
  int code=frozen_strong_main(argc,argv);if(code)return code;
  const vector<string>methods={"ForwardTopK","ReverseRecursiveTopK","LazyShared"};
  int checks=0;for(int seed=0;seed<6;++seed){auto g=generate_degree(37,4,8,seed);if(seed==0)fill(g.x.begin(),g.x.end(),0);
   for(int a:{0,17,36})for(int k:{1,2,3,5,10,100}){Query q{a,k,{0,1,0,1,0,1},vector<double>(4,0)};auto eligible=oracle_set(g,q);Counters oc;auto truth=rank_set(g,q,eligible,oc);
    for(auto&m:methods){Counters c;if(execute(g,q,m,c,256)!=truth)throw runtime_error("new-k test mismatch");++checks;}
   }
  }cout<<"PASS: "<<checks<<" additional degree-8/new-k exact comparisons, all-zero ties and empty-anchor included\n";return 0;
 }
 ofstream out(argc>1?argv[1]:"k_sweep_raw.csv");header(out);
 ofstream ds("query_diagnostics.csv"),trace("root_trace.csv");
 ds<<"seed,n,d,graph_edges,degree,h,k,anchor,eligible,selectivity,first_eligible_rank,kth_eligible_rank,first_root_edges,first_root_states,lazy_prefix_edges,lazy_prefix_states,eligible_fnv64,answer_fnv64,graph_fnv64,vectors_fnv64,query_fnv64\n";
 trace<<"seed,rank,vertex,eligible,closure_new_edges,closure_new_states,cumulative_edges,cumulative_states,found\n";
 int qid=0;const vector<string>methods={"ForwardTopK","ReverseRecursiveTopK","LazyShared"};
 for(int seed=0;seed<10;++seed){
  Graph g=generate_degree(100000,64,8,seed);
  for(int k:{1,2,3,5,10,100}){
   Query q{(seed*7919+37)%(g.n-1),k,{0,1,0,1,0,1},vector<double>(g.d)};
   mt19937 gen(seed*1697+6);normal_distribution<double>norm(0,1);for(double&v:q.q)v=norm(gen);
   bench(g,q,"matched_boundary_k","degree_8",seed,qid++,out,3,256,methods);
  }
  // After all measured k values for this seed; never included in algorithm latency.
  Query q{(seed*7919+37)%(g.n-1),100,{0,1,0,1,0,1},vector<double>(g.d)};
  mt19937 gen(seed*1697+6);normal_distribution<double>norm(0,1);for(double&v:q.q)v=norm(gen);
  diagnostics(g,q,seed,ds,trace);
  cerr<<"done seed="<<seed<<" queries="<<qid<<'\n';
 }
 cerr<<"ALL_EXACT queries="<<qid<<" plan_cells="<<qid*methods.size()<<" timed="<<qid*methods.size()*3<<'\n';
 }catch(exception&e){cerr<<e.what()<<'\n';return 1;}return 0;}
