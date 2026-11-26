#include "engine.cpp"
#include <cstring>
#include <filesystem>
#include <map>
#include <sstream>
namespace fs=std::filesystem;
const vector<string> METHODS={"ForwardTopK","ReverseRecursiveTopK","LazyShared","LazyHeap"};
uint64_t word_hash(uint64_t h,uint64_t word){for(int i=0;i<8;++i){h^=(word>>(8*i))&255;h*=1099511628211ull;}return h;}
uint64_t ids_hash(const vector<int>&a){uint64_t h=14695981039346656037ull;for(int v:a)h=word_hash(h,v);return h;}
uint64_t doubles_hash(const vector<double>&a){uint64_t h=14695981039346656037ull;for(double v:a){uint64_t bits;memcpy(&bits,&v,8);h=word_hash(h,bits);}return h;}
struct Identity{uint64_t graph, vectors;};
Identity identity(const Graph&g){uint64_t h=14695981039346656037ull;for(const auto&e:g.edges){h=word_hash(h,e.u);h=word_hash(h,e.v);h=word_hash(h,e.r);}return{h,doubles_hash(g.x)};}
Graph generate_degree(int n,int d,int degree,uint64_t seed){Graph g(n,d,2);mt19937_64 rng(seed*104729+937);normal_distribution<double>normal(0,1);for(double&z:g.x)z=normal(rng);for(int r=0;r<2;++r)for(int u=0;u<n-1;++u)for(int j=0;j<degree;++j)g.add(u,rng()%n,r);return g;}
Query query(int n,int d,int seed,int h,int k,bool origin=false,int anchor=-1){Query q{anchor<0?(seed*7919+37)%(n-1):anchor,k,{},vector<double>(d,0)};for(int i=0;i<h;++i)q.w.push_back(i%2);if(!origin){mt19937 gen(seed*1697+h);normal_distribution<double>norm(0,1);for(double&v:q.q)v=norm(gen);}return q;}
struct Spec{int id;Query q;string condition;};
struct Outputs{
 ofstream raw,queries;uint64_t executions=0,warmups=0;int configs=0;
 Outputs(const fs::path&dir,const string&name){fs::create_directories(dir);auto rp=dir/(name+"_raw.csv"),qp=dir/(name+"_queries.csv");if(fs::exists(rp)||fs::exists(qp))throw runtime_error("refusing to overwrite measured cohort "+name);raw.open(rp);queries.open(qp);if(!raw||!queries)throw runtime_error("output open failed");
 raw<<"cohort,family,condition,seed,n,d,m,h,k,anchor,anchor_degree,eligible,query_id,rep,method,ms,edge_examinations,state_entries,distances,ordering_items,root_checks,pauses,state_bytes,heap_pops,root_digest,answer_fnv64,exact\n";
 queries<<"cohort,family,condition,seed,n,d,m,h,k,anchor,eligible,query_id,word,graph_fnv64,vectors_fnv64,query_fnv64,eligible_fnv64,answer_fnv64\n";}
};
bool same_lazy(const Counters&a,const Counters&b){return a.edges==b.edges&&a.states==b.states&&a.distances==b.distances&&a.root_checks==b.root_checks&&a.root_digest==b.root_digest&&a.pauses==b.pauses&&a.sort_items==b.sort_items;}
void bench(const Graph&g,const Identity&ii,const Query&q,const string&cohort,const string&family,const string&condition,int seed,int qid,Outputs&o){
 // The graph oracle scans flat relation edges; execution plans never see truthset.
 auto truthset=oracle_set(g,q);Counters oc;auto truth=rank_set(g,q,truthset,oc);uint64_t ah=ids_hash(truth);
 o.queries<<cohort<<','<<family<<','<<condition<<','<<seed<<','<<g.n<<','<<g.d<<','<<g.edges.size()<<','<<q.w.size()<<','<<q.k<<','<<q.a<<','<<truthset.size()<<','<<qid<<',';
 for(size_t j=0;j<q.w.size();++j){if(j)o.queries<<'|';o.queries<<q.w[j];}
 o.queries<<','<<ii.graph<<','<<ii.vectors<<','<<doubles_hash(q.q)<<','<<ids_hash(truthset)<<','<<ah<<'\n';
 for(int rep=-1;rep<3;++rep){
  auto order=METHODS;mt19937 gen(20261028u+seed*10007u+qid*19u+(rep+1)*1009u);shuffle(order.begin(),order.end(),gen);
  map<string,Counters>counters;map<string,vector<int>>answers;
  for(const string&method:order){Counters c;auto start=Clock::now();auto got=execute(g,q,method,c,256);auto stop=Clock::now();
   if(got!=truth)throw runtime_error("oracle mismatch "+cohort+" q"+to_string(qid)+" "+method);
   if(method=="LazyHeap"&&c.heap_pops!=c.root_checks)throw runtime_error("heap pop mismatch");
   counters[method]=c;answers[method]=got;
   if(rep<0){++o.warmups;continue;}
   double ms=chrono::duration<double,milli>(stop-start).count();
   o.raw<<cohort<<','<<family<<','<<condition<<','<<seed<<','<<g.n<<','<<g.d<<','<<g.edges.size()<<','<<q.w.size()<<','<<q.k<<','<<q.a<<','<<g.out[q.w.front()*g.n+q.a].size()<<','<<truthset.size()<<','<<qid<<','<<rep<<','<<method<<','<<setprecision(12)<<ms<<','<<c.edges<<','<<c.states<<','<<c.distances<<','<<c.sort_items<<','<<c.root_checks<<','<<c.pauses<<','<<c.bytes<<','<<c.heap_pops<<','<<c.root_digest<<','<<ah<<",1\n";++o.executions;
  }
  if(answers["LazyShared"]!=answers["LazyHeap"]||!same_lazy(counters["LazyShared"],counters["LazyHeap"]))throw runtime_error("heap/full-sort prefix mismatch "+cohort+" q"+to_string(qid));
 }
 ++o.configs;o.raw.flush();o.queries.flush();
}
void batch(const Graph&g,const Identity&ii,vector<Spec>specs,const string&cohort,const string&family,int seed,Outputs&o){mt19937 gen(20261028u+seed*7919u+(specs.empty()?0:specs[0].id));shuffle(specs.begin(),specs.end(),gen);for(const auto&s:specs)bench(g,ii,s.q,cohort,family,s.condition,seed,s.id,o);}
void unit_tests(){uint64_t checks=0,heap_orders=0;
 for(int seed=0;seed<12;++seed){auto g=generate(37,4,seed%2?"dense":"sparse",seed);if(seed<2)fill(g.x.begin(),g.x.end(),0);
  for(int a:{0,17,36})for(int h=1;h<=6;++h)for(int k:{1,2,3,5,10,100}){auto q=query(g.n,g.d,seed,h,k,true,a);auto truthset=oracle_set(g,q);Counters oc;auto truth=rank_set(g,q,truthset,oc);
   vector<pair<double,int>>sorted;for(int v=0;v<g.n;++v)if(v!=q.a)sorted.emplace_back(dist(g,v,q),v);sort(sorted.begin(),sorted.end());auto heap=sorted;reverse(heap.begin(),heap.end());greater<pair<double,int>>cmp;make_heap(heap.begin(),heap.end(),cmp);vector<pair<double,int>>popped;while(!heap.empty()){pop_heap(heap.begin(),heap.end(),cmp);popped.push_back(heap.back());heap.pop_back();}if(popped!=sorted)throw runtime_error("complete heap order mismatch");++heap_orders;
   for(int quantum:{1,2,7,256}){map<string,Counters>cs;for(const auto&m:METHODS){Counters c;auto answer=execute(g,q,m,c,quantum);if(answer!=truth)throw runtime_error("unit exact mismatch");if(seed<2){auto expected=truthset;sort(expected.begin(),expected.end());expected.resize(min(expected.size(),(size_t)k));if(answer!=expected)throw runtime_error("explicit tie mismatch");}cs[m]=c;++checks;}if(!same_lazy(cs["LazyShared"],cs["LazyHeap"]))throw runtime_error("unit lazy prefix mismatch");}
  }
 }
 cout<<"PASS exact_plan_comparisons="<<checks<<" complete_heap_order_checks="<<heap_orders<<" quanta=1,2,7,256; dense/sparse all-zero ties, cycles, underfilled and empty included\n";
}
int main(int argc,char**argv){try{
 string mode=argc>1?argv[1]:"test";if(mode=="test"){unit_tests();return 0;}fs::path outdir=argc>2?argv[2]:"results";Outputs o(outdir,mode);
 if(mode=="main"||mode=="pilot"){
  vector<tuple<int,string,int,int>>graphs;int base=0;
  for(int n:{1000,10000,100000})for(string family:{"sparse","uniform","dense","fanout","fanin"})for(int seed=0;seed<10;++seed){if(mode=="main"||(n==100000&&seed==0&&(family=="dense"||family=="fanout")))graphs.emplace_back(n,family,seed,base);base+=10;}
  mt19937 order(20261028);shuffle(graphs.begin(),graphs.end(),order);
  for(auto [n,family,seed,baseid]:graphs){auto g=generate(n,64,family,seed);auto ii=identity(g);vector<Spec>specs;int j=0;for(int h:{2,4,6})for(int k:{1,10,100})specs.push_back({baseid+j++,query(n,64,seed,h,k,false,family=="fanout"?seed%8:-1),"gaussian"});specs.push_back({baseid+9,query(n,64,seed,4,10,true,n-1),"origin_empty"});batch(g,ii,specs,mode,family,seed,o);cerr<<mode<<" completed_graph n="<<n<<" family="<<family<<" seed="<<seed<<" configs="<<o.configs<<'\n';}
 }else if(mode=="degree"){
  vector<tuple<int,int,int,int>>graphs;int base=0;for(int n:{10000,100000})for(int degree:{4,6,8,10,12,16,24,32})for(int seed=0;seed<10;++seed){graphs.emplace_back(n,degree,seed,base);base+=9;}mt19937 order(20261029);shuffle(graphs.begin(),graphs.end(),order);
  for(auto [n,degree,seed,baseid]:graphs){auto g=generate_degree(n,64,degree,seed);auto ii=identity(g);vector<Spec>specs;int j=0;for(int h:{2,4,6})for(int k:{1,10,100})specs.push_back({baseid+j++,query(n,64,seed,h,k),"gaussian"});batch(g,ii,specs,mode,"degree_"+to_string(degree),seed,o);cerr<<mode<<" completed_graph n="<<n<<" degree="<<degree<<" seed="<<seed<<" configs="<<o.configs<<'\n';}
 }else if(mode=="matched"){
  vector<int>seeds(10);iota(seeds.begin(),seeds.end(),0);mt19937 order(20261030);shuffle(seeds.begin(),seeds.end(),order);
  for(int seed:seeds){auto g=generate_degree(100000,64,8,seed);auto ii=identity(g);vector<Spec>specs;int j=0;for(int k:{1,2,3,5,10,100})specs.push_back({seed*6+j++,query(g.n,g.d,seed,6,k),"gaussian"});batch(g,ii,specs,mode,"degree_8",seed,o);cerr<<mode<<" completed_seed="<<seed<<" configs="<<o.configs<<'\n';}
 }else if(mode=="dimension"){
  vector<tuple<int,string,int,int>>graphs;int base=0;for(int d:{16,256,768})for(string family:{"uniform","dense"})for(int seed=0;seed<10;++seed){graphs.emplace_back(d,family,seed,base);base+=4;}mt19937 order(20261031);shuffle(graphs.begin(),graphs.end(),order);
  for(auto [d,family,seed,baseid]:graphs){auto g=generate(10000,d,family,seed,true);auto ii=identity(g);vector<Spec>specs;int j=0;for(bool origin:{true,false})for(int h:{2,6})specs.push_back({baseid+j++,query(g.n,d,seed,h,10,origin),origin?"origin":"gaussian"});batch(g,ii,specs,mode,family,seed,o);cerr<<mode<<" completed_graph d="<<d<<" family="<<family<<" seed="<<seed<<" configs="<<o.configs<<'\n';}
 }else if(mode=="real"){
  fs::path ip=argc>3?argv[3]:"inputs";ifstream f(ip/"graph.bin",ios::binary);uint32_t n,d,m;f.read((char*)&n,4);f.read((char*)&d,4);f.read((char*)&m,4);vector<Edge>es(m);f.read((char*)es.data(),m*sizeof(Edge));int R=0;for(const auto&e:es)R=max(R,(int)e.r+1);Graph g(n,d,R);for(const auto&e:es)g.add(e.u,e.v,e.r);f.read((char*)g.x.data(),g.x.size()*8);if(!f)throw runtime_error("real input binary invalid");auto ii=identity(g);ifstream qs(ip/"queries.txt");int a,h,id=0;vector<Spec>specs;while(qs>>a>>h){Query q{a,0,vector<int>(h),vector<double>(d)};for(int&r:q.w)qs>>r;qs>>q.k;for(double&x:q.q)qs>>x;if(!qs)throw runtime_error("real query invalid");specs.push_back({id++,q,"text"});}mt19937 order(20261101);shuffle(specs.begin(),specs.end(),order);for(const auto&s:specs)bench(g,ii,s.q,mode,"FB15k237",s.condition,s.id%10,s.id,o);
 }else throw runtime_error("unknown cohort");
 cerr<<"ALL_EXACT cohort="<<mode<<" configurations="<<o.configs<<" timed="<<o.executions<<" warmups="<<o.warmups<<" heap_fullsort_prefix_equal=1\n";
 }catch(const exception&e){cerr<<e.what()<<'\n';return 1;}return 0;}
