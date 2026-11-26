#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
using namespace std;
using Clock=chrono::steady_clock;
struct Edge{uint32_t u,v,r;};
struct Graph{
 int n,d,R; vector<Edge> edges; vector<vector<int>> out,in; vector<double>x;
 Graph(int n,int d,int R):n(n),d(d),R(R),out(n*R),in(n*R),x((size_t)n*d){}
 void add(int u,int v,int r){edges.push_back({(uint32_t)u,(uint32_t)v,(uint32_t)r});out[r*n+u].push_back(v);in[r*n+v].push_back(u);}
};
struct Query{int a,k; vector<int>w;vector<double>q;};
struct Counters{uint64_t edges=0,states=0,distances=0,sort_items=0,bytes=0,root_checks=0,pauses=0;};
double dist(const Graph&g,int v,const Query&q){double s=0;for(int j=0;j<g.d;++j){double z=g.x[(size_t)v*g.d+j]-q.q[j];s+=z*z;}return s;}
vector<int>rank_set(const Graph&g,const Query&q,const vector<int>&eligible,Counters&c){vector<pair<double,int>>s;s.reserve(eligible.size());for(int v:eligible){s.emplace_back(dist(g,v,q),v);++c.distances;} c.sort_items+=s.size();sort(s.begin(),s.end());vector<int>ans;for(size_t i=0;i<min((size_t)q.k,s.size());++i)ans.push_back(s[i].second);return ans;}
// Independent oracle: scan the flat edge relation at every layer; do not use
// either execution adjacency, its frontier routine, or any memoized predicate.
vector<int>oracle_set(const Graph&g,const Query&q){vector<unsigned char>prev(g.n),next(g.n);prev[q.a]=1;for(int r:q.w){fill(next.begin(),next.end(),0);for(const Edge&e:g.edges)if(e.r==(unsigned)r&&prev[e.u])next[e.v]=1;prev.swap(next);} vector<int>ans;for(int v=0;v<g.n;++v)if(v!=q.a&&prev[v])ans.push_back(v);return ans;}
struct ForwardState{
 const Graph&g;const Query&q;Counters&c;vector<int>front,next,stamp;int layer=0;size_t p=0,z=0;
 ForwardState(const Graph&g,const Query&q,Counters&c):g(g),q(q),c(c),front{q.a},stamp(g.n,-1){front.reserve(g.n);next.reserve(g.n);c.bytes=(front.capacity()+next.capacity()+stamp.capacity())*sizeof(int);}
 bool advance(int budget){while(layer<(int)q.w.size()){
  if(p==front.size()){front.swap(next);next.clear();p=z=0;++layer;if(front.empty()){layer=q.w.size();return true;}continue;}
  const auto&adj=g.out[q.w[layer]*g.n+front[p]];if(z==adj.size()){++p;z=0;continue;}
  int v=adj[z++];++c.edges;if(stamp[v]!=layer){stamp[v]=layer;next.push_back(v);++c.states;}
  if(--budget==0)return false;
 }return true;}
};
vector<int>forward(const Graph&g,const Query&q,Counters&c,bool resume,int quantum){
 if(resume){ForwardState state(g,q,c);while(!state.advance(quantum))++c.pauses;state.front.erase(remove(state.front.begin(),state.front.end(),q.a),state.front.end());return rank_set(g,q,state.front,c);}
 vector<int>front{q.a},next;vector<int>stamp(g.n,-1);c.bytes=sizeof(int)*(3ull*g.n);for(int i=0;i<(int)q.w.size();++i){next.clear();for(int u:front)for(int v:g.out[q.w[i]*g.n+u]){++c.edges;if(stamp[v]!=i){stamp[v]=i;next.push_back(v);++c.states;}}front.swap(next);if(front.empty())break;}front.erase(remove(front.begin(),front.end(),q.a),front.end());return rank_set(g,q,front,c);}
struct Reverse{
 const Graph&g;const Query&q;Counters&c;int n,h;vector<int8_t>memo;vector<uint32_t>epoch;uint32_t generation=1;
 struct Frame{int i,v;size_t cursor;int child;};vector<Frame>stack;
 Reverse(const Graph&g,const Query&q,Counters&c):g(g),q(q),c(c),n(g.n),h(q.w.size()),memo((h+1)*n,-1),epoch((h+1)*n,0){stack.reserve(h+1);c.bytes=memo.size()*sizeof(int8_t)+epoch.size()*sizeof(uint32_t)+(h+1)*sizeof(Frame);}
 int value(int i,int v){int id=i*n+v;return epoch[id]==generation?memo[id]:-1;}
 void save(int i,int v,int val){int id=i*n+v;if(epoch[id]!=generation){epoch[id]=generation;++c.states;}memo[id]=val;}
 bool rec(int i,int v){int known=value(i,v);if(known>=0)return known;if(i==0){save(i,v,v==q.a);return v==q.a;}for(int u:g.in[q.w[i-1]*n+v]){++c.edges;if(rec(i-1,u)){save(i,v,1);return true;}}save(i,v,0);return false;}
 bool advance(int quantum){uint64_t credit=0;while(!stack.empty()){
  auto &f=stack.back();int known=value(f.i,f.v);if(known>=0){stack.pop_back();continue;}
  if(f.i==0){save(0,f.v,f.v==q.a);stack.pop_back();continue;}
  if(f.child>=0){int val=value(f.i-1,f.child);if(val<0)throw runtime_error("pending child unfinished");f.child=-1;if(val){save(f.i,f.v,1);stack.pop_back();continue;}}
  const auto&adj=g.in[q.w[f.i-1]*n+f.v];if(f.cursor==adj.size()){save(f.i,f.v,0);stack.pop_back();continue;}
  int u=adj[f.cursor++];++c.edges;int val=value(f.i-1,u);if(val==1){save(f.i,f.v,1);stack.pop_back();}else if(val<0){f.child=u;stack.push_back({f.i-1,u,0,-1});}
  if(++credit==(unsigned)quantum)return false;
 }return true;}
 bool iterative(int root,int quantum){if(value(h,root)>=0)return value(h,root);stack.push_back({h,root,0,-1});while(!advance(quantum))++c.pauses;return value(h,root);}
};
vector<int>execute(const Graph&g,const Query&q,const string&method,Counters&c,int quantum){
 if(method=="ForwardGF")return forward(g,q,c,false,quantum);
 if(method=="ForwardResume")return forward(g,q,c,true,quantum);
 Reverse r(g,q,c);vector<int>eligible;
 if(method=="ReverseRecursive"||method=="ReverseResume"){
  for(int v=0;v<g.n;++v)if(v!=q.a){++c.root_checks;if(method=="ReverseRecursive"?r.rec(r.h,v):r.iterative(v,quantum))eligible.push_back(v);}
  return rank_set(g,q,eligible,c);
 }
 vector<pair<double,int>>order;order.reserve(g.n-1);for(int v=0;v<g.n;++v)if(v!=q.a){order.emplace_back(dist(g,v,q),v);++c.distances;}c.sort_items=order.size();c.bytes+=order.capacity()*sizeof(pair<double,int>);sort(order.begin(),order.end());
 for(const auto&dv:order){++c.root_checks;if(method=="LazySeparate")++r.generation;if(r.iterative(dv.second,quantum)){eligible.push_back(dv.second);if((int)eligible.size()==q.k)break;}}
 return eligible;
}
Graph generate(int n,int d,const string&family,uint64_t seed){Graph g(n,d,2);mt19937_64 rng(seed*104729+937);normal_distribution<double>normal(0,1);for(double&z:g.x)z=normal(rng);rng.seed(seed*104729+937);int deg=(family=="dense"?32:(family=="sparse"?1:4));
 for(int r=0;r<2;++r)for(int u=0;u<n-1;++u){if(family=="fanout"&&u<8){for(int v=0;v<n;++v)if((v+u+r)%2==0)g.add(u,v,r);}else{for(int j=0;j<deg;++j){int v;if(family=="fanin")v=(rng()%10<8)?rng()%16:rng()%n;else if(family=="sparse")v=(u+1+r)%n;else v=rng()%n;g.add(u,v,r);}}}
 return g;}
void header(ostream&o){o<<"dataset,family,seed,n,d,m,h,k,anchor,anchor_degree,eligible,query_id,rep,method,ms,edge_examinations,state_entries,distances,sort_items,root_checks,pauses,state_bytes,exact\n";}
void bench(const Graph&g,Query q,const string&dataset,const string&family,int seed,int qid,ostream&out,int reps,int quantum,const vector<string>&methods){
 auto truthset=oracle_set(g,q);Counters oc;auto truth=rank_set(g,q,truthset,oc);auto emit=[&](int rep,const string&m){Counters c;auto start=Clock::now();auto got=execute(g,q,m,c,quantum);auto stop=Clock::now();if(got!=truth)throw runtime_error("exact mismatch "+m+" query "+to_string(qid));if(rep<0)return;double ms=chrono::duration<double,milli>(stop-start).count();out<<dataset<<','<<family<<','<<seed<<','<<g.n<<','<<g.d<<','<<g.edges.size()<<','<<q.w.size()<<','<<q.k<<','<<q.a<<','<<g.out[q.w.front()*g.n+q.a].size()<<','<<truthset.size()<<','<<qid<<','<<rep<<','<<m<<','<<setprecision(10)<<ms<<','<<c.edges<<','<<c.states<<','<<c.distances<<','<<c.sort_items<<','<<c.root_checks<<','<<c.pauses<<','<<c.bytes<<",1\n";};
 for(auto&m:methods)emit(-1,m);for(int rep=0;rep<reps;++rep){auto shuffled=methods;mt19937 gen(seed*10007+qid*19+rep);shuffle(shuffled.begin(),shuffled.end(),gen);for(auto&m:shuffled)emit(rep,m);}out.flush();
}
int main(int argc,char**argv){try{
 string mode=argc>1?argv[1]:"test";vector<string>methods={"ForwardGF","ForwardResume","ReverseRecursive","ReverseResume","LazyShared","LazySeparate"};
 if(mode=="test"){
  for(int seed=0;seed<12;++seed){Graph g=generate(35,4,seed%2?"dense":"sparse",seed);if(seed==0)fill(g.x.begin(),g.x.end(),0);for(int a: {0,17,34})for(int h=1;h<=6;++h)for(int k:{1,10,100}){Query q{a,k,{},vector<double>(4,0)};for(int i=0;i<h;++i)q.w.push_back(i%2);auto eligible=oracle_set(g,q);Counters c;auto truth=rank_set(g,q,eligible,c);for(int quantum:{1,2,7,256})for(auto&m:methods){Counters cc;if(execute(g,q,m,cc,quantum)!=truth)throw runtime_error("test mismatch");}}}
  cout<<"PASS: 12 graphs x 3 anchors x 6 lengths x 3 k x 4 quanta x 6 methods = 15552 exact comparisons; explicit all-zero vector ties, empty/underfilled included\n";return 0;
 }
 ofstream out(argc>2?argv[2]:"raw.csv");header(out);int qid=0;int startqid=(mode=="synthetic"&&argc>3)?stoi(argv[3]):0;
 if(mode=="synthetic"){
  for(int n:{1000,10000,100000})for(string family:{"sparse","uniform","dense","fanout","fanin"})for(int seed=0;seed<10;++seed){auto g=generate(n,64,family,seed);for(int h:{2,4,6})for(int k:{1,10,100}){int a=(family=="fanout"?seed%8:(seed*7919+37)%(n-1));Query q{a,k,{},vector<double>(g.d)};for(int i=0;i<h;++i)q.w.push_back(i%2);mt19937 gen(seed*1697+h);normal_distribution<double>norm(0,1);for(double&v:q.q)v=norm(gen);if(qid>=startqid)bench(g,q,"synthetic",family,seed,qid,out,3,256,methods);++qid;}
   Query empty{n-1,10,{0,1,0,1},vector<double>(g.d,0)};auto empty_methods=methods;if(n>1000)empty_methods.pop_back();if(qid>=startqid)bench(g,empty,"synthetic",family,seed,qid,out,3,256,empty_methods);++qid;cerr<<"done n="<<n<<" family="<<family<<" seed="<<seed<<" queries="<<qid<<'\n';}
 }else if(mode=="sensitivity"){
  for(int d:{16,256,768})for(string family:{"uniform","dense"})for(int seed=0;seed<10;++seed){auto g=generate(10000,d,family,seed);for(int h:{2,6}){Query q{(seed*7919+37)%(g.n-1),10,{},vector<double>(d,0)};for(int i=0;i<h;++i)q.w.push_back(i%2);bench(g,q,"dimension",family,seed,qid++,out,3,256,methods);}}
 }else if(mode=="real"){
  if(argc<5)throw runtime_error("real output.csv graph.bin queries.txt");ifstream f(argv[3],ios::binary);uint32_t n,d,m;f.read((char*)&n,4);f.read((char*)&d,4);f.read((char*)&m,4);vector<Edge>edges(m);f.read((char*)edges.data(),m*sizeof(Edge));int R=0;for(auto&e:edges)R=max(R,(int)e.r+1);Graph g(n,d,R);for(auto&e:edges)g.add(e.u,e.v,e.r);f.read((char*)g.x.data(),g.x.size()*8);if(!f)throw runtime_error("invalid binary");ifstream qs(argv[4]);int a,h;while(qs>>a>>h){Query q{a,0,vector<int>(h),vector<double>(d)};for(int&i:q.w)qs>>i;qs>>q.k;for(double&v:q.q)qs>>v;if(!qs)throw runtime_error("invalid query");bench(g,q,"FB15k237","real",qid%10,qid,out,3,256,methods);++qid;cerr<<"real query="<<qid<<'\n';}
 }else throw runtime_error("unknown mode");
 cerr<<"ALL_EXACT queries="<<qid<<" timed="<<qid*methods.size()*3<<'\n';
 }catch(exception&e){cerr<<e.what()<<'\n';return 1;}return 0;}
