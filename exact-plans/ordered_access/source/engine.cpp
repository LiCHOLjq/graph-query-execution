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
struct Counters{uint64_t edges=0,states=0,distances=0,sort_items=0,bytes=0,root_checks=0,pauses=0,heap_pops=0,root_digest=14695981039346656037ull;};
double dist(const Graph&g,int v,const Query&q){double s=0;for(int j=0;j<g.d;++j){double z=g.x[(size_t)v*g.d+j]-q.q[j];s+=z*z;}return s;}
vector<int>rank_set(const Graph&g,const Query&q,const vector<int>&eligible,Counters&c){vector<pair<double,int>>s;s.reserve(eligible.size());for(int v:eligible){s.emplace_back(dist(g,v,q),v);++c.distances;} c.sort_items+=s.size();partial_sort(s.begin(),s.begin()+min((size_t)q.k,s.size()),s.end());vector<int>ans;for(size_t i=0;i<min((size_t)q.k,s.size());++i)ans.push_back(s[i].second);return ans;}
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
 if(method=="ForwardTopK")return forward(g,q,c,false,quantum);
 Reverse r(g,q,c);vector<int>eligible;
 if(method=="ReverseRecursiveTopK"){
  for(int v=0;v<g.n;++v)if(v!=q.a){++c.root_checks;if(r.rec(r.h,v))eligible.push_back(v);}
  return rank_set(g,q,eligible,c);
 }
 if(method!="LazyShared"&&method!="LazyHeap")throw runtime_error("unknown method");
 vector<pair<double,int>>order;order.reserve(g.n-1);
 for(int v=0;v<g.n;++v)if(v!=q.a){order.emplace_back(dist(g,v,q),v);++c.distances;}
 c.sort_items=order.size();c.bytes+=order.capacity()*sizeof(pair<double,int>);
 // Both branches have the same exact total key and shared proof service.
 // No cardinality, truth set, or eligibility certificate enters this function.
 auto check=[&](int v){++c.root_checks;c.root_digest^=(uint64_t)v;c.root_digest*=1099511628211ull;
  if(r.iterative(v,quantum)){eligible.push_back(v);}
  return (int)eligible.size()==q.k;};
 if(method=="LazyHeap"){
  const greater<pair<double,int>> cmp{};
  make_heap(order.begin(),order.end(),cmp);
  while(!order.empty()){
   pop_heap(order.begin(),order.end(),cmp);int v=order.back().second;order.pop_back();++c.heap_pops;
   if(check(v))break;
  }
 }else{
  sort(order.begin(),order.end());for(const auto&dv:order)if(check(dv.second))break;
 }
 return eligible;
}
Graph generate(int n,int d,const string&family,uint64_t seed,bool fixed_topology=false){Graph g(n,d,2);mt19937_64 rng(seed*104729+937);normal_distribution<double>normal(0,1);for(double&z:g.x)z=normal(rng);if(fixed_topology)rng.seed(seed*104729+937);int deg=(family=="dense"?32:(family=="sparse"?1:4));
 for(int r=0;r<2;++r)for(int u=0;u<n-1;++u){if(family=="fanout"&&u<8){for(int v=0;v<n;++v)if((v+u+r)%2==0)g.add(u,v,r);}else{for(int j=0;j<deg;++j){int v;if(family=="fanin")v=(rng()%10<8)?rng()%16:rng()%n;else if(family=="sparse")v=(u+1+r)%n;else v=rng()%n;g.add(u,v,r);}}}
 return g;}
