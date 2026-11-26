// Frozen NATIVE six-method source supplies the original graph and full-sort kernel.
#define main frozen_landscape_main
#include "landscape_base.cpp"
#undef main

vector<int>forward_reserved(const Graph&g,const Query&q,Counters&c){
 vector<int>front{q.a},next;vector<int>stamp(g.n,-1);
 front.reserve(g.n);next.reserve(g.n);
 c.bytes=(front.capacity()+next.capacity()+stamp.capacity())*sizeof(int);
 for(int i=0;i<(int)q.w.size();++i){next.clear();for(int u:front)for(int v:g.out[q.w[i]*g.n+u]){
  ++c.edges;if(stamp[v]!=i){stamp[v]=i;next.push_back(v);++c.states;}
 }front.swap(next);if(front.empty())break;}
 front.erase(remove(front.begin(),front.end(),q.a),front.end());return rank_set(g,q,front,c);
}
vector<int>exec_alloc(const Graph&g,const Query&q,const string&m,Counters&c,int quantum){
 if(m=="ForwardReserved")return forward_reserved(g,q,c);
 return forward(g,q,c,m=="ForwardResume",quantum);
}
void bench_alloc(const Graph&g,Query q,const string&family,int seed,int qid,ostream&out){
 auto truthset=oracle_set(g,q);Counters oc;auto truth=rank_set(g,q,truthset,oc);
 vector<string>methods={"ForwardGF","ForwardReserved","ForwardResume"};
 auto emit=[&](int rep,const string&m){Counters c;auto start=Clock::now();auto got=exec_alloc(g,q,m,c,256);auto stop=Clock::now();
  if(got!=truth)throw runtime_error("exact mismatch "+m+" query "+to_string(qid));if(rep<0)return;
  double ms=chrono::duration<double,milli>(stop-start).count();
  out<<"allocation,"<<family<<','<<seed<<','<<g.n<<','<<g.d<<','<<g.edges.size()<<','<<q.w.size()<<','<<q.k<<','<<q.a<<','<<g.out[q.w.front()*g.n+q.a].size()<<','<<truthset.size()<<','<<qid<<','<<rep<<','<<m<<','<<setprecision(10)<<ms<<','<<c.edges<<','<<c.states<<','<<c.distances<<','<<c.sort_items<<','<<c.root_checks<<','<<c.pauses<<','<<c.bytes<<",1\n";
 };
 for(auto&m:methods)emit(-1,m);for(int rep=0;rep<3;++rep){auto shuffled=methods;mt19937 gen(seed*10007+qid*19+rep);shuffle(shuffled.begin(),shuffled.end(),gen);for(auto&m:shuffled)emit(rep,m);}out.flush();
}
int main(int argc,char**argv){try{
 if(argc>1&&string(argv[1])=="test"){
  int checks=0;for(int seed=0;seed<12;++seed){Graph g=generate(35,4,seed%2?"dense":"sparse",seed);if(seed==0)fill(g.x.begin(),g.x.end(),0);
   for(int a:{0,17,34})for(int h=1;h<=6;++h)for(int k:{1,10,100}){Query q{a,k,{},vector<double>(4,0)};for(int i=0;i<h;++i)q.w.push_back(i%2);
    auto eligible=oracle_set(g,q);Counters c;auto truth=rank_set(g,q,eligible,c);
    for(int quantum:{1,2,7,256})for(string m:{"ForwardGF","ForwardReserved","ForwardResume"}){Counters cc;if(exec_alloc(g,q,m,cc,quantum)!=truth)throw runtime_error("test mismatch");++checks;}
   }
  }cout<<"PASS "<<checks<<" independent-oracle exact comparisons\n";return 0;
 }
 ofstream out(argc>1?argv[1]:"allocation_raw.csv");header(out);int qid=0;
 for(int n:{1000,10000,100000})for(string family:{"sparse","uniform","dense","fanout","fanin"})for(int seed=0;seed<10;++seed){
  Graph g=generate(n,64,family,seed);
  for(int h:{2,4,6})for(int k:{1,10,100}){
   Query q{family=="fanout"?seed%8:(seed*7919+37)%(n-1),k,{},vector<double>(g.d)};for(int i=0;i<h;++i)q.w.push_back(i%2);
   mt19937 gen(seed*1697+h);normal_distribution<double>norm(0,1);for(double&v:q.q)v=norm(gen);
   bench_alloc(g,q,family,seed,qid++,out);
  }
  Query empty{n-1,10,{0,1,0,1},vector<double>(g.d,0)};bench_alloc(g,empty,family,seed,qid++,out);
  cerr<<"done n="<<n<<" family="<<family<<" seed="<<seed<<" queries="<<qid<<'\n';
 }
 cerr<<"ALL_EXACT queries="<<qid<<" timed="<<qid*9<<'\n';
 }catch(exception&e){cerr<<e.what()<<'\n';return 1;}return 0;}
