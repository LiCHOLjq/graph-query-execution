#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
int main(){using C=std::chrono::steady_clock;std::vector<long long>positive;int zero=0;auto before=C::now();for(int i=0;i<100000;++i){auto after=C::now();auto d=std::chrono::duration_cast<std::chrono::nanoseconds>(after-before).count();before=after;if(d==0)++zero;else positive.push_back(d);}std::sort(positive.begin(),positive.end());std::cout<<"{\"clock_readings\":100001,\"adjacent_deltas\":100000,\"zero_deltas\":"<<zero<<",\"minimum_positive_ns\":"<<(positive.empty()?0:positive.front())<<",\"median_positive_ns\":"<<(positive.empty()?0:positive[positive.size()/2])<<",\"clock_is_steady\":"<<(C::is_steady?"true":"false")<<"}\n";}
